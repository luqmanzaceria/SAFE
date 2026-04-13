#!/usr/bin/env python3
"""
LSTM Failure Detector
=====================

Replaces the MLP+running-mean with a causal LSTM that naturally models
temporal dependencies — i.e., what the robot did *earlier* influences the
current failure probability.

This is the architecture used by SAFE (IndepModel=MLP is their baseline;
their stronger model uses LSTM).  This script provides a direct apples-to-apples
comparison against SAFE's LSTM on your data.

Why LSTM can outperform MLP+running-mean
-----------------------------------------
  MLP+running-mean:  treats each timestep independently, then averages.
                     "Is this hidden state failure-like?" — no memory.
  LSTM:              maintains a hidden state h_t = f(h_{t-1}, x_t).
                     "Given everything I've seen so far, is this a failure?"
                     Can detect patterns like "the robot reached for the object
                     at step 10 but never moved again" (sequential reasoning).

Three variants trained for comparison
--------------------------------------
  1. Base MLP + running mean            (your existing baseline)
  2. LSTM + running mean                (causal temporal modelling)
  3. LSTM + CLIP task conditioning      (if CLIP available; optional)

Outputs (output_dir/)
---------------------
  training_loss.png         all training curves
  roc_comparison.png        ROC: MLP vs LSTM (vs LSTM+CLIP if enabled)
  prc_comparison.png        PRC: same
  detection_time.png        Detection tradeoff curve
  score_curves.png          Score trajectories (success / failure)
  lstm_hidden_pca.png       PCA of final LSTM hidden state h_T
  ablation_table.png        Metrics table
  summary.txt               Numeric results

Usage
-----
    python scripts/lstm_detector.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/

    # with CLIP task conditioning (requires: pip install transformers)
    python scripts/lstm_detector.py \\
        --data_path ... --use_clip
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from failure_detector import (
    load_rollouts, FailureDetector, _compute_loss,
    _pad_collate, RolloutDataset, train_model, predict,
)
from combined_detector import (
    CombinedFailureDetector, _train_combined, _predict_combined,
    calibrate_threshold, _eval_at_thresh,
    compute_detection_curve, _compute_model_metrics, plot_ablation_table,
    _interp, _score_series, _draw_score_curves,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# optional CLIP import
try:
    from clip_task_detector import CLIPTaskEncoder, _train_clip, _predict_clip
    _CLIP_OK = True
except Exception:
    _CLIP_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  LSTM detector
# ══════════════════════════════════════════════════════════════════════════════

class LSTMDetector(nn.Module):
    """
    Single or multi-layer causal LSTM → linear score head → sigmoid.

    forward_raw(features)       → per-step raw scores (B, T)   [for BCE training]
    forward(features, masks)    → causal running-mean score (B, T) [for inference]

    Optional task conditioning: concatenate a task embedding to each step's
    input before the LSTM.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_layers: int = 1, dropout: float = 0.1,
                 task_embed_dim: int = 0):
        super().__init__()
        self.task_embed_dim = task_embed_dim
        lstm_input = input_dim + task_embed_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        score_linear = nn.Linear(hidden_dim // 2, 1)
        # Bias the output toward 0 (not failure) at initialisation.
        # Without this the LSTM collapses to always predicting failure.
        with torch.no_grad():
            score_linear.bias.fill_(-2.0)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            score_linear,
            nn.Sigmoid(),
        )

    def _cat_task(self, features: torch.Tensor,
                  task_embeds: torch.Tensor | None) -> torch.Tensor:
        if task_embeds is None or self.task_embed_dim == 0:
            return features
        te = task_embeds.unsqueeze(1).expand(-1, features.shape[1], -1)
        return torch.cat([features, te], dim=-1)

    def forward_raw(self, features: torch.Tensor,
                    task_embeds: torch.Tensor | None = None) -> torch.Tensor:
        """Returns per-step sigmoid scores (B, T) for BCE training."""
        x   = self._cat_task(features, task_embeds)
        out, _ = self.lstm(x)                           # (B, T, H)
        return self.score_head(out).squeeze(-1)         # (B, T)

    def forward(self, features: torch.Tensor,
                valid_masks: torch.Tensor,
                task_embeds: torch.Tensor | None = None) -> torch.Tensor:
        """Returns causal running-mean of LSTM scores (B, T)."""
        raw  = self.forward_raw(features, task_embeds) * valid_masks
        cnt  = torch.cumsum(valid_masks, dim=-1) + 1e-8
        return torch.cumsum(raw, dim=-1) / cnt * valid_masks

    def get_final_hidden(self, features: torch.Tensor,
                         task_embeds: torch.Tensor | None = None) -> torch.Tensor:
        """Returns final LSTM hidden state h_T for each episode. (B, H)"""
        x   = self._cat_task(features, task_embeds)
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]   # last layer's final hidden state (B, H)


# ══════════════════════════════════════════════════════════════════════════════
#  Training / inference
# ══════════════════════════════════════════════════════════════════════════════

def train_lstm(model: LSTMDetector, rollouts, n_epochs=300, lr=1e-3,
               lambda_reg=1e-2, batch_size=32, device="cpu",
               task_embeds=None):
    """Train the LSTMDetector with BCE loss (same recipe as FailureDetector)."""
    dataset = RolloutDataset(rollouts)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=_pad_collate)

    # If task embeds are used, we need a custom loader
    if task_embeds is not None:
        from combined_detector import CombinedDataset, _combined_collate
        dataset = CombinedDataset(rollouts, task_embeds)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=_combined_collate)

    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    model.train()
    losses = []
    pbar = tqdm(range(n_epochs), desc="Training LSTM", unit="ep")
    for _ in pbar:
        epoch_loss = 0.0
        for batch in loader:
            feat = batch["features"].to(device)
            mask = batch["valid_masks"].to(device)
            lbl  = batch["success_labels"].to(device)
            te   = batch.get("task_embeds", None)
            if te is not None: te = te.to(device)

            opt.zero_grad()
            raw  = model.forward_raw(feat, te)
            loss = _compute_loss(raw, mask, lbl, lambda_reg, model)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        avg = epoch_loss / len(loader)
        losses.append(avg)
        pbar.set_description(f"Loss {avg:.4f}")
        sched.step()
    return losses


@torch.no_grad()
def predict_lstm(model: LSTMDetector, rollouts, device="cpu",
                 task_embeds=None):
    """Returns list of score arrays (one per rollout)."""
    model.eval()
    scores = []
    for i, r in enumerate(rollouts):
        feat = r.hidden_states.unsqueeze(0).to(device)
        mask = torch.ones(1, feat.shape[1], device=device)
        te   = None
        if task_embeds is not None:
            te = torch.from_numpy(task_embeds[i:i+1]).to(device)
        s = model(feat, mask, te)
        scores.append(s.squeeze(0).cpu().numpy())
    return scores


@torch.no_grad()
def get_lstm_hidden(model: LSTMDetector, rollouts, device="cpu",
                    task_embeds=None):
    """Returns (N, H) array of final LSTM hidden states for PCA."""
    model.eval()
    hiddens = []
    for i, r in enumerate(rollouts):
        feat = r.hidden_states.unsqueeze(0).to(device)
        te   = None
        if task_embeds is not None:
            te = torch.from_numpy(task_embeds[i:i+1]).to(device)
        h = model.get_final_hidden(feat, te)
        hiddens.append(h.squeeze(0).cpu().numpy())
    return np.stack(hiddens)


# ══════════════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════════════

def plot_lstm_hidden_pca(all_r, hidden_states, out_dir, title="LSTM Final Hidden State PCA"):
    """2-D PCA of LSTM final h_T, coloured by outcome and task."""
    outcomes = np.array([r.episode_success for r in all_r])
    task_ids = np.array([r.task_id          for r in all_r])

    pca = PCA(n_components=2, random_state=0)
    X2  = pca.fit_transform(hidden_states)
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for outcome, label, color, marker in [
        (1, "Success", "seagreen", "o"),
        (0, "Failure", "crimson",  "x"),
    ]:
        mask = outcomes == outcome
        axes[0].scatter(X2[mask, 0], X2[mask, 1],
                        c=color, label=f"{label} (n={mask.sum()})",
                        s=16, alpha=0.55, marker=marker)
    axes[0].set_title(f"{title}\n— by Outcome")
    axes[0].set_xlabel(f"PC1 ({var[0]:.1%})"); axes[0].set_ylabel(f"PC2 ({var[1]:.1%})")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    unique_tasks = sorted(set(task_ids))
    cmap = plt.cm.get_cmap("tab10", len(unique_tasks))
    for i, tid in enumerate(unique_tasks):
        mask = task_ids == tid
        axes[1].scatter(X2[mask, 0], X2[mask, 1],
                        color=cmap(i), label=f"Task {tid}", s=16, alpha=0.55)
    axes[1].set_title(f"{title}\n— by Task")
    axes[1].set_xlabel(f"PC1 ({var[0]:.1%})"); axes[1].set_ylabel(f"PC2 ({var[1]:.1%})")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, "lstm_hidden_pca.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_score_curves_comparison(test_r, scores_dict, out_dir):
    """Side-by-side score curves for all models."""
    n = len(scores_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), squeeze=False)
    x = np.linspace(0, 1, 100)
    for ax, (name, sc) in zip(axes[0], scores_dict.items()):
        # sc is plain score array (MLP) or (score, weight) tuple (combined)
        ss, sf = _score_series(test_r, sc)
        _draw_score_curves(ax, ss, sf, f"{name}", x)
        ax.set_ylabel("Failure score")
    fig.suptitle("Score Curves by Model", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "score_curves.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_roc_prc(test_r, scores_dict, out_dir):
    y_true = np.array([1 - r.episode_success for r in test_r])
    if len(np.unique(y_true)) < 2:
        return

    colors = ["steelblue", "darkorange", "crimson", "mediumseagreen", "mediumpurple"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for (name, sc), col in zip(scores_dict.items(), colors):
        finals = np.array([
            float(s[0][-1]) if isinstance(s, tuple) else float(s[-1])
            for s in sc
        ])
        auc = roc_auc_score(y_true, finals)
        ap  = average_precision_score(y_true, finals)
        fpr, tpr, _ = roc_curve(y_true, finals)
        pre, rec, _ = precision_recall_curve(y_true, finals)
        axes[0].plot(fpr, tpr, lw=2, color=col, label=f"{name}  AUC={auc:.3f}")
        axes[1].plot(rec, pre, lw=2, color=col, label=f"{name}  AP={ap:.3f}")

    axes[0].plot([0,1],[0,1],"k--",lw=1)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve"); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

    axes[1].axhline(float(y_true.mean()), color="gray", ls="--", lw=1)
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("PRC"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0,1); axes[1].set_ylim(0,1.05)

    fig.suptitle("ROC & PRC Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "roc_prc.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LSTM failure detector — SAFE architecture comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",         required=True)
    parser.add_argument("--output_dir",        default="./lstm_results")
    parser.add_argument("--unseen_task_ratio", type=float, default=0.30)
    parser.add_argument("--target_recall",     type=float, default=0.90)
    parser.add_argument("--n_epochs",          type=int,   default=300)
    parser.add_argument("--lr",                type=float, default=5e-4,
                        help="LSTM trains best with a slightly lower LR than MLP")
    parser.add_argument("--lambda_reg",        type=float, default=1e-2)
    parser.add_argument("--hidden_dim",        type=int,   default=256)
    parser.add_argument("--n_lstm_layers",     type=int,   default=1)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--use_clip",          action="store_true",
                        help="Add a third LSTM+CLIP model (requires transformers)")
    parser.add_argument("--clip_proj_dim",     type=int,   default=64)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.use_clip and not _CLIP_OK:
        print("WARNING: --use_clip requested but clip_task_detector import failed. "
              "Install transformers or check clip_task_detector.py exists.")
        args.use_clip = False

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load & split ───────────────────────────────────────────────────────
    print("\n[1/5] Loading rollouts ...")
    all_r     = load_rollouts(args.data_path)
    input_dim = all_r[0].hidden_states.shape[-1]

    rng = np.random.RandomState(args.seed)
    all_task_ids = sorted(set(r.task_id for r in all_r))
    shuffled     = all_task_ids.copy(); rng.shuffle(shuffled)
    n_unseen     = max(1, round(args.unseen_task_ratio * len(all_task_ids)))
    unseen_ids   = set(shuffled[:n_unseen])
    seen_ids     = set(shuffled[n_unseen:])

    seen_idx   = [i for i, r in enumerate(all_r) if r.task_id in seen_ids]
    unseen_idx = [i for i, r in enumerate(all_r) if r.task_id in unseen_ids]
    rng.shuffle(seen_idx)

    seen_s = [i for i in seen_idx if     all_r[i].episode_success]
    seen_f = [i for i in seen_idx if not all_r[i].episode_success]
    n_tr_s = max(1, int(len(seen_s) * 0.75))
    n_tr_f = max(1, int(len(seen_f) * 0.75))
    tr_idx = seen_s[:n_tr_s] + seen_f[:n_tr_f]
    ca_idx = seen_s[n_tr_s:] + seen_f[n_tr_f:]
    te_idx = unseen_idx

    train_r = [all_r[i] for i in tr_idx]
    calib_r = [all_r[i] for i in ca_idx]
    test_r  = [all_r[i] for i in te_idx]

    def _cnt(r):
        ns = sum(x.episode_success for x in r)
        return f"{len(r)} ({ns}s/{len(r)-ns}f)"
    print(f"  Seen: {sorted(seen_ids)}   Unseen: {sorted(unseen_ids)}  ← test")
    print(f"  Train: {_cnt(train_r)}  Calib: {_cnt(calib_r)}  Test: {_cnt(test_r)}")

    alpha = 1.0 - args.target_recall

    # Optional: CLIP embeddings
    clip_embeds = {s: None for s in ("train", "calib", "test")}
    E_clip = 0
    if args.use_clip:
        print("\n  Loading CLIP encoder ...")
        clip_enc = CLIPTaskEncoder(device=args.device)
        all_descs    = [r.task_description for r in all_r]
        all_clip_emb = clip_enc.encode(all_descs)
        clip_embeds["train"] = all_clip_emb[tr_idx]
        clip_embeds["calib"] = all_clip_emb[ca_idx]
        clip_embeds["test"]  = all_clip_emb[te_idx]
        E_clip = all_clip_emb.shape[-1]

    # ── 2. Train base MLP ─────────────────────────────────────────────────────
    print("\n[2/5] Training base MLP ...")
    mlp = FailureDetector(input_dim, hidden_dim=args.hidden_dim,
                          n_layers=2).to(args.device)
    l_mlp = train_model(mlp, train_r, n_epochs=args.n_epochs,
                        lr=args.lr, lambda_reg=args.lambda_reg,
                        batch_size=args.batch_size, device=args.device)
    sc_mlp_ca = predict(mlp, calib_r, device=args.device)
    sc_mlp_te = predict(mlp, test_r,  device=args.device)
    tau_mlp   = calibrate_threshold(calib_r, sc_mlp_ca, alpha)

    # ── 3. Train LSTM ─────────────────────────────────────────────────────────
    print("\n[3/5] Training LSTM ...")
    lstm = LSTMDetector(input_dim, hidden_dim=args.hidden_dim,
                        n_layers=args.n_lstm_layers).to(args.device)
    l_lstm = train_lstm(lstm, train_r, n_epochs=args.n_epochs,
                        lr=args.lr, lambda_reg=args.lambda_reg,
                        batch_size=args.batch_size, device=args.device)
    sc_lstm_ca = predict_lstm(lstm, calib_r, device=args.device)
    sc_lstm_te = predict_lstm(lstm, test_r,  device=args.device)
    tau_lstm   = calibrate_threshold(calib_r, sc_lstm_ca, alpha)

    # ── 4. (Optional) LSTM + CLIP ─────────────────────────────────────────────
    sc_lc_ca = sc_lc_te = None; tau_lc = 0.5; l_lc = []
    if args.use_clip and E_clip > 0:
        print("\n[3b] Training LSTM + CLIP ...")
        lstm_clip = LSTMDetector(input_dim, hidden_dim=args.hidden_dim,
                                 n_layers=args.n_lstm_layers,
                                 task_embed_dim=args.clip_proj_dim).to(args.device)
        # Build projected CLIP embeds via a fixed linear projection
        proj = nn.Linear(E_clip, args.clip_proj_dim).to(args.device)
        with torch.no_grad():
            tr_proj = proj(torch.from_numpy(clip_embeds["train"]).to(args.device)).cpu().numpy()
            ca_proj = proj(torch.from_numpy(clip_embeds["calib"]).to(args.device)).cpu().numpy()
            te_proj = proj(torch.from_numpy(clip_embeds["test"] ).to(args.device)).cpu().numpy()

        l_lc     = train_lstm(lstm_clip, train_r, n_epochs=args.n_epochs,
                               lr=args.lr, lambda_reg=args.lambda_reg,
                               batch_size=args.batch_size, device=args.device,
                               task_embeds=tr_proj)
        sc_lc_ca = predict_lstm(lstm_clip, calib_r, device=args.device, task_embeds=ca_proj)
        sc_lc_te = predict_lstm(lstm_clip, test_r,  device=args.device, task_embeds=te_proj)
        tau_lc   = calibrate_threshold(calib_r, sc_lc_ca, alpha)

    # ── 5. Evaluate & plot ────────────────────────────────────────────────────
    print("\n[4/5] Evaluating ...")
    scores_dict = {"MLP (base)": sc_mlp_te, "LSTM": sc_lstm_te}
    taus_dict   = {"MLP (base)": tau_mlp,   "LSTM": tau_lstm}
    if sc_lc_te is not None:
        scores_dict["LSTM+CLIP"] = sc_lc_te
        taus_dict["LSTM+CLIP"]   = tau_lc

    lines = [f"LSTM DETECTOR — RESULTS (seed={args.seed})",
             f"Unseen tasks: {sorted(unseen_ids)}", ""]
    ablation = {}
    for name, sc in scores_dict.items():
        m = _compute_model_metrics(test_r, sc, taus_dict[name])
        ablation[name] = m
        lines += [
            f"─── {name} ───",
            f"  AUC: {m['auc']:.4f}  AP: {m['ap']:.4f}  Acc@0.5: {m['acc']:.4f}",
            f"  τ={taus_dict[name]:.4f}: recall={m['recall']:.4f}  "
            f"FAR={m['far']:.4f}  avg_det={m['avg_det']:.4f}", "",
        ]
    print("\n".join(lines))
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines))

    print("\n[5/5] Plotting ...")

    # Training loss
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(l_mlp,  lw=1.5, color="steelblue",  label="MLP (base)")
    ax.plot(l_lstm, lw=1.5, color="darkorange", label="LSTM")
    if l_lc: ax.plot(l_lc, lw=1.5, color="crimson", label="LSTM+CLIP")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args.output_dir, "training_loss.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")

    plot_roc_prc(test_r, scores_dict, args.output_dir)
    plot_score_curves_comparison(test_r, scores_dict, args.output_dir)

    # Detection time
    fig, ax = plt.subplots(figsize=(6, 5))
    colors = ["steelblue", "darkorange", "crimson"]
    for (name, sc), col in zip(scores_dict.items(), colors):
        rec, dts, _ = compute_detection_curve(test_r, sc, n_points=50)
        ax.plot(dts, rec, lw=2, color=col, label=name)
    ax.set_xlabel("Avg normalised detection time ↓")
    ax.set_ylabel("Recall")
    ax.set_title("Early Detection Tradeoff\n(upper-left = best)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
    fig.tight_layout()
    p = os.path.join(args.output_dir, "detection_time.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")

    # LSTM hidden state PCA (all rollouts for richer visualisation)
    print("  Computing LSTM hidden states for PCA ...")
    all_te = None
    if args.use_clip and E_clip > 0:
        with torch.no_grad():
            all_proj = proj(torch.from_numpy(
                clip_enc.encode([r.task_description for r in all_r])
            ).to(args.device)).cpu().numpy()
        all_te = all_proj
    all_hiddens = get_lstm_hidden(lstm, all_r, device=args.device, task_embeds=all_te)
    plot_lstm_hidden_pca(all_r, all_hiddens, args.output_dir)

    plot_ablation_table(ablation, args.output_dir)

    print(f"\n  All outputs saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
