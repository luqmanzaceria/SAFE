#!/usr/bin/env python3
"""
Attention Temporal Failure Detector
=====================================

The base detector uses a *hard* running mean to aggregate per-step scores
over time — every step contributes equally regardless of how informative it
is.  This script replaces that with a *learned* soft-attention mechanism:

    raw_t     = MLP(h_t)           ← same per-step sigmoid as base
    weight_t  = sigmoid(w(h_t))    ← learned scalar importance of step t
    score_t   = Σ_{i≤t} weight_i · raw_i
                ─────────────────────────    ← normalised running weighted mean
                Σ_{i≤t} weight_i

Because weight_t is learned, the model can:
  - Down-weight early steps (robot still approaching object → little signal)
  - Up-weight critical moments (grasp attempt, placement, collision)

At inference the attention weights are also visualised per episode, showing
exactly *which moments* the detector considered most informative — a form of
built-in explainability.

Comparison
----------
The script trains both the base detector and the attention detector under
identical conditions and reports AUC / accuracy for both, plus plots:

  attention_weights.png   - mean ± std attention weight over normalised time,
                            split by success/failure outcome
  attention_heatmap.png   - per-episode heatmap (test set, sorted by outcome)
  comparison_roc.png      - ROC curves: base vs attention
  score_curves_attn.png   - score curve comparison

Usage
-----
    python scripts/attention_detector.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
        --output_dir ./attention_results
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, auc as sk_auc

sys.path.insert(0, os.path.dirname(__file__))
from failure_detector import (
    load_rollouts,
    _pad_collate,
    RolloutDataset,
    _compute_loss,
    train_model,
    predict,
    FailureDetector,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


# ──────────────────────────── Model ────────────────────────────────────────

class AttentionFailureDetector(nn.Module):
    """
    Per-timestep MLP with a causal learned-attention aggregation.

    Two separate small MLPs share the same hidden features:
      - `score_head`   → raw failure probability in [0,1] at each step
      - `weight_head`  → importance weight in [0,1] for that step

    Aggregation (causal, usable online):
        score_t = Σ_{i≤t} w_i · p_i  /  Σ_{i≤t} w_i

    Training loss is BCE on the raw per-step scores (same as base detector)
    so the score head learns to discriminate successes from failures.
    The weight head is trained jointly via the gradient that flows from the
    aggregated score at the *end* of the episode (auxiliary loss term).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        # Shared encoder
        enc_layers, in_d = [], input_dim
        for _ in range(n_layers - 1):
            enc_layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(),
                           nn.Dropout(dropout)]
            in_d = hidden_dim
        self.encoder = nn.Sequential(*enc_layers)

        # Score head: produces per-step failure probability
        self.score_head  = nn.Sequential(nn.Linear(in_d, 1), nn.Sigmoid())

        # Attention head: produces per-step importance weight
        # Separate small network so it can learn independent importance signal
        self.weight_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _encode(self, features: torch.Tensor):
        """features: (B, T, D) → enc: (B, T, H), raw_scores, weights"""
        enc     = self.encoder(features)                 # (B, T, H)
        raw     = self.score_head(enc).squeeze(-1)       # (B, T)
        weights = self.weight_head(features).squeeze(-1) # (B, T)
        return raw, weights

    def forward_raw(self, features: torch.Tensor) -> torch.Tensor:
        """Per-step sigmoid score without attention — used for BCE loss."""
        enc = self.encoder(features)
        return self.score_head(enc).squeeze(-1)          # (B, T)

    def forward(self, features: torch.Tensor,
                valid_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          scores  (B, T) — causal attention-weighted running mean
          weights (B, T) — per-step attention weights (for visualisation)
        """
        raw, w = self._encode(features)                  # (B, T) each

        # Zero out padded positions
        raw = raw * valid_masks
        w   = w   * valid_masks

        # Cumulative weighted sum & normalise
        cum_w    = torch.cumsum(w,       dim=-1) + 1e-8  # (B, T)
        cum_raw  = torch.cumsum(w * raw, dim=-1)         # (B, T)
        scores   = cum_raw / cum_w * valid_masks         # (B, T)

        return scores, w


def _train_attention(model, rollouts, n_epochs=300, lr=1e-3,
                     lambda_reg=1e-2, lambda_attn=0.1,
                     batch_size=32, device="cpu"):
    """
    Two-term loss:
      L = BCE(raw_per_step, targets)           ← discriminative per-step loss
        + λ_attn * BCE(final_attn_score, targets)  ← end-of-episode loss
                                                     trains the weight head

    The second term trains the weight head to produce high scores at the end
    of failure episodes (and low at success) — i.e. it must correctly
    *weight* the informative steps so the final aggregated score is correct.
    """
    dataset = RolloutDataset(rollouts)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=_pad_collate)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    model.train()
    pbar  = tqdm(range(n_epochs), desc="Training (attention)", unit="ep")
    for _ in pbar:
        epoch_loss = 0.0
        for batch in loader:
            feat = batch["features"].to(device)
            mask = batch["valid_masks"].to(device)
            lbl  = batch["success_labels"].to(device)

            opt.zero_grad()

            # Per-step BCE loss (trains score head)
            raw_scores = model.forward_raw(feat)
            loss_raw   = _compute_loss(raw_scores, mask, lbl, lambda_reg, model)

            # End-of-episode attention-aggregated loss (trains weight head)
            attn_scores, _ = model(feat, mask)
            # Final score per episode = score at last valid step
            lengths = mask.long().sum(dim=-1) - 1          # (B,)
            final   = attn_scores[torch.arange(len(lbl)), lengths]  # (B,)
            targets = (1 - lbl)                             # 1=failure
            n_s = lbl.sum().item() + 1e-6
            n_f = (1 - lbl).sum().item() + 1e-6
            pw  = torch.tensor(n_s / n_f, device=device)
            bce_f = nn.functional.binary_cross_entropy(final, targets,
                                                        reduction="none")
            bce_f = (bce_f * (targets * pw + (1 - targets))).mean()

            loss = loss_raw + lambda_attn * bce_f
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        pbar.set_description(f"Loss {epoch_loss / len(loader):.4f}")
        sched.step()


@torch.no_grad()
def _predict_attention(model, rollouts, device="cpu"):
    """Returns list of (score_curve, weight_curve) per rollout."""
    model.eval()
    out = []
    for r in rollouts:
        feat = r.hidden_states.unsqueeze(0).to(device)
        mask = torch.ones(1, feat.shape[1], device=device)
        s, w = model(feat, mask)
        out.append((s.squeeze(0).cpu().numpy(),
                    w.squeeze(0).cpu().numpy()))
    return out


# ──────────────────────────── Visualisations ───────────────────────────────

def _interp(arr, n=100):
    return np.interp(np.linspace(0, len(arr)-1, n),
                     np.arange(len(arr)), arr)


def plot_attention_weights(rollouts, attn_out, out_dir):
    """Mean ± std attention weight curve, split by outcome."""
    n = 100
    x = np.linspace(0, 1, n)
    succ_w, fail_w = [], []
    for r, (_, w) in zip(rollouts, attn_out):
        (succ_w if r.episode_success else fail_w).append(_interp(w, n))

    fig, ax = plt.subplots(figsize=(9, 4))
    for curves, label, color in [
        (succ_w, f"Success  (n={len(succ_w)})", "seagreen"),
        (fail_w, f"Failure  (n={len(fail_w)})", "crimson"),
    ]:
        if not curves:
            continue
        arr  = np.stack(curves)
        mean = arr.mean(0); std = arr.std(0)
        for trace in arr:
            ax.plot(x, trace, color=color, lw=0.3, alpha=0.1)
        ax.plot(x, mean, color=color, lw=2.5, label=label, zorder=3)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25)
    ax.set_xlabel("Normalised time"); ax.set_ylabel("Attention weight")
    ax.set_title("Learned Temporal Attention Weights by Outcome")
    ax.set_ylim(-0.05, 1.05); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "attention_weights.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")


def plot_attention_heatmap(rollouts, attn_out, out_dir, max_eps=60):
    """
    Each row = one episode, colour = attention weight at each timestep.
    Episodes are sorted: successes on top, failures below, with a divider.
    """
    n = 100
    succ_rows, fail_rows = [], []
    for r, (_, w) in zip(rollouts, attn_out):
        (succ_rows if r.episode_success else fail_rows).append(_interp(w, n))

    # Limit to max_eps for readability
    half = max_eps // 2
    rows = succ_rows[:half] + fail_rows[:half]
    if not rows:
        return

    mat = np.stack(rows)          # (N, 100)
    n_s = min(len(succ_rows), half)
    n_f = min(len(fail_rows), half)

    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.18)))
    im = ax.imshow(mat, aspect="auto", origin="upper", cmap="hot",
                   vmin=0, vmax=1, extent=[0, 1, len(rows), 0])
    ax.axhline(n_s, color="cyan", lw=1.5, linestyle="--",
               label=f"↑ Success ({n_s})  |  Failure ({n_f}) ↓")
    ax.set_xlabel("Normalised time"); ax.set_ylabel("Episode index")
    ax.set_title("Attention Weight Heatmap  (bright = high importance)")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "attention_heatmap.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")


def plot_roc_comparison(rollouts, base_scores, attn_out, out_dir):
    y_true    = np.array([1 - r.episode_success for r in rollouts])
    y_base    = np.array([s[-1]  for s in base_scores])
    y_attn    = np.array([sc[-1] for sc, _ in attn_out])

    if len(np.unique(y_true)) < 2:
        return

    fig, ax = plt.subplots(figsize=(5, 5))
    for y_score, label, color in [
        (y_base,  f"Base (AUC={roc_auc_score(y_true, y_base):.3f})",  "steelblue"),
        (y_attn,  f"Attention (AUC={roc_auc_score(y_true, y_attn):.3f})", "darkorange"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        ax.plot(fpr, tpr, lw=2, color=color, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC: Base vs Attention Detector")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_roc.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Attention temporal failure detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",    required=True)
    parser.add_argument("--output_dir",   default="./attention_results")
    parser.add_argument("--train_ratio",  type=float, default=0.7)
    parser.add_argument("--n_epochs",     type=int,   default=300)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--lambda_reg",   type=float, default=1e-2)
    parser.add_argument("--lambda_attn",  type=float, default=0.1,
                        help="Weight of end-of-episode attention loss term")
    parser.add_argument("--hidden_dim",   type=int,   default=256)
    parser.add_argument("--n_layers",     type=int,   default=2)
    parser.add_argument("--batch_size",   type=int,   default=32)
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load & split ─────────────────────────────────────────────────────
    print("[1/4] Loading rollouts ...")
    all_r = load_rollouts(args.data_path)
    rng   = np.random.RandomState(args.seed)
    s_idx = [i for i, r in enumerate(all_r) if     r.episode_success]
    f_idx = [i for i, r in enumerate(all_r) if not r.episode_success]
    rng.shuffle(s_idx); rng.shuffle(f_idx)
    n_str = max(1, int(len(s_idx) * args.train_ratio))
    n_ftr = max(1, int(len(f_idx) * args.train_ratio))
    train_r = [all_r[i] for i in s_idx[:n_str] + f_idx[:n_ftr]]
    test_r  = [all_r[i] for i in s_idx[n_str:] + f_idx[n_ftr:]]

    input_dim = all_r[0].hidden_states.shape[-1]

    # ── 2. Train base ────────────────────────────────────────────────────────
    print("\n[2/4] Training base detector ...")
    base = FailureDetector(input_dim, hidden_dim=args.hidden_dim,
                           n_layers=args.n_layers).to(args.device)
    train_model(base, train_r, n_epochs=args.n_epochs, lr=args.lr,
                lambda_reg=args.lambda_reg, batch_size=args.batch_size,
                device=args.device)

    # ── 3. Train attention ───────────────────────────────────────────────────
    print("\n[3/4] Training attention detector ...")
    attn = AttentionFailureDetector(input_dim, hidden_dim=args.hidden_dim,
                                    n_layers=args.n_layers).to(args.device)
    _train_attention(attn, train_r, n_epochs=args.n_epochs, lr=args.lr,
                     lambda_reg=args.lambda_reg, lambda_attn=args.lambda_attn,
                     batch_size=args.batch_size, device=args.device)

    # ── 4. Evaluate & visualise ──────────────────────────────────────────────
    print("\n[4/4] Evaluating & visualising ...")
    base_scores = predict(base, test_r, device=args.device)
    attn_out    = _predict_attention(attn, test_r, device=args.device)

    y_true = np.array([1 - r.episode_success for r in test_r])
    y_base = np.array([s[-1]  for s in base_scores])
    y_attn = np.array([s[-1]  for s, _ in attn_out])

    print(f"\n  Base AUC:      {roc_auc_score(y_true, y_base):.4f}"
          if len(np.unique(y_true)) > 1 else "  (single class — no AUC)")
    if len(np.unique(y_true)) > 1:
        auc_b = roc_auc_score(y_true, y_base)
        auc_a = roc_auc_score(y_true, y_attn)
        print(f"  Base AUC:      {auc_b:.4f}")
        print(f"  Attention AUC: {auc_a:.4f}  "
              f"({'↑' if auc_a > auc_b else '↓'}{abs(auc_a-auc_b):.4f})")

    acc_base = np.mean((y_base > 0.5) == y_true)
    acc_attn = np.mean((y_attn > 0.5) == y_true)
    print(f"  Base Acc@0.5:  {acc_base:.4f}")
    print(f"  Attn Acc@0.5:  {acc_attn:.4f}")

    plot_attention_weights(test_r,  attn_out, args.output_dir)
    plot_attention_heatmap(test_r,  attn_out, args.output_dir)
    plot_roc_comparison(test_r, base_scores, attn_out, args.output_dir)
    print(f"\n  Outputs saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
