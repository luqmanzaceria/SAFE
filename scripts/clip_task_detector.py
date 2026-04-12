#!/usr/bin/env python3
"""
CLIP Task-Conditioned Failure Detector
=======================================

Replaces the TF-IDF/LSA task encoder with a frozen CLIP text encoder (512-d).

Key hypothesis
--------------
LSA embeddings are built from co-occurrence statistics of your 10 task
descriptions only.  An unseen task description falls nearly out-of-vocabulary.
CLIP was trained on 400M image-text pairs and encodes semantic similarity in a
rich, continuous space — "place the green cube" is close to "put the block"
even if neither appears in your training set.

This should restore task-conditioning gains that were lost with LSA:
    LSA result:  +task ≈ base  (no improvement)
    Expected:    CLIP-task > base  (task signal helps unseen tasks)

Three models trained for direct comparison
------------------------------------------
  1. No-task + attention   (reproduced from combined_detector)
  2. LSA-task + attention  (reproduced from combined_detector)
  3. CLIP-task + attention (NEW)

The CLIP encoder is frozen.  A lightweight learnable projection (512→proj_dim)
is trained jointly with the detector to keep the task signal dimensionally
balanced against the 4096-d hidden state.

Outputs (output_dir/)
---------------------
  training_loss.png         3 training curves
  roc_comparison.png        ROC: all 3 models
  prc_comparison.png        PRC: all 3 models
  detection_time.png        Avg det-time vs recall: all 3 models
  clip_embed_pca.png        PCA of CLIP task embeddings (seen/unseen marked)
  score_distribution.png    Final-score histograms
  results_table.png         Side-by-side metrics table
  summary.txt               Numeric results

Setup
-----
  pip install transformers          # for CLIP text encoder
  # first run downloads ~600 MB (ViT-B/32 weights, cached afterwards)

Usage
-----
    python scripts/clip_task_detector.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/

    # faster run (fewer epochs, good for checking):
    python scripts/clip_task_detector.py \\
        --data_path ... --n_epochs 100 --no_lsa
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    average_precision_score, precision_recall_curve,
)

sys.path.insert(0, os.path.dirname(__file__))
from failure_detector import (
    load_rollouts, FailureDetector, _compute_loss, train_model, predict,
)
from combined_detector import (
    CombinedFailureDetector, CombinedDataset, _combined_collate,
    _train_combined, _predict_combined,
    calibrate_threshold, _eval_at_thresh,
    compute_detection_curve, _compute_model_metrics, plot_ablation_table,
    TaskEncoder,
    _interp, _score_series, _draw_score_curves,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── CLIP availability ────────────────────────────────────────────────────────
try:
    from transformers import CLIPTokenizer, CLIPTextModel
    _CLIP_OK = True
except ImportError:
    _CLIP_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  CLIP task encoder
# ══════════════════════════════════════════════════════════════════════════════

class CLIPTaskEncoder:
    """
    Frozen CLIP ViT-B/32 text encoder → 512-d embeddings.

    Embeddings are computed once and cached.  A separate learnable projection
    (512 → proj_dim) is trained alongside the detector to balance dimensionality.
    """
    EMBED_DIM = 512   # ViT-B/32 text embedding size

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32",
                 device: str = "cpu"):
        if not _CLIP_OK:
            raise ImportError(
                "transformers not found.  "
                "Run: pip install transformers  then retry."
            )
        print(f"  [CLIP] Loading {model_name} (first run downloads ~600 MB) ...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model     = CLIPTextModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device
        self._cache: dict[str, np.ndarray] = {}

    @torch.no_grad()
    def encode(self, descriptions: list[str]) -> np.ndarray:
        """Return (N, 512) float32 array.  Results are cached by description."""
        results = []
        for desc in descriptions:
            if desc not in self._cache:
                toks = self.tokenizer(
                    [desc], padding=True, truncation=True,
                    max_length=77, return_tensors="pt",
                ).to(self.device)
                out = self.model(**toks)
                # pooler_output = projected [EOS] token (the standard CLIP text rep)
                emb = out.pooler_output[0].cpu().float().numpy()
                self._cache[desc] = emb
            results.append(self._cache[desc])
        return np.stack(results).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  CLIP-conditioned combined detector
# ══════════════════════════════════════════════════════════════════════════════

class CLIPCondFailureDetector(nn.Module):
    """
    CombinedFailureDetector with a learnable 512→proj_dim projection applied to
    the CLIP embedding before concatenation.  The CLIP encoder itself is frozen.

    Architecture per timestep:
        [ hidden_state (D) || proj(clip_embed) (P) ]
              ↓  shared MLP encoder (n_layers-1 hidden layers)
           latent_t
          ↙        ↘
    score_head    weight_head
    p_t=σ(...)    w_t=σ(...)
         ↘        ↙
      causal weighted mean → score_t
    """
    def __init__(self, hidden_state_dim: int, clip_embed_dim: int = 512,
                 proj_dim: int = 64, hidden_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.clip_proj   = nn.Linear(clip_embed_dim, proj_dim)
        self.input_dim   = hidden_state_dim + proj_dim

        enc, in_d = [], self.input_dim
        for _ in range(n_layers - 1):
            enc += [nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_d = hidden_dim
        self.encoder     = nn.Sequential(*enc)
        self.score_head  = nn.Sequential(nn.Linear(in_d, 1), nn.Sigmoid())
        self.weight_head = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid(),
        )

    def _cat(self, features: torch.Tensor,
             clip_embeds: torch.Tensor) -> torch.Tensor:
        """(B,T,D) + (B,512) → (B,T,D+P) after projection."""
        proj = self.clip_proj(clip_embeds)              # (B, P)
        te   = proj.unsqueeze(1).expand(-1, features.shape[1], -1)
        return torch.cat([features, te], dim=-1)        # (B, T, D+P)

    def forward_raw(self, features: torch.Tensor,
                    clip_embeds: torch.Tensor) -> torch.Tensor:
        x = self._cat(features, clip_embeds)
        return self.score_head(self.encoder(x)).squeeze(-1)

    def forward(self, features: torch.Tensor,
                clip_embeds: torch.Tensor,
                valid_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x   = self._cat(features, clip_embeds)
        raw = self.score_head(self.encoder(x)).squeeze(-1) * valid_masks
        w   = self.weight_head(x).squeeze(-1)              * valid_masks
        cum_w   = torch.cumsum(w,       dim=-1) + 1e-8
        cum_raw = torch.cumsum(w * raw, dim=-1)
        return cum_raw / cum_w * valid_masks, w


# ══════════════════════════════════════════════════════════════════════════════
#  Training / inference wrappers for the CLIP model
# ══════════════════════════════════════════════════════════════════════════════

def _train_clip(model, rollouts, clip_embeds, n_epochs=300, lr=1e-3,
                lambda_reg=1e-2, lambda_attn=0.1,
                batch_size=32, device="cpu"):
    """Same training loop as _train_combined but uses CLIPCondFailureDetector."""
    from combined_detector import CombinedDataset, _combined_collate
    dataset = CombinedDataset(rollouts, clip_embeds)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=_combined_collate)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    model.train()
    losses = []
    pbar = tqdm(range(n_epochs), desc="Training (CLIP-cond)", unit="ep")
    for _ in pbar:
        epoch_loss = 0.0
        for batch in loader:
            feat = batch["features"].to(device)
            te   = batch["task_embeds"].to(device)
            mask = batch["valid_masks"].to(device)
            lbl  = batch["success_labels"].to(device)
            opt.zero_grad()
            raw  = model.forward_raw(feat, te)
            loss = _compute_loss(raw, mask, lbl, lambda_reg, model)
            if lambda_attn > 0:
                scores, _ = model(feat, te, mask)
                lengths   = mask.long().sum(dim=-1) - 1
                final     = scores[torch.arange(len(lbl)), lengths]
                targets   = 1.0 - lbl
                n_s = lbl.sum().item() + 1e-6
                n_f = (1 - lbl).sum().item() + 1e-6
                pw  = torch.tensor(n_s / n_f, device=device)
                bce = nn.functional.binary_cross_entropy(
                    final, targets, reduction="none")
                loss = loss + lambda_attn * (bce * (targets*pw + (1-targets))).mean()
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
def _predict_clip(model, rollouts, clip_embeds, device="cpu"):
    model.eval()
    out = []
    for i, r in enumerate(rollouts):
        feat = r.hidden_states.unsqueeze(0).to(device)
        te   = torch.from_numpy(clip_embeds[i:i+1]).to(device)
        mask = torch.ones(1, feat.shape[1], device=device)
        s, w = model(feat, te, mask)
        out.append((s.squeeze(0).cpu().numpy(),
                    w.squeeze(0).cpu().numpy()))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════════════

def _finals(scores_or_pairs):
    return np.array([
        float(s[0][-1]) if isinstance(s, tuple) else float(s[-1])
        for s in scores_or_pairs
    ])


def plot_clip_embed_pca(encoder_clip, encoder_lsa, all_r,
                        unseen_ids, out_dir):
    """PCA of CLIP embeddings coloured by seen/unseen task."""
    unique_descs = list(dict.fromkeys(r.task_description for r in all_r))
    unique_ids   = [next(r.task_id for r in all_r if r.task_description == d)
                    for d in unique_descs]

    clip_embs = encoder_clip.encode(unique_descs)
    pca = PCA(n_components=2, random_state=0)
    proj = pca.fit_transform(clip_embs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: seen vs unseen
    ax = axes[0]
    for is_unseen, label, color, marker in [
        (False, "Seen (train/calib)", "steelblue",  "o"),
        (True,  "Unseen (test)",      "crimson",    "^"),
    ]:
        idxs = [i for i, tid in enumerate(unique_ids) if (tid in unseen_ids) == is_unseen]
        if idxs:
            ax.scatter(proj[idxs, 0], proj[idxs, 1],
                       c=color, label=label, s=120, marker=marker, zorder=3)
            for i in idxs:
                short = unique_descs[i][:40] + "…" if len(unique_descs[i]) > 40 else unique_descs[i]
                ax.annotate(short, proj[i], fontsize=6, alpha=0.85)
    ax.set_title("CLIP Task Embeddings (PCA)\nSeen vs Unseen Tasks")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Right: cosine similarity matrix
    ax = axes[1]
    norms = clip_embs / (np.linalg.norm(clip_embs, axis=1, keepdims=True) + 1e-8)
    sim   = norms @ norms.T
    im    = ax.imshow(sim, vmin=0, vmax=1, cmap="Blues", aspect="auto")
    short_descs = [d[:25] + "…" if len(d) > 25 else d for d in unique_descs]
    ax.set_xticks(range(len(unique_descs))); ax.set_xticklabels(short_descs, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(unique_descs))); ax.set_yticklabels(short_descs, fontsize=7)
    ax.set_title("CLIP Cosine Similarity Between Tasks")
    plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)

    fig.suptitle("CLIP Text Embedding Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "clip_embed_pca.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_three_way_comparison(test_r, scores_dict, taus_dict, out_dir):
    """ROC, PRC, detection-time, and score distribution for all 3 models."""
    y_true = np.array([1 - r.episode_success for r in test_r])
    if len(np.unique(y_true)) < 2:
        print("  (skipping comparison plots — test set has only one class)")
        return

    colors  = {"No-task+Attn": "steelblue", "LSA+Attn": "goldenrod", "CLIP+Attn": "crimson"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # ── ROC ──────────────────────────────────────────────────────────────────
    ax = axes[0, 0]
    for name, finals in [(k, _finals(v)) for k, v in scores_dict.items()]:
        auc = roc_auc_score(y_true, finals)
        fpr, tpr, _ = roc_curve(y_true, finals)
        ax.plot(fpr, tpr, lw=2, color=colors[name], label=f"{name}  AUC={auc:.3f}")
    ax.plot([0,1],[0,1],"k--",lw=1); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── PRC ──────────────────────────────────────────────────────────────────
    ax = axes[0, 1]
    for name, finals in [(k, _finals(v)) for k, v in scores_dict.items()]:
        ap = average_precision_score(y_true, finals)
        pre, rec, _ = precision_recall_curve(y_true, finals)
        ax.plot(rec, pre, lw=2, color=colors[name], label=f"{name}  AP={ap:.3f}")
    ax.axhline(float(y_true.mean()), color="gray", ls="--", lw=1)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # ── Detection time tradeoff ───────────────────────────────────────────────
    ax = axes[1, 0]
    for name, sc in scores_dict.items():
        rec, dts, _ = compute_detection_curve(test_r, sc, n_points=50)
        ax.plot(dts, rec, lw=2, color=colors[name], label=name)
    ax.set_xlabel("Avg normalised detection time ↓")
    ax.set_ylabel("Recall"); ax.set_title("Early Detection Tradeoff\n(upper-left = best)")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    # ── Score distributions ───────────────────────────────────────────────────
    ax = axes[1, 1]
    bins = np.linspace(0, 1, 30)
    offset = 0
    for name, sc in scores_dict.items():
        finals = _finals(sc)
        succ = finals[y_true == 0]; fail = finals[y_true == 1]
        tau  = taus_dict[name]
        ax.hist(succ, bins=bins+offset*0.003, alpha=0.4, color=colors[name],
                label=f"{name} succ", density=True, histtype="step", lw=2)
        ax.hist(fail, bins=bins+offset*0.003, alpha=0.4, color=colors[name],
                label=f"{name} fail", density=True, linestyle="--", histtype="step", lw=2)
        ax.axvline(tau, color=colors[name], lw=1, linestyle=":")
        offset += 1
    ax.set_xlabel("Final failure score"); ax.set_ylabel("Density")
    ax.set_title("Score Distributions (solid=succ, dashed=fail, dot=τ)")
    ax.legend(fontsize=6); ax.grid(True, alpha=0.3)

    fig.suptitle("Three-Way Model Comparison  (test on unseen tasks)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "comparison_all.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="CLIP task-conditioned failure detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",         required=True)
    parser.add_argument("--output_dir",        default="./clip_results")
    parser.add_argument("--unseen_task_ratio", type=float, default=0.30)
    parser.add_argument("--target_recall",     type=float, default=0.90)
    parser.add_argument("--clip_model",        default="openai/clip-vit-base-patch32")
    parser.add_argument("--clip_proj_dim",     type=int,   default=64,
                        help="Dimension to project 512-d CLIP embedding to. "
                             "Set to 512 to use full embedding with no projection.")
    parser.add_argument("--lsa_dim",           type=int,   default=32)
    parser.add_argument("--n_epochs",          type=int,   default=300)
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--lambda_reg",        type=float, default=1e-2)
    parser.add_argument("--lambda_attn",       type=float, default=0.1)
    parser.add_argument("--hidden_dim",        type=int,   default=256)
    parser.add_argument("--n_layers",          type=int,   default=2)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--seed",              type=int,   default=42)
    parser.add_argument("--no_lsa",            action="store_true",
                        help="Skip LSA baseline (faster, only compares no-task vs CLIP)")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if not _CLIP_OK:
        print("ERROR: transformers not installed.  Run: pip install transformers")
        sys.exit(1)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("\n[1/6] Loading rollouts ...")
    all_r     = load_rollouts(args.data_path)
    input_dim = all_r[0].hidden_states.shape[-1]

    # ── 2. Task encoders ─────────────────────────────────────────────────────
    print("\n[2/6] Building task embeddings ...")
    clip_enc = CLIPTaskEncoder(args.clip_model, device=args.device)
    all_descs  = [r.task_description for r in all_r]
    all_clip_emb = clip_enc.encode(all_descs)   # (N, 512)
    print(f"  CLIP: {all_clip_emb.shape[1]}-d embeddings for {len(all_r)} rollouts")

    lsa_enc = None; all_lsa_emb = None; E_lsa = 0
    if not args.no_lsa:
        lsa_enc = TaskEncoder(n_components=args.lsa_dim)
        lsa_enc.fit(list(dict.fromkeys(all_descs)))
        all_lsa_emb = lsa_enc.transform(all_descs)
        E_lsa = all_lsa_emb.shape[-1]
        print(f"  LSA: {E_lsa}-d embeddings")

    # ── 3. Seen / unseen task split ───────────────────────────────────────────
    print("\n[3/6] Splitting ...")
    rng = np.random.RandomState(args.seed)
    all_task_ids = sorted(set(r.task_id for r in all_r))
    shuffled     = all_task_ids.copy(); rng.shuffle(shuffled)
    n_unseen     = max(1, round(args.unseen_task_ratio * len(all_task_ids)))
    unseen_ids   = set(shuffled[:n_unseen])
    seen_ids     = set(shuffled[n_unseen:])
    print(f"  Seen   tasks: {sorted(seen_ids)}")
    print(f"  Unseen tasks: {sorted(unseen_ids)}  ← test set")

    seen_idx   = [i for i, r in enumerate(all_r) if r.task_id in seen_ids]
    unseen_idx = [i for i, r in enumerate(all_r) if r.task_id in unseen_ids]
    rng.shuffle(seen_idx)

    split_frac = 0.75   # 75% of seen → train, 25% → calib
    seen_s = [i for i in seen_idx if     all_r[i].episode_success]
    seen_f = [i for i in seen_idx if not all_r[i].episode_success]
    n_tr_s = max(1, int(len(seen_s) * split_frac))
    n_tr_f = max(1, int(len(seen_f) * split_frac))

    tr_idx = seen_s[:n_tr_s] + seen_f[:n_tr_f]
    ca_idx = seen_s[n_tr_s:] + seen_f[n_tr_f:]
    te_idx = unseen_idx

    def _g(idx):
        idx = list(idx)
        return [all_r[i] for i in idx], idx

    train_r, tr_i = _g(tr_idx)
    calib_r, ca_i = _g(ca_idx)
    test_r,  te_i = _g(te_idx)

    def _emb(arr, idx): return arr[idx] if arr is not None else None

    train_clip = all_clip_emb[tr_i]; calib_clip = all_clip_emb[ca_i]; test_clip = all_clip_emb[te_i]
    train_lsa  = _emb(all_lsa_emb, tr_i)
    calib_lsa  = _emb(all_lsa_emb, ca_i)
    test_lsa   = _emb(all_lsa_emb, te_i)

    def _cnt(r):
        ns = sum(x.episode_success for x in r)
        return f"{len(r)} ({ns}s/{len(r)-ns}f)"
    print(f"  Train: {_cnt(train_r)}  Calib: {_cnt(calib_r)}  Test: {_cnt(test_r)} (unseen)")

    alpha = 1.0 - args.target_recall

    # ── 4. Train models ───────────────────────────────────────────────────────
    print("\n[4/6] Training models ...")

    # Model 1: no-task + attention
    print("\n  [1/3] No-task + attention ...")
    zero_tr = np.zeros((len(train_r), 0), np.float32)
    zero_ca = np.zeros((len(calib_r), 0), np.float32)
    zero_te = np.zeros((len(test_r),  0), np.float32)
    m_notask = CombinedFailureDetector(
        hidden_state_dim=input_dim, task_embed_dim=0,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
    ).to(args.device)
    l_notask = _train_combined(
        m_notask, train_r, zero_tr,
        n_epochs=args.n_epochs, lr=args.lr, lambda_reg=args.lambda_reg,
        lambda_attn=args.lambda_attn, batch_size=args.batch_size,
        device=args.device,
    )
    sc_notask_ca = _predict_combined(m_notask, calib_r, zero_ca, device=args.device)
    sc_notask_te = _predict_combined(m_notask, test_r,  zero_te, device=args.device)
    tau_notask   = calibrate_threshold(calib_r, sc_notask_ca, alpha)

    # Model 2: LSA + attention
    sc_lsa_ca = sc_lsa_te = None
    tau_lsa = 0.5; l_lsa = []
    if not args.no_lsa:
        print("\n  [2/3] LSA + attention ...")
        m_lsa = CombinedFailureDetector(
            hidden_state_dim=input_dim, task_embed_dim=E_lsa,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        ).to(args.device)
        l_lsa = _train_combined(
            m_lsa, train_r, train_lsa,
            n_epochs=args.n_epochs, lr=args.lr, lambda_reg=args.lambda_reg,
            lambda_attn=args.lambda_attn, batch_size=args.batch_size,
            device=args.device,
        )
        sc_lsa_ca = _predict_combined(m_lsa, calib_r, calib_lsa, device=args.device)
        sc_lsa_te = _predict_combined(m_lsa, test_r,  test_lsa,  device=args.device)
        tau_lsa   = calibrate_threshold(calib_r, sc_lsa_ca, alpha)

    # Model 3: CLIP + attention
    print("\n  [3/3] CLIP + attention ...")
    proj_dim = args.clip_proj_dim if args.clip_proj_dim < 512 else 512
    m_clip = CLIPCondFailureDetector(
        hidden_state_dim=input_dim,
        clip_embed_dim=512,
        proj_dim=proj_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(args.device)
    l_clip = _train_clip(
        m_clip, train_r, train_clip,
        n_epochs=args.n_epochs, lr=args.lr, lambda_reg=args.lambda_reg,
        lambda_attn=args.lambda_attn, batch_size=args.batch_size,
        device=args.device,
    )
    sc_clip_ca = _predict_clip(m_clip, calib_r, calib_clip, device=args.device)
    sc_clip_te = _predict_clip(m_clip, test_r,  test_clip,  device=args.device)
    tau_clip   = calibrate_threshold(calib_r, sc_clip_ca, alpha)

    # ── 5. Results ────────────────────────────────────────────────────────────
    print("\n[5/6] Evaluating ...")
    y_true = np.array([1 - r.episode_success for r in test_r])

    scores_dict = {"No-task+Attn": sc_notask_te, "CLIP+Attn": sc_clip_te}
    taus_dict   = {"No-task+Attn": tau_notask,   "CLIP+Attn": tau_clip}
    if sc_lsa_te is not None:
        scores_dict["LSA+Attn"] = sc_lsa_te
        taus_dict["LSA+Attn"]   = tau_lsa
    # Sort for display
    order = ["No-task+Attn", "LSA+Attn", "CLIP+Attn"]
    scores_dict = {k: scores_dict[k] for k in order if k in scores_dict}
    taus_dict   = {k: taus_dict[k]   for k in order if k in taus_dict}

    lines = [f"CLIP TASK DETECTOR — RESULTS (seed={args.seed})",
             f"Unseen tasks: {sorted(unseen_ids)}", ""]
    ablation = {}
    for name, sc in scores_dict.items():
        m = _compute_model_metrics(test_r, sc, taus_dict[name])
        ablation[name] = m
        lines.append(f"─── {name} ───")
        lines.append(f"  AUC: {m['auc']:.4f}  AP: {m['ap']:.4f}  "
                     f"Acc@0.5: {m['acc']:.4f}")
        lines.append(f"  Conformal τ={taus_dict[name]:.4f}: "
                     f"recall={m['recall']:.4f}  FAR={m['far']:.4f}  "
                     f"avg_det={m['avg_det']:.4f}")
        lines.append("")

    print("\n" + "\n".join(lines))
    txt_p = os.path.join(args.output_dir, "summary.txt")
    with open(txt_p, "w") as f: f.write("\n".join(lines))
    print(f"  -> {txt_p}")

    # ── 6. Plots ──────────────────────────────────────────────────────────────
    print("\n[6/6] Plotting ...")

    # Training loss
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(l_notask, lw=1.5, color="steelblue", label="No-task+Attn")
    if l_lsa: ax.plot(l_lsa, lw=1.5, color="goldenrod", label="LSA+Attn")
    ax.plot(l_clip,   lw=1.5, color="crimson",   label="CLIP+Attn")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(args.output_dir, "training_loss.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")

    plot_three_way_comparison(test_r, scores_dict, taus_dict, args.output_dir)
    plot_clip_embed_pca(clip_enc, lsa_enc, all_r, unseen_ids, args.output_dir)
    plot_ablation_table(ablation, args.output_dir)

    print(f"\n  All outputs saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
