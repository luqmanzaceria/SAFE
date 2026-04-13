#!/usr/bin/env python3
"""
Combined Novelty Failure Detector
===================================

Unifies the three improvements that showed measurable gains into one model:

  ┌─────────────────────────────────────────────────────────────────────┐
  │  1. Task conditioning   — task description embedded via LSA and     │
  │                           concatenated with the hidden state so the │
  │                           model learns task-specific failure modes.  │
  │                                                                     │
  │  2. Temporal attention  — a learned importance weight per timestep  │
  │                           replaces the hard running mean so the     │
  │                           model can focus on critical moments        │
  │                           (e.g. grasp attempt, placement).          │
  │                                                                     │
  │  3. Conformal calibration — after training, a held-out calibration  │
  │                           set is used to derive a threshold τ_α     │
  │                           that formally guarantees                   │
  │                           P(detected | failure) ≥ 1−α              │
  └─────────────────────────────────────────────────────────────────────┘

Architecture
------------
  input_t  =  [ hidden_state_t (D=4096)  ||  task_embed (E) ]
                         ↓  shared encoder (MLP)
                    latent_t   (H-dim)
                  ↙               ↘
         score_head            weight_head
      p_t = σ(lin(latent_t))   w_t = σ(lin(input_t))
              ↘                ↙
    causal weighted mean (Σ_{i≤t} w_i·p_i / Σ_{i≤t} w_i) = score_t

Training loss (two terms)
  L = BCE(raw_step_scores, targets)          per-step discrimination
    + λ_attn · BCE(final_attn_score, target)  end-of-episode term that
                                               trains the weight head

Evaluation
----------
  Base detector   (no task cond, no attn, fixed threshold)
  Combined model  (task cond + attn, fixed threshold 0.5)
  Combined model  (task cond + attn, conformal threshold τ_α)

Outputs (output_dir/)
---------------------
  training_loss.png          base vs combined training curves
  roc_comparison.png         ROC: base vs combined
  prc_comparison.png         Precision-Recall: base vs combined         (NEW)
  score_curves.png           score trajectories (success / failure)
  attention_weights.png      mean attention weight over time by outcome
  attention_heatmap.png      per-episode heatmap (test set)
  coverage_curve.png         conformal recall vs false-alarm tradeoff
  conformal_histogram.png    score distributions with both thresholds
  detection_time_curve.png   avg detection time vs recall tradeoff      (NEW)
  hidden_state_pca.png       PCA of hidden states by outcome / task     (NEW)
  ablation_table.png         4-row ablation: base/+task/+attn/combined  (NEW)
  per_task_auc.png           per-task AUC: base vs combined
  task_embed_pca.png         2-D PCA of task embeddings
  summary.png                9-panel (3×3) dashboard                    (NEW)
  summary.txt                numeric results table

Usage
-----
    python scripts/combined_detector.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/

    # tune key hyperparameters
    python scripts/combined_detector.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
        --target_recall 0.90 \\
        --hidden_dim 256 \\
        --lambda_attn 0.1 \\
        --n_epochs 400 \\
        --output_dir ./combined_results
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc as sk_auc,
    precision_recall_curve, average_precision_score,
)

sys.path.insert(0, os.path.dirname(__file__))
from failure_detector import (
    load_rollouts,
    FailureDetector,
    _compute_loss,
    _pad_collate,
    RolloutDataset,
    train_model,
    predict,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
#  Task encoder
# ══════════════════════════════════════════════════════════════════════════════

class TaskEncoder:
    """TF-IDF + Truncated SVD (LSA) task description → fixed-length vector."""

    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(ngram_range=(1, 2), lowercase=True,
                                     strip_accents="unicode")
        self.svd   = TruncatedSVD(n_components=n_components, random_state=0)
        self.fitted     = False
        self.actual_dim = None

    def fit(self, descriptions: list):
        mat = self.tfidf.fit_transform(descriptions)
        # Cap n_components at (rank of matrix - 1) to avoid SVD errors
        max_k = min(mat.shape) - 1
        if self.svd.n_components > max_k:
            self.svd = TruncatedSVD(n_components=max_k, random_state=0)
        self.svd.fit(mat)
        self.actual_dim = self.svd.n_components
        var = self.svd.explained_variance_ratio_.sum()
        print(f"  [TaskEncoder] {self.actual_dim}-d LSA embedding "
              f"(requested {self.n_components}, capped at SVD rank {max_k+1}), "
              f"explains {var:.1%} of TF-IDF variance")
        self.fitted = True

    def transform(self, descriptions: list) -> np.ndarray:
        assert self.fitted
        mat = self.tfidf.transform(descriptions)
        return self.svd.transform(mat).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class CombinedDataset(Dataset):
    def __init__(self, rollouts: list, task_embeds: np.ndarray):
        self.rollouts    = rollouts
        self.task_embeds = torch.from_numpy(task_embeds)

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        r = self.rollouts[idx]
        return {
            "features":   r.hidden_states,
            "task_embed": self.task_embeds[idx],
            "label":      torch.tensor(float(r.episode_success)),
        }


def _combined_collate(batch):
    max_len = max(b["features"].shape[0] for b in batch)
    D = batch[0]["features"].shape[-1]
    E = batch[0]["task_embed"].shape[-1]
    B = len(batch)
    features   = torch.zeros(B, max_len, D)
    task_embs  = torch.zeros(B, E)
    masks      = torch.zeros(B, max_len)
    labels     = torch.zeros(B)
    for i, b in enumerate(batch):
        T = b["features"].shape[0]
        features[i, :T] = b["features"]
        masks[i, :T]    = 1.0
        task_embs[i]    = b["task_embed"]
        labels[i]       = b["label"]
    return {"features": features, "task_embeds": task_embs,
            "valid_masks": masks, "success_labels": labels}


# ══════════════════════════════════════════════════════════════════════════════
#  Combined model: task conditioning + temporal attention
# ══════════════════════════════════════════════════════════════════════════════

class CombinedFailureDetector(nn.Module):
    """
    Task-conditioned failure detector with learned temporal attention.

    Per-step input:  [ hidden_state (D) || task_embed (E) ]
    Shared encoder:  (D+E) → H-dim latent
    Score head:      H → sigmoid → raw failure probability p_t    (trains on BCE)
    Attention head:  (D+E) → sigmoid → step importance w_t  (trained via end-of-episode loss)

    Causal aggregation (usable online):
        score_t = Σ_{i≤t} w_i · p_i  /  Σ_{i≤t} w_i
    """
    def __init__(self, hidden_state_dim: int, task_embed_dim: int,
                 hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.task_embed_dim = task_embed_dim
        self.input_dim = hidden_state_dim + task_embed_dim

        # Shared encoder (n_layers-1 hidden layers)
        enc, in_d = [], self.input_dim
        for _ in range(n_layers - 1):
            enc += [nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_d = hidden_dim
        self.encoder    = nn.Sequential(*enc)
        self.score_head = nn.Sequential(nn.Linear(in_d, 1), nn.Sigmoid())

        # Separate attention head operates directly on concatenated input
        self.weight_head = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def _cat(self, features: torch.Tensor,
             task_embeds: torch.Tensor) -> torch.Tensor:
        """(B,T,D) + (B,E) → (B,T,D+E). Handles E=0 (attn-only ablation)."""
        if task_embeds is None or task_embeds.shape[-1] == 0:
            return features
        te = task_embeds.unsqueeze(1).expand(-1, features.shape[1], -1)
        return torch.cat([features, te], dim=-1)

    def forward_raw(self, features: torch.Tensor,
                    task_embeds: torch.Tensor) -> torch.Tensor:
        """Per-step sigmoid score (B,T) — used for BCE training loss."""
        x = self._cat(features, task_embeds)
        return self.score_head(self.encoder(x)).squeeze(-1)

    def forward(self, features: torch.Tensor,
                task_embeds: torch.Tensor,
                valid_masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          scores  (B,T) — causal attention-weighted running mean
          weights (B,T) — per-step attention weights (for visualisation)
        """
        x   = self._cat(features, task_embeds)
        raw = self.score_head(self.encoder(x)).squeeze(-1)   # (B,T)
        w   = self.weight_head(x).squeeze(-1)                # (B,T)

        raw = raw * valid_masks
        w   = w   * valid_masks

        cum_w   = torch.cumsum(w,       dim=-1) + 1e-8
        cum_raw = torch.cumsum(w * raw, dim=-1)
        scores  = cum_raw / cum_w * valid_masks
        return scores, w


# ══════════════════════════════════════════════════════════════════════════════
#  Training
# ══════════════════════════════════════════════════════════════════════════════

def _train_combined(model, rollouts, task_embeds, n_epochs=300, lr=1e-3,
                    lambda_reg=1e-2, lambda_attn=0.1, lambda_early=0.05,
                    batch_size=32, device="cpu"):
    dataset = CombinedDataset(rollouts, task_embeds)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=_combined_collate)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    model.train()
    losses = []
    pbar   = tqdm(range(n_epochs), desc="Training (combined)", unit="ep")
    for _ in pbar:
        epoch_loss = 0.0
        for batch in loader:
            feat = batch["features"].to(device)
            te   = batch["task_embeds"].to(device)
            mask = batch["valid_masks"].to(device)
            lbl  = batch["success_labels"].to(device)

            opt.zero_grad()

            # ── Term 1: per-step BCE (trains score head & encoder) ──────────
            raw  = model.forward_raw(feat, te)
            loss = _compute_loss(raw, mask, lbl, lambda_reg, model)

            # ── Term 2: end-of-episode attn loss (trains weight head) ────────
            if lambda_attn > 0 or lambda_early > 0:
                scores, weights = model(feat, te, mask)
                lengths = mask.long().sum(dim=-1) - 1

                if lambda_attn > 0:
                    final   = scores[torch.arange(len(lbl)), lengths]
                    targets = (1.0 - lbl)
                    n_s = lbl.sum().item() + 1e-6
                    n_f = (1 - lbl).sum().item() + 1e-6
                    pw  = torch.tensor(n_s / n_f, device=device)
                    bce = nn.functional.binary_cross_entropy(
                        final, targets, reduction="none"
                    )
                    loss = loss + lambda_attn * (bce * (targets * pw + (1 - targets))).mean()

                if lambda_early > 0:
                    # Earliness regularisation: for failure episodes, penalise
                    # high scores at late timesteps.
                    # L_early = mean over failure eps of Σ_t (t/T) * score_t
                    # This is differentiable and directly optimises avg_det.
                    T     = scores.shape[1]
                    pos   = torch.arange(T, device=device).float() / (T - 1 + 1e-8)
                    pos   = pos.unsqueeze(0)           # (1, T)
                    fail  = (1.0 - lbl).unsqueeze(1)   # (B, 1)  1 = failure episode
                    early_loss = (pos * scores * mask * fail).sum() / (fail.sum() + 1e-8)
                    loss  = loss + lambda_early * early_loss

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
def _predict_combined(model, rollouts, task_embeds, device="cpu"):
    """Returns list of (score_curve, weight_curve) per rollout."""
    model.eval()
    out = []
    for i, r in enumerate(rollouts):
        feat = r.hidden_states.unsqueeze(0).to(device)
        te   = torch.from_numpy(task_embeds[i:i+1]).to(device)
        mask = torch.ones(1, feat.shape[1], device=device)
        s, w = model(feat, te, mask)
        out.append((s.squeeze(0).cpu().numpy(),
                    w.squeeze(0).cpu().numpy()))
    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Conformal calibration
# ══════════════════════════════════════════════════════════════════════════════

def calibrate_threshold(calib_rollouts, calib_scores, alpha: float) -> float:
    """
    Returns the α-th quantile of calibration failure scores.
    Guarantees P(score ≥ τ | failure) ≥ 1−α on exchangeable test data.
    Accepts plain score arrays or (score, weight) tuples from the combined model.
    """
    fail_scores = []
    for r, s in zip(calib_rollouts, calib_scores):
        if not r.episode_success:
            final = s[0][-1] if isinstance(s, tuple) else s[-1]
            fail_scores.append(float(final))
    if not fail_scores:
        raise ValueError("No failures in calibration set.")
    n     = len(fail_scores)
    q_idx = int(np.ceil(alpha * (n + 1))) - 1
    q_idx = max(0, min(n - 1, q_idx))
    return float(np.sort(fail_scores)[q_idx])


def _eval_at_thresh(rollouts, scores, tau: float):
    y_true = np.array([1 - r.episode_success for r in rollouts])
    finals = np.array([
        float(s[0][-1]) if isinstance(s, tuple) else float(s[-1])
        for s in scores
    ])
    y_pred = (finals >= tau).astype(int)
    n_fail = y_true.sum(); n_succ = (1 - y_true).sum()
    recall = (y_pred[y_true == 1] == 1).sum() / max(n_fail, 1)
    far    = (y_pred[y_true == 0] == 1).sum() / max(n_succ, 1)
    acc    = (y_pred == y_true).mean()
    return float(recall), float(far), float(acc)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _interp(arr, n=100):
    return np.interp(np.linspace(0, len(arr) - 1, n),
                     np.arange(len(arr)), arr)


def _score_series(rollouts, scores_or_pairs, n=100):
    """Return (succ_curves, fail_curves) interpolated to n points."""
    succ, fail = [], []
    for r, item in zip(rollouts, scores_or_pairs):
        s = item[0] if isinstance(item, tuple) else item
        (succ if r.episode_success else fail).append(_interp(s, n))
    return succ, fail


# ══════════════════════════════════════════════════════════════════════════════
#  Individual plots
# ══════════════════════════════════════════════════════════════════════════════

def _draw_score_curves(ax, succ, fail, title, x=None):
    if x is None:
        x = np.linspace(0, 1, 100)
    for curves, label, color in [(succ, f"Success (n={len(succ)})", "seagreen"),
                                  (fail, f"Failure (n={len(fail)})", "crimson")]:
        if not curves:
            continue
        arr  = np.stack(curves)
        mean = arr.mean(0)
        for trace in arr:
            ax.plot(x, trace, color=color, lw=0.3, alpha=0.1)
        ax.plot(x, mean, color=color, lw=2.5, label=label, zorder=3)
        ax.fill_between(x, mean - arr.std(0), mean + arr.std(0),
                        color=color, alpha=0.25)
    ax.axhline(0.5, color="orange", linestyle="--", lw=1)
    ax.set_ylim(-0.05, 1.05); ax.set_title(title)
    ax.set_xlabel("Normalised time"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)


def plot_training_loss(base_losses, comb_losses, out_dir):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(base_losses, lw=1.5, color="steelblue",   label="Base")
    ax.plot(comb_losses, lw=1.5, color="darkorange", label="Combined")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "training_loss.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_roc_comparison(test_r, base_scores, comb_out, out_dir):
    y_true = np.array([1 - r.episode_success for r in test_r])
    if len(np.unique(y_true)) < 2:
        return
    y_base = np.array([s[-1]  for s in base_scores])
    y_comb = np.array([s[-1]  for s, _ in comb_out])
    fig, ax = plt.subplots(figsize=(5, 5))
    for ys, label, color in [
        (y_base, f"Base  (AUC={roc_auc_score(y_true, y_base):.3f})", "steelblue"),
        (y_comb, f"Combined (AUC={roc_auc_score(y_true, y_comb):.3f})", "darkorange"),
    ]:
        fpr, tpr, _ = roc_curve(y_true, ys)
        ax.plot(fpr, tpr, lw=2, color=color, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title("ROC Curve: Base vs Combined"); ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "roc_comparison.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_score_curves(test_r, base_scores, comb_out, out_dir):
    x    = np.linspace(0, 1, 100)
    bs, bf = _score_series(test_r, base_scores)
    cs, cf = _score_series(test_r, comb_out)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    _draw_score_curves(axes[0], bs, bf, "Score Curves — Base (test)", x)
    _draw_score_curves(axes[1], cs, cf, "Score Curves — Combined (test)", x)
    axes[0].set_ylabel("Failure score")
    fig.suptitle("Score Curves Comparison", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "score_curves.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_attention_weights(test_r, comb_out, out_dir):
    x = np.linspace(0, 1, 100)
    succ_w, fail_w = [], []
    for r, (_, w) in zip(test_r, comb_out):
        (succ_w if r.episode_success else fail_w).append(_interp(w, 100))
    fig, ax = plt.subplots(figsize=(9, 4))
    for curves, label, color in [
        (succ_w, f"Success (n={len(succ_w)})", "seagreen"),
        (fail_w, f"Failure (n={len(fail_w)})", "crimson"),
    ]:
        if not curves:
            continue
        arr  = np.stack(curves)
        mean = arr.mean(0)
        for trace in arr:
            ax.plot(x, trace, color=color, lw=0.3, alpha=0.1)
        ax.plot(x, mean, color=color, lw=2.5, label=label, zorder=3)
        ax.fill_between(x, mean - arr.std(0), mean + arr.std(0),
                        color=color, alpha=0.25)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Normalised time"); ax.set_ylabel("Attention weight")
    ax.set_title("Learned Temporal Attention Weights (Combined Model)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, "attention_weights.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_attention_heatmap(test_r, comb_out, out_dir, max_eps=60):
    n   = 100
    succ_rows, fail_rows = [], []
    for r, (_, w) in zip(test_r, comb_out):
        (succ_rows if r.episode_success else fail_rows).append(_interp(w, n))
    half  = max_eps // 2
    rows  = succ_rows[:half] + fail_rows[:half]
    if not rows:
        return
    mat   = np.stack(rows)
    n_s   = min(len(succ_rows), half)
    n_f   = min(len(fail_rows), half)
    fig, ax = plt.subplots(figsize=(10, max(3, len(rows) * 0.18)))
    im = ax.imshow(mat, aspect="auto", origin="upper", cmap="hot",
                   vmin=0, vmax=1, extent=[0, 1, len(rows), 0])
    ax.axhline(n_s, color="cyan", lw=1.5, linestyle="--",
               label=f"↑ Success ({n_s})  |  Failure ({n_f}) ↓")
    ax.set_xlabel("Normalised time"); ax.set_ylabel("Episode")
    ax.set_title("Attention Weight Heatmap  (bright = high importance)")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    p = os.path.join(out_dir, "attention_heatmap.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_coverage_curve(calib_r, calib_comb, calib_base,
                        test_r,  test_comb,  test_base, out_dir):
    alphas   = np.linspace(0.02, 0.40, 40)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax_idx, (calib_s, test_s, label, color) in enumerate([
        (calib_base, test_base, "Base",     "steelblue"),
        (calib_comb, test_comb, "Combined", "darkorange"),
    ]):
        recalls, fars, taus = [], [], []
        for a in alphas:
            try:
                tau = calibrate_threshold(calib_r, calib_s, a)
            except ValueError:
                recalls.append(np.nan); fars.append(np.nan); taus.append(np.nan)
                continue
            rec, far, _ = _eval_at_thresh(test_r, test_s, tau)
            recalls.append(rec); fars.append(far); taus.append(tau)

        x = 1 - alphas
        axes[0].plot(x, recalls, color=color, lw=2, label=label)
        axes[1].plot(x, fars,    color=color, lw=2, label=label)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    axes[0].set_xlabel("Target recall  (1−α)"); axes[0].set_ylabel("Empirical recall")
    axes[0].set_title("Coverage: Target vs Empirical")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Target recall  (1−α)"); axes[1].set_ylabel("False-alarm rate")
    axes[1].set_title("False-alarm Rate vs Target Recall")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, "coverage_curve.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_conformal_histogram(test_r, base_scores, comb_out,
                              tau_base, tau_comb, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    bins = np.linspace(0, 1, 30)

    for ax, scores_raw, tau, label in [
        (axes[0], [s[-1] for s in base_scores],     tau_base, "Base"),
        (axes[1], [s[-1] for s, _ in comb_out],     tau_comb, "Combined"),
    ]:
        succ = [s for r, s in zip(test_r, scores_raw) if     r.episode_success]
        fail = [s for r, s in zip(test_r, scores_raw) if not r.episode_success]
        ax_r = ax.twinx()
        if succ:
            ax.hist(succ,  bins=bins, color="seagreen", alpha=0.6,
                    label=f"Success (n={len(succ)})", density=True)
        if fail:
            ax_r.hist(fail, bins=bins, color="crimson", alpha=0.5,
                      label=f"Failure (n={len(fail)})", density=True)
        ax.axvline(0.5, color="orange",  linestyle="--", lw=2,
                   label="Fixed τ=0.50")
        ax.axvline(tau, color="purple",  linestyle="-.", lw=2,
                   label=f"Conformal τ={tau:.3f}")
        ax.set_xlabel("Final failure score")
        ax.set_ylabel("Density (success)", color="seagreen")
        ax_r.set_ylabel("Density (failure)", color="crimson")
        ax.tick_params(axis="y", labelcolor="seagreen")
        ax_r.tick_params(axis="y", labelcolor="crimson")
        ax.set_title(f"Score Distribution — {label}")
        lines1, lbl1 = ax.get_legend_handles_labels()
        lines2, lbl2 = ax_r.get_legend_handles_labels()
        ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    p = os.path.join(out_dir, "conformal_histogram.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_per_task_auc(test_r, base_scores, comb_out, out_dir):
    task_ids = sorted(set(r.task_id for r in test_r))
    base_aucs, comb_aucs, labels = [], [], []
    for tid in task_ids:
        idx = [i for i, r in enumerate(test_r) if r.task_id == tid]
        yt  = np.array([1 - test_r[i].episode_success for i in idx])
        if len(np.unique(yt)) < 2:
            continue
        yb = np.array([base_scores[i][-1] for i in idx])
        yc = np.array([comb_out[i][0][-1] for i in idx])
        base_aucs.append(roc_auc_score(yt, yb))
        comb_aucs.append(roc_auc_score(yt, yc))
        labels.append(f"T{tid}")
    if not labels:
        return
    x = np.arange(len(labels)); w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.9), 4))
    ax.bar(x - w/2, base_aucs, w, color="steelblue",   alpha=0.8, label="Base")
    ax.bar(x + w/2, comb_aucs, w, color="darkorange", alpha=0.8, label="Combined")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.1); ax.axhline(0.5, color="gray", lw=1, linestyle="--")
    ax.set_ylabel("ROC-AUC"); ax.set_title("Per-Task AUC: Base vs Combined")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    p = os.path.join(out_dir, "per_task_auc.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_task_embed_pca(all_r, encoder, out_dir):
    unique_d = list(dict.fromkeys(r.task_description for r in all_r))
    if len(unique_d) < 2:
        return
    embeds = encoder.transform(unique_d)
    n_comp = min(2, embeds.shape[0] - 1, embeds.shape[1])
    proj   = PCA(n_components=n_comp).fit_transform(embeds)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[:, 0], proj[:, 1], s=80, alpha=0.85)
    for i, d in enumerate(unique_d):
        short = d[:45] + "…" if len(d) > 45 else d
        ax.annotate(short, proj[i], fontsize=6, alpha=0.8)
    ax.set_title("Task Embedding PCA"); ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.3); fig.tight_layout()
    p = os.path.join(out_dir, "task_embed_pca.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Ablation helpers
# ══════════════════════════════════════════════════════════════════════════════

class _RolloutProxy:
    """Wraps a Rollout, replacing hidden_states for the task-concat ablation."""
    def __init__(self, original, new_hidden_states):
        self._orig         = original
        self.hidden_states = new_hidden_states

    def __getattr__(self, name):
        return getattr(self._orig, name)


def _make_task_concat_rollouts(rollouts, task_embeds):
    """Return proxy rollouts with task_embed broadcast-appended at every step."""
    result = []
    for r, emb in zip(rollouts, task_embeds):
        T     = r.hidden_states.shape[0]
        emb_t = torch.from_numpy(emb).unsqueeze(0).expand(T, -1)
        new_hs = torch.cat([r.hidden_states, emb_t], dim=-1)
        result.append(_RolloutProxy(r, new_hs))
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Detection-time analysis
# ══════════════════════════════════════════════════════════════════════════════

def compute_detection_curve(rollouts, scores_or_pairs, n_points=60):
    """
    Sweep thresholds 0→1.  At each threshold return:
      recall   — fraction of failures that triggered before episode end
      avg_det  — mean normalised detection time over *all* failures
                 (missed episodes are penalised as 1.0)
      far      — false-alarm rate on successes
    """
    thresholds = np.linspace(0.0, 1.0, n_points)
    recalls, det_times, fars = [], [], []
    for tau in thresholds:
        n_fail = n_succ = n_fa = 0
        times  = []
        for r, item in zip(rollouts, scores_or_pairs):
            s = item[0] if isinstance(item, tuple) else item
            if r.episode_success:
                n_succ += 1
                if float(s[-1]) >= tau:
                    n_fa += 1
            else:
                n_fail += 1
                exceed = np.where(s >= tau)[0]
                if len(exceed) > 0:
                    times.append(exceed[0] / max(len(s) - 1, 1))
                else:
                    times.append(1.0)   # missed → penalise as full episode
        recalls.append(sum(t < 1.0 for t in times) / max(n_fail, 1))
        det_times.append(float(np.mean(times)) if times else 1.0)
        fars.append(n_fa / max(n_succ, 1))
    return np.array(recalls), np.array(det_times), np.array(fars)


def plot_detection_time_curve(test_r, base_scores, comb_out, out_dir,
                               extra_curves=None):
    """
    Two-panel figure:
      Left : avg normalised detection time (↓) vs recall (↑)  — Pareto front
      Right: FAR (↓) vs recall (↑)
    extra_curves: list of (scores_or_pairs, label, color) for ablation models
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    all_series = [
        (base_scores, "Base",     "steelblue",  "-"),
        (comb_out,    "Combined", "darkorange", "-"),
    ]
    if extra_curves:
        for sc, lbl, col in extra_curves:
            all_series.append((sc, lbl, col, "--"))

    for scores, label, color, ls in all_series:
        rec, dts, far = compute_detection_curve(test_r, scores)
        axes[0].plot(dts, rec, color=color, lw=2, ls=ls, label=label)
        axes[1].plot(rec, far, color=color, lw=2, ls=ls, label=label)

    axes[0].set_xlabel("Avg normalised detection time  (0=instant, 1=end of ep)")
    axes[0].set_ylabel("Recall  (fraction of failures caught)")
    axes[0].set_title("Early Detection Tradeoff\n(upper-left corner = best)")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim(-0.02, 1.02); axes[0].set_ylim(-0.02, 1.02)

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("False-alarm rate  ↓")
    axes[1].set_title("Recall–FAR Tradeoff")
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(-0.02, 1.02); axes[1].set_ylim(-0.02, 1.02)

    fig.suptitle("Detection Time Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "detection_time_curve.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Hidden-state PCA
# ══════════════════════════════════════════════════════════════════════════════

def plot_hidden_state_pca(all_r, out_dir, n_samples=600, seed=42):
    """
    2-D PCA of mean-pooled hidden states.
    Left panel  : coloured by success/failure outcome
    Right panel : coloured by task ID
    """
    rng     = np.random.RandomState(seed)
    indices = np.arange(len(all_r))
    if len(indices) > n_samples:
        indices = rng.choice(indices, n_samples, replace=False)

    hs       = np.stack([all_r[i].hidden_states.float().mean(0).numpy()
                         for i in indices])
    outcomes = np.array([all_r[i].episode_success for i in indices])
    task_ids = np.array([all_r[i].task_id          for i in indices])

    pca = PCA(n_components=2, random_state=seed)
    X2  = pca.fit_transform(hs)
    var = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for outcome, label, color, marker in [
        (1, "Success", "seagreen", "o"),
        (0, "Failure", "crimson",  "x"),
    ]:
        mask = outcomes == outcome
        axes[0].scatter(X2[mask, 0], X2[mask, 1],
                        c=color, label=f"{label} (n={mask.sum()})",
                        s=14, alpha=0.55, marker=marker)
    axes[0].set_title("Hidden State PCA — by Outcome")
    axes[0].set_xlabel(f"PC1 ({var[0]:.1%} var)")
    axes[0].set_ylabel(f"PC2 ({var[1]:.1%} var)")
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

    unique_tasks = sorted(set(task_ids))
    cmap = plt.cm.get_cmap("tab10", len(unique_tasks))
    for i, tid in enumerate(unique_tasks):
        mask = task_ids == tid
        axes[1].scatter(X2[mask, 0], X2[mask, 1],
                        color=cmap(i), label=f"Task {tid}",
                        s=14, alpha=0.55)
    axes[1].set_title("Hidden State PCA — by Task")
    axes[1].set_xlabel(f"PC1 ({var[0]:.1%} var)")
    axes[1].set_ylabel(f"PC2 ({var[1]:.1%} var)")
    axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.3)

    fig.suptitle("OpenVLA Hidden State Feature Space", fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "hidden_state_pca.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Precision-recall curve
# ══════════════════════════════════════════════════════════════════════════════

def plot_prc_comparison(test_r, base_scores, comb_out, out_dir,
                        extra_curves=None):
    y_true = np.array([1 - r.episode_success for r in test_r])
    if len(np.unique(y_true)) < 2:
        return
    y_base = np.array([s[-1]      for s in base_scores])
    y_comb = np.array([s[-1] for s, _ in comb_out])

    fig, ax = plt.subplots(figsize=(5, 5))

    all_series = [
        (y_base, "Base",     "steelblue"),
        (y_comb, "Combined", "darkorange"),
    ]
    if extra_curves:
        for ys, lbl, col in extra_curves:
            all_series.append((ys, lbl, col))

    for ys, label, color in all_series:
        ap          = average_precision_score(y_true, ys)
        pre, rec, _ = precision_recall_curve(y_true, ys)
        ax.plot(rec, pre, lw=2, color=color, label=f"{label}  AP={ap:.3f}")

    base_rate = float(y_true.mean())
    ax.axhline(base_rate, color="gray", linestyle="--", lw=1,
               label=f"Chance ({base_rate:.2f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve"); ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3); ax.set_xlim(0, 1); ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = os.path.join(out_dir, "prc_comparison.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Ablation table
# ══════════════════════════════════════════════════════════════════════════════

def _compute_model_metrics(rollouts, scores, tau):
    """Return {auc, ap, acc, recall, far, avg_det} for a scored test set."""
    y_true = np.array([1 - r.episode_success for r in rollouts])
    finals = np.array([
        float(s[0][-1]) if isinstance(s, tuple) else float(s[-1])
        for s in scores
    ])
    has_both = len(np.unique(y_true)) > 1
    auc_val  = roc_auc_score(y_true, finals)          if has_both else None
    ap_val   = average_precision_score(y_true, finals) if has_both else None
    acc      = float(((finals >= 0.5).astype(int) == y_true).mean())
    nf, ns   = int(y_true.sum()), int((1 - y_true).sum())
    yp       = (finals >= tau).astype(int)
    recall   = float((yp[y_true == 1] == 1).sum() / max(nf, 1))
    far      = float((yp[y_true == 0] == 1).sum() / max(ns, 1))
    times    = []
    for r, item in zip(rollouts, scores):
        s = item[0] if isinstance(item, tuple) else item
        if not r.episode_success:
            exceed = np.where(s >= tau)[0]
            times.append(exceed[0] / max(len(s) - 1, 1) if len(exceed) > 0 else 1.0)
    avg_det = float(np.mean(times)) if times else 1.0
    return {"auc": auc_val, "ap": ap_val, "acc": acc,
            "recall": recall, "far": far, "avg_det": avg_det}


def plot_ablation_table(ablation_results, out_dir):
    """Render the ablation results as a styled matplotlib table."""
    col_labels = ["Model", "AUC ↑", "AP ↑", "Acc@0.5 ↑",
                  "Recall@τ ↑", "FAR@τ ↓", "Det.Time ↓"]
    rows = []
    for name, m in ablation_results.items():
        rows.append([
            name,
            f"{m['auc']:.4f}" if m["auc"] is not None else "—",
            f"{m['ap']:.4f}"  if m["ap"]  is not None else "—",
            f"{m['acc']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['far']:.4f}",
            f"{m['avg_det']:.4f}",
        ])

    fig, ax = plt.subplots(figsize=(14, 1.6 + 0.8 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 2.5)

    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2c5f8a")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        bg = "#eef4fb" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
    for j in range(len(col_labels)):           # highlight last row (full combined)
        tbl[len(rows), j].set_facecolor("#fff0cc")
        tbl[len(rows), j].set_text_props(fontweight="bold")

    ax.set_title("Ablation Study: Contribution of Each Component",
                 fontsize=12, fontweight="bold", pad=14, y=0.98)
    fig.tight_layout()
    p = os.path.join(out_dir, "ablation_table.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {p}")


def plot_summary(test_r, base_scores, comb_out,
                 base_losses, comb_losses,
                 tau_base, tau_comb,
                 calib_r, calib_base, calib_comb,
                 out_dir):
    """9-panel (3×3) summary dashboard."""
    fig = plt.figure(figsize=(18, 14))
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)
    x   = np.linspace(0, 1, 100)

    # ── Panel 1: training loss ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(base_losses, lw=1.5, color="steelblue",   label="Base")
    ax.plot(comb_losses, lw=1.5, color="darkorange", label="Combined")
    ax.set_title("Training Loss"); ax.set_xlabel("Epoch"); ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: score curves (combined, test) ───────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    cs, cf = _score_series(test_r, comb_out)
    _draw_score_curves(ax, cs, cf, "Score Curves — Combined (test)", x)

    # ── Panel 3: ROC comparison ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    y_true = np.array([1 - r.episode_success for r in test_r])
    y_base = np.array([s[-1]      for s in base_scores])
    y_comb = np.array([s[-1]      for s, _ in comb_out])
    if len(np.unique(y_true)) > 1:
        for ys, label, color in [
            (y_base, f"Base  AUC={roc_auc_score(y_true,y_base):.3f}", "steelblue"),
            (y_comb, f"Comb  AUC={roc_auc_score(y_true,y_comb):.3f}", "darkorange"),
        ]:
            fpr, tpr, _ = roc_curve(y_true, ys)
            ax.plot(fpr, tpr, lw=2, color=color, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title("ROC Comparison"); ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── Panel 4: PRC comparison ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    if len(np.unique(y_true)) > 1:
        for ys, label, color in [
            (y_base, f"Base  AP={average_precision_score(y_true,y_base):.3f}", "steelblue"),
            (y_comb, f"Comb  AP={average_precision_score(y_true,y_comb):.3f}", "darkorange"),
        ]:
            pre, rec, _ = precision_recall_curve(y_true, ys)
            ax.plot(rec, pre, lw=2, color=color, label=label)
        ax.axhline(float(y_true.mean()), color="gray", lw=1, linestyle="--")
    ax.set_title("Precision-Recall Curve"); ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3); ax.set_xlim(0,1); ax.set_ylim(0,1.05)

    # ── Panel 5: detection time tradeoff ────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    for scores, label, color, ls in [
        (base_scores, "Base",     "steelblue",  "-"),
        (comb_out,    "Combined", "darkorange", "-"),
    ]:
        rec, dts, _ = compute_detection_curve(test_r, scores, n_points=40)
        ax.plot(dts, rec, color=color, lw=2, ls=ls, label=label)
    ax.set_xlabel("Avg det. time"); ax.set_ylabel("Recall")
    ax.set_title("Early Detection Tradeoff\n(upper-left = best)")
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    # ── Panel 6: attention weights ───────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    succ_w, fail_w = [], []
    for r, (_, w) in zip(test_r, comb_out):
        (succ_w if r.episode_success else fail_w).append(_interp(w, 100))
    for curves, label, color in [(succ_w, "Success", "seagreen"),
                                  (fail_w, "Failure", "crimson")]:
        if not curves:
            continue
        arr = np.stack(curves)
        ax.plot(x, arr.mean(0), color=color, lw=2, label=label)
        ax.fill_between(x, arr.mean(0) - arr.std(0),
                        arr.mean(0) + arr.std(0), color=color, alpha=0.25)
    ax.set_title("Attention Weights (Combined)"); ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Normalised time"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── Panel 7: conformal coverage ──────────────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    alphas = np.linspace(0.02, 0.40, 35)
    for calib_s, test_s, label, color in [
        (calib_base, base_scores, "Base",     "steelblue"),
        (calib_comb, comb_out,    "Combined", "darkorange"),
    ]:
        recalls = []
        for a in alphas:
            try:
                tau = calibrate_threshold(calib_r, calib_s, a)
                rec, _, _ = _eval_at_thresh(test_r, test_s, tau)
            except Exception:
                rec = np.nan
            recalls.append(rec)
        ax.plot(1 - alphas, recalls, color=color, lw=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set_title("Conformal Coverage"); ax.set_xlabel("Target recall")
    ax.set_ylabel("Empirical recall"); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # ── Panel 8: final score histogram (combined) ────────────────────────────
    ax  = fig.add_subplot(gs[2, 1])
    ax2 = ax.twinx()
    bins = np.linspace(0, 1, 25)
    succ_f = [s[-1] for r, (s, _) in zip(test_r, comb_out) if     r.episode_success]
    fail_f = [s[-1] for r, (s, _) in zip(test_r, comb_out) if not r.episode_success]
    if succ_f:
        ax.hist(succ_f,  bins=bins, color="seagreen", alpha=0.6,
                label=f"Succ (n={len(succ_f)})", density=True)
    if fail_f:
        ax2.hist(fail_f, bins=bins, color="crimson", alpha=0.5,
                 label=f"Fail (n={len(fail_f)})", density=True)
    ax.axvline(0.5,      color="orange", linestyle="--", lw=1.5, label="τ=0.5")
    ax.axvline(tau_comb, color="purple", linestyle="-.", lw=1.5,
               label=f"τ_conf={tau_comb:.2f}")
    ax.set_xlabel("Final score"); ax.set_ylabel("Density (succ)", color="seagreen")
    ax2.set_ylabel("Density (fail)", color="crimson")
    ax.tick_params(axis="y", labelcolor="seagreen")
    ax2.tick_params(axis="y", labelcolor="crimson")
    ax.set_title("Score Distribution — Combined")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=6)
    ax.grid(True, alpha=0.3)

    # ── Panel 9: score curves (base, test) ───────────────────────────────────
    ax = fig.add_subplot(gs[2, 2])
    bs, bf = _score_series(test_r, base_scores)
    _draw_score_curves(ax, bs, bf, "Score Curves — Base (test)", x)

    fig.suptitle("Combined Detector — Dashboard", fontsize=14, fontweight="bold")
    fig.savefig(os.path.join(out_dir, "summary.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {os.path.join(out_dir, 'summary.png')}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Combined task-conditioned + attention + conformal detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",         required=True)
    parser.add_argument("--output_dir",        default="./combined_results")
    parser.add_argument("--train_ratio",       type=float, default=0.60,
                        help="Fraction of SEEN episodes used for training")
    parser.add_argument("--calib_ratio",       type=float, default=0.20,
                        help="Fraction of SEEN episodes used for conformal calibration")
    parser.add_argument("--unseen_task_ratio", type=float, default=0.30,
                        help="Fraction of task IDs held out entirely for testing. "
                             "Set to 0.0 to fall back to a random episode split "
                             "(faster but inflated metrics due to task leakage).")
    parser.add_argument("--target_recall",  type=float, default=0.90,
                        help="Desired recall for the conformal threshold")
    parser.add_argument("--task_embed_dim", type=int,   default=32)
    parser.add_argument("--n_epochs",       type=int,   default=300)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--lambda_reg",     type=float, default=1e-2)
    parser.add_argument("--lambda_attn",    type=float, default=0.1,
                        help="Weight of end-of-episode attention loss term")
    parser.add_argument("--hidden_dim",     type=int,   default=256)
    parser.add_argument("--n_layers",       type=int,   default=2)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--run_ablation",   action="store_true", default=True,
                        help="Train +task-only and +attn-only models for ablation table")
    parser.add_argument("--no_ablation",    dest="run_ablation", action="store_false")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    print("\n[1/7] Loading rollouts ...")
    all_r     = load_rollouts(args.data_path)
    input_dim = all_r[0].hidden_states.shape[-1]

    # ── 2. Task embeddings ───────────────────────────────────────────────────
    print("\n[2/7] Building task embeddings ...")
    encoder = TaskEncoder(n_components=args.task_embed_dim)
    encoder.fit(list(dict.fromkeys(r.task_description for r in all_r)))
    all_emb = encoder.transform([r.task_description for r in all_r])
    E       = all_emb.shape[-1]   # actual (possibly capped) embed dim

    # ── 3. Split ─────────────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)

    def _gather(idx):
        idx = list(idx)
        return ([all_r[i] for i in idx],
                all_emb[idx] if len(idx) else np.zeros((0, E), np.float32))

    def _counts(r):
        ns = sum(x.episode_success for x in r)
        return f"{len(r)} ({ns} succ / {len(r)-ns} fail)"

    if args.unseen_task_ratio > 0:
        # ── Seen / unseen TASK split ─────────────────────────────────────────
        # Completely hold out a fraction of task IDs.
        # Train + calibrate on seen tasks only; test EXCLUSIVELY on unseen tasks.
        # This gives an honest generalisation estimate with no task leakage.
        all_task_ids = sorted(set(r.task_id for r in all_r))
        shuffled_ids = all_task_ids.copy()
        rng.shuffle(shuffled_ids)
        n_unseen    = max(1, round(args.unseen_task_ratio * len(all_task_ids)))
        unseen_ids  = set(shuffled_ids[:n_unseen])
        seen_ids    = set(shuffled_ids[n_unseen:])

        seen_idx   = [i for i, r in enumerate(all_r) if r.task_id in seen_ids]
        unseen_idx = [i for i, r in enumerate(all_r) if r.task_id in unseen_ids]
        rng.shuffle(seen_idx)

        # Split seen into train / calibration (remaining seen → optional seen-test)
        seen_s_idx = [i for i in seen_idx if     all_r[i].episode_success]
        seen_f_idx = [i for i in seen_idx if not all_r[i].episode_success]
        # Use train_ratio / (train_ratio + calib_ratio) of seen for train
        split_frac = args.train_ratio / (args.train_ratio + args.calib_ratio + 1e-9)
        n_tr_s = max(1, int(len(seen_s_idx) * split_frac))
        n_tr_f = max(1, int(len(seen_f_idx) * split_frac))

        tr_idx = seen_s_idx[:n_tr_s] + seen_f_idx[:n_tr_f]
        ca_idx = seen_s_idx[n_tr_s:]  + seen_f_idx[n_tr_f:]
        te_idx = unseen_idx    # ← entirely new tasks

        print(f"\n  Seen   tasks ({len(seen_ids)}):   {sorted(seen_ids)}")
        print(f"  Unseen tasks ({len(unseen_ids)}): {sorted(unseen_ids)}  "
              f"← test set")
    else:
        # ── Random episode split (backward-compat, inflated metrics) ─────────
        print("\n  WARNING: --unseen_task_ratio=0  →  random episode split."
              " Metrics are inflated due to task leakage.")
        s_idx = [i for i, r in enumerate(all_r) if     r.episode_success]
        f_idx = [i for i, r in enumerate(all_r) if not r.episode_success]
        rng.shuffle(s_idx); rng.shuffle(f_idx)

        def _split3(idx):
            n    = len(idx)
            n_tr = max(1, int(n * args.train_ratio))
            n_ca = max(1, int(n * args.calib_ratio))
            return idx[:n_tr], idx[n_tr:n_tr+n_ca], idx[n_tr+n_ca:]

        s_tr, s_ca, s_te = _split3(s_idx)
        f_tr, f_ca, f_te = _split3(f_idx)
        tr_idx = s_tr + f_tr
        ca_idx = s_ca + f_ca
        te_idx = s_te + f_te

    train_r, train_emb = _gather(tr_idx)
    calib_r, calib_emb = _gather(ca_idx)
    test_r,  test_emb  = _gather(te_idx)

    print(f"  Train: {_counts(train_r)}")
    print(f"  Calib: {_counts(calib_r)}")
    print(f"  Test:  {_counts(test_r)}"
          + (" (unseen tasks)" if args.unseen_task_ratio > 0 else ""))

    # ── 4. Train base ────────────────────────────────────────────────────────
    print("\n[3/7] Training base detector ...")
    base = FailureDetector(input_dim, hidden_dim=args.hidden_dim,
                           n_layers=args.n_layers).to(args.device)
    base_losses = train_model(base, train_r, n_epochs=args.n_epochs,
                              lr=args.lr, lambda_reg=args.lambda_reg,
                              batch_size=args.batch_size, device=args.device)

    # ── 5. Train combined ────────────────────────────────────────────────────
    print("\n[4/7] Training combined detector "
          f"(task-cond {E}-d + attn, λ_attn={args.lambda_attn}) ...")
    comb = CombinedFailureDetector(
        hidden_state_dim=input_dim,
        task_embed_dim=E,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(args.device)
    comb_losses = _train_combined(
        comb, train_r, train_emb,
        n_epochs=args.n_epochs, lr=args.lr,
        lambda_reg=args.lambda_reg, lambda_attn=args.lambda_attn,
        batch_size=args.batch_size, device=args.device,
    )

    # ── 6. Score all splits ──────────────────────────────────────────────────
    print("\n[5/7] Scoring ...")
    base_train = predict(base, train_r, device=args.device)
    base_calib = predict(base, calib_r, device=args.device)
    base_test  = predict(base, test_r,  device=args.device)

    comb_calib = _predict_combined(comb, calib_r, calib_emb, device=args.device)
    comb_test  = _predict_combined(comb, test_r,  test_emb,  device=args.device)

    # Conformal calibration
    alpha    = 1.0 - args.target_recall
    tau_base = calibrate_threshold(calib_r, base_calib, alpha)
    tau_comb = calibrate_threshold(calib_r, comb_calib, alpha)

    # ── 6b. Ablation: +task-only and +attn-only ───────────────────────────────
    task_test = attn_test = task_calib = attn_calib = None
    tau_task  = tau_attn  = 0.5
    if args.run_ablation and E > 0:
        print("\n[Ablation] Training +task-cond, no-attention model ...")
        aug_train = _make_task_concat_rollouts(train_r, train_emb)
        aug_calib = _make_task_concat_rollouts(calib_r, calib_emb)
        aug_test  = _make_task_concat_rollouts(test_r,  test_emb)
        task_base = FailureDetector(input_dim + E,
                                    hidden_dim=args.hidden_dim,
                                    n_layers=args.n_layers).to(args.device)
        train_model(task_base, aug_train, n_epochs=args.n_epochs,
                    lr=args.lr, lambda_reg=args.lambda_reg,
                    batch_size=args.batch_size, device=args.device)
        task_calib = predict(task_base, aug_calib, device=args.device)
        task_test  = predict(task_base, aug_test,  device=args.device)
        tau_task   = calibrate_threshold(calib_r, task_calib, alpha)

        print("\n[Ablation] Training +attention-only, no-task model ...")
        zero_tr = np.zeros((len(train_r), 0), dtype=np.float32)
        zero_ca = np.zeros((len(calib_r), 0), dtype=np.float32)
        zero_te = np.zeros((len(test_r),  0), dtype=np.float32)
        attn_model = CombinedFailureDetector(
            hidden_state_dim=input_dim, task_embed_dim=0,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        ).to(args.device)
        _train_combined(attn_model, train_r, zero_tr,
                        n_epochs=args.n_epochs, lr=args.lr,
                        lambda_reg=args.lambda_reg, lambda_attn=args.lambda_attn,
                        batch_size=args.batch_size, device=args.device)
        attn_calib = _predict_combined(attn_model, calib_r, zero_ca, device=args.device)
        attn_test  = _predict_combined(attn_model, test_r,  zero_te, device=args.device)
        tau_attn   = calibrate_threshold(calib_r, attn_calib, alpha)

    # ── 7. Results ───────────────────────────────────────────────────────────
    print("\n[6/7] Evaluating & reporting ...")
    y_true  = np.array([1 - r.episode_success for r in test_r])
    y_base  = np.array([s[-1]      for s in base_test])
    y_comb  = np.array([s[-1]      for s, _ in comb_test])

    def _report(title, ys, tau):
        lines.append(f"\n─── {title} ───")
        if len(np.unique(y_true)) > 1:
            lines.append(f"  AUC:                 {roc_auc_score(y_true, ys):.4f}")
            lines.append(f"  AP:                  {average_precision_score(y_true, ys):.4f}")
        yt = y_true
        acc_f  = float(((ys >= 0.5).astype(int) == yt).mean())
        nf, ns = yt.sum(), (1 - yt).sum()
        yp_tau = (ys >= tau).astype(int)
        rec = float((yp_tau[yt == 1] == 1).sum() / max(nf, 1))
        far = float((yp_tau[yt == 0] == 1).sum() / max(ns, 1))
        acc_c = float((yp_tau == yt).mean())
        times = []
        for r, item in zip(test_r, [s for s in base_test] if ys is y_base else
                           [(s, _) for s, _ in comb_test]):
            s_arr = item if isinstance(item, np.ndarray) else item[0]
            if not r.episode_success:
                exceed = np.where(s_arr >= tau)[0]
                times.append(exceed[0] / max(len(s_arr)-1, 1) if len(exceed)>0 else 1.0)
        avg_det = float(np.mean(times)) if times else 1.0
        lines.append(f"  Accuracy @ 0.50:     {acc_f:.4f}")
        lines.append(f"  Conformal τ={tau:.4f}: recall={rec:.4f}  FAR={far:.4f}  "
                     f"acc={acc_c:.4f}  avg_det={avg_det:.4f}")

    lines = []
    _report("Base detector",     y_base, tau_base)
    _report("Combined detector", y_comb, tau_comb)

    # Ablation rows in text report
    if task_test is not None:
        y_task = np.array([s[-1] for s in task_test])
        _report("+Task (no attn)", y_task, tau_task)
    if attn_test is not None:
        y_attn = np.array([s[-1] for s, _ in attn_test])
        _report("+Attn (no task)", y_attn, tau_attn)

    print("\n" + "=" * 60)
    print("COMBINED DETECTOR — RESULTS")
    print("=" * 60)
    for l in lines:
        print(l)
    print("=" * 60)

    # Save summary text
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  -> {summary_path}")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    print("\n[7/7] Generating plots ...")
    plot_training_loss(base_losses, comb_losses, args.output_dir)
    plot_roc_comparison(test_r, base_test, comb_test, args.output_dir)
    plot_score_curves(test_r, base_test, comb_test, args.output_dir)
    plot_attention_weights(test_r, comb_test, args.output_dir)
    plot_attention_heatmap(test_r, comb_test, args.output_dir)
    plot_coverage_curve(calib_r, comb_calib, base_calib,
                        test_r, comb_test, base_test, args.output_dir)
    plot_conformal_histogram(test_r, base_test, comb_test,
                              tau_base, tau_comb, args.output_dir)
    plot_per_task_auc(test_r, base_test, comb_test, args.output_dir)
    plot_task_embed_pca(all_r, encoder, args.output_dir)

    # New analysis plots
    ablation_extra = []
    if task_test is not None:
        ablation_extra.append((task_test, "+Task only", "mediumseagreen"))
    if attn_test is not None:
        ablation_extra.append((attn_test, "+Attn only", "mediumpurple"))

    plot_detection_time_curve(test_r, base_test, comb_test, args.output_dir,
                               extra_curves=ablation_extra if ablation_extra else None)
    plot_hidden_state_pca(all_r, args.output_dir)
    plot_prc_comparison(test_r, base_test, comb_test, args.output_dir,
                        extra_curves=(
                            [(np.array([s[-1] for s in task_test]),
                              "+Task only", "mediumseagreen")]
                            if task_test is not None else None
                        ))

    # Ablation table
    if args.run_ablation:
        ablation_results = {
            "Base (MLP, running mean)":
                _compute_model_metrics(test_r, base_test, tau_base),
        }
        if task_test is not None:
            ablation_results["+Task cond (MLP+task, running mean)"] = \
                _compute_model_metrics(test_r, task_test, tau_task)
        if attn_test is not None:
            ablation_results["+Attn only (no task, causal attn)"] = \
                _compute_model_metrics(test_r, attn_test, tau_attn)
        ablation_results["Combined (+task +attn, conformal)"] = \
            _compute_model_metrics(test_r, comb_test, tau_comb)
        plot_ablation_table(ablation_results, args.output_dir)

    plot_summary(test_r, base_test, comb_test,
                 base_losses, comb_losses,
                 tau_base, tau_comb,
                 calib_r, base_calib, comb_calib,
                 args.output_dir)

    print(f"\n  All outputs saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
