#!/usr/bin/env python3
"""
Standalone failure detector for OpenVLA LIBERO rollouts.

Loads hidden-state rollouts saved by vla-safe/openvla's run_libero_eval.py,
trains an MLP failure detector, then generates:

  output_dir/
    training_loss.png        - training loss curve
    score_curves_train.png   - mean +/- std score for successes vs failures (train)
    score_curves_test.png    - same for test split
    score_histogram.png      - distribution of final scores
    roc_curve.png            - ROC curve with AUC
    pca_scatter.png          - PCA of hidden states coloured by outcome
    per_task_auc.png         - per-task detection AUC bar chart
    summary.png              - 4-panel overview figure
    videos/                  - annotated MP4s (success=green border, failure=red)

Usage:
    python scripts/failure_detector.py \\
        --data_path /path/to/rollouts/single-foward/libero_10/ \\
        --save_videos

    # Custom hyper-params:
    python scripts/failure_detector.py \\
        --data_path /path/to/rollouts/single-foward/libero_10/ \\
        --n_epochs 500 --lr 3e-4 --hidden_dim 512 \\
        --save_videos --n_videos 20
"""

import os
import re
import glob
import pickle
import argparse

import cv2
import imageio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ──────────────────────────────────── Data ─────────────────────────────────────

def _parse_filename(fname: str):
    """task3--ep12--succ1.csv  ->  (task_id=3, ep_idx=12, success=True)"""
    m = re.match(r"task(\d+)--ep(\d+)--succ(\d+)\.csv", fname)
    if not m:
        raise ValueError(f"Cannot parse filename: {fname}")
    return int(m.group(1)), int(m.group(2)), bool(int(m.group(3)))


def _process_hidden_states(h) -> torch.Tensor:
    """Normalise to (T, D) regardless of saved format."""
    if isinstance(h, list):
        h = torch.stack(h, dim=0)
    h = h.float()
    if h.ndim == 3:          # (T, N_tokens, D) -> mean over tokens
        h = h.mean(dim=1)
    return h                 # (T, D)


class Rollout:
    __slots__ = ("hidden_states", "task_id", "episode_idx", "episode_success",
                 "task_description", "mp4_path")

    def __init__(self, hidden_states, task_id, episode_idx,
                 episode_success, task_description, mp4_path):
        self.hidden_states   = hidden_states       # torch.Tensor (T, D)
        self.task_id         = task_id
        self.episode_idx     = episode_idx
        self.episode_success = int(episode_success)
        self.task_description = task_description
        self.mp4_path        = mp4_path


def load_rollouts(data_path: str) -> list:
    csv_files = sorted(glob.glob(os.path.join(data_path, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {data_path!r}")

    rollouts = []
    for csv_path in tqdm(csv_files, desc="Loading rollouts"):
        fname = os.path.basename(csv_path)
        try:
            task_id, ep_idx, success = _parse_filename(fname)
        except ValueError:
            continue

        pkl_path = csv_path.replace(".csv", ".pkl")
        mp4_path = csv_path.replace(".csv", ".mp4")

        if not os.path.exists(pkl_path):
            continue

        with open(pkl_path, "rb") as f:
            meta = pickle.load(f)

        # Fix historical typo in some saved rollouts
        if "eposide_idx" in meta:
            meta["episode_idx"] = meta.pop("eposide_idx")

        hidden_states = _process_hidden_states(meta["hidden_states"])
        task_desc = meta.get("task_description", f"Task {task_id}")

        rollouts.append(Rollout(
            hidden_states=hidden_states,
            task_id=task_id,
            episode_idx=ep_idx,
            episode_success=success,
            task_description=task_desc,
            mp4_path=mp4_path if os.path.exists(mp4_path) else None,
        ))

    n_succ = sum(r.episode_success for r in rollouts)
    n_fail = len(rollouts) - n_succ
    print(f"Loaded {len(rollouts)} rollouts  ({n_succ} success / {n_fail} failure)")
    return rollouts


# ──────────────────────────────────── Dataset ──────────────────────────────────

def _pad_collate(batch):
    """Pad variable-length rollouts to the same sequence length."""
    max_len = max(b["features"].shape[0] for b in batch)
    D = batch[0]["features"].shape[-1]
    B = len(batch)

    features = torch.zeros(B, max_len, D)
    masks    = torch.zeros(B, max_len)
    labels   = torch.zeros(B)

    for i, b in enumerate(batch):
        T = b["features"].shape[0]
        features[i, :T] = b["features"]
        masks[i, :T]    = 1.0
        labels[i]       = b["label"]

    return {"features": features, "valid_masks": masks, "success_labels": labels}


class RolloutDataset(Dataset):
    def __init__(self, rollouts: list):
        self.rollouts = rollouts

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        r = self.rollouts[idx]
        return {
            "features": r.hidden_states,
            "label":    torch.tensor(float(r.episode_success)),
        }


# ──────────────────────────────────── Model ────────────────────────────────────

class FailureDetector(nn.Module):
    """
    Per-timestep MLP failure detector.

    Training  uses raw per-step sigmoid outputs with BCE loss — this prevents
    the score-collapse that occurs when training directly on the running mean.

    Inference returns a running-mean score in [0, 1] at each timestep so the
    visualisations accumulate evidence smoothly over the episode:
      Score -> 0  means the model thinks this rollout will succeed.
      Score -> 1  means the model thinks this rollout is failing.
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256,
                 n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(),
                       nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers += [nn.Linear(in_dim, 1), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

    def forward_raw(self, features: torch.Tensor) -> torch.Tensor:
        """Raw per-step failure probability (B, T) — used for training loss."""
        return self.mlp(features).squeeze(-1)

    def forward(self, features: torch.Tensor,
                valid_masks: torch.Tensor) -> torch.Tensor:
        """
        Running-mean failure score (B, T) — used for visualisation.
        Each position t holds the mean of raw outputs from step 0..t,
        giving a score that accumulates evidence as the episode progresses.
        Padded positions are zeroed out.
        """
        raw    = self.forward_raw(features)                   # (B, T)
        cum    = torch.cumsum(raw, dim=-1)                    # (B, T)
        t      = torch.arange(1, raw.shape[1] + 1,
                               device=raw.device).float()
        scores = cum / t.unsqueeze(0)                         # (B, T)
        return scores * valid_masks


def _compute_loss(raw_scores, masks, labels,
                  lambda_reg: float = 1e-2, model=None):
    """
    BCE loss on raw per-step outputs.

    Target for each step = 1 if the episode fails, 0 if it succeeds.
    Class imbalance is handled via pos_weight (inverse frequency ratio).

    Training on raw (pre-cumulation) outputs avoids the collapse seen when
    training on running-mean scores: the hinge loss on running-mean values
    requires pushing every single raw output to 1.0 for failures, which
    causes all scores to saturate.  BCE on raw outputs provides clean,
    local gradients at each timestep instead.
    """
    n_s = labels.sum().item() + 1e-6
    n_f = (1 - labels).sum().item() + 1e-6
    # pos_weight > 1 when successes outnumber failures (typical)
    pos_weight = torch.tensor(n_s / n_f, device=raw_scores.device)

    # Target: 1 = failure episode, 0 = success episode (same for every step)
    targets = (1 - labels).unsqueeze(1).expand_as(raw_scores)  # (B, T)

    bce = nn.functional.binary_cross_entropy(
        raw_scores, targets, reduction="none"
    )                                                          # (B, T)
    # Apply pos_weight manually (BCE expects logits for built-in pos_weight)
    weighted_bce = bce * (targets * pos_weight + (1 - targets))
    loss = (weighted_bce * masks).sum() / (masks.sum() + 1e-8)

    if model is not None and lambda_reg > 0:
        reg = sum(p.pow(2).sum()
                  for name, p in model.named_parameters() if "weight" in name)
        loss = loss + lambda_reg * reg

    return loss


def train_model(model, rollouts, n_epochs=300, lr=1e-3,
                lambda_reg=1e-2, batch_size=32, device="cpu"):
    dataset = RolloutDataset(rollouts)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=_pad_collate)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    model.train()
    losses = []
    pbar = tqdm(range(n_epochs), desc="Training", unit="ep")
    for _ in pbar:
        epoch_loss = 0.0
        for batch in loader:
            feat = batch["features"].to(device)
            mask = batch["valid_masks"].to(device)
            lbl  = batch["success_labels"].to(device)

            opt.zero_grad()
            # Train on raw per-step outputs (not the cumulated visualisation score)
            raw_scores = model.forward_raw(feat)
            loss = _compute_loss(raw_scores, mask, lbl, lambda_reg, model)
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
def predict(model, rollouts, device="cpu"):
    """Returns a list of 1-D numpy arrays -- one score sequence per rollout."""
    model.eval()
    out = []
    for r in rollouts:
        feat = r.hidden_states.unsqueeze(0).to(device)
        mask = torch.ones(1, feat.shape[1], device=device)
        s    = model(feat, mask).squeeze(0).cpu().numpy()
        out.append(s)
    return out


# ──────────────────────────────── Visualisations ───────────────────────────────

def _interp_to_n(scores, n=100):
    """Interpolate a score sequence to a fixed length for alignment."""
    return np.interp(
        np.linspace(0, len(scores) - 1, n),
        np.arange(len(scores)),
        scores,
    )


def plot_training_loss(losses, save_path):
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(losses, lw=1.5)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training Loss"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  -> {save_path}")


def plot_score_curves(rollouts, scores, title, save_path):
    """
    Mean +/- std failure score over normalised time for each outcome.
    Individual rollout traces are drawn faintly behind the mean so that
    curves at the extremes (near 0 or 1) are always visible.
    """
    n = 100
    succ, fail = [], []
    for r, s in zip(rollouts, scores):
        (succ if r.episode_success else fail).append(_interp_to_n(s, n))

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.linspace(0, 1, n)

    for curves, label, color in [
        (succ, f"Success  (n={len(succ)})", "seagreen"),
        (fail, f"Failure  (n={len(fail)})", "crimson"),
    ]:
        if not curves:
            continue
        arr = np.stack(curves)

        # Draw individual rollout traces faintly so lines at 0/1 are visible
        for trace in arr:
            ax.plot(x, trace, color=color, lw=0.4, alpha=0.15)

        mean, std = arr.mean(0), arr.std(0)
        ax.plot(x, mean, color=color, lw=2.5, label=label, zorder=3)
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.25)

    ax.axhline(0.5, color="orange", linestyle="--", lw=1, label="Threshold 0.5")
    ax.set_xlabel("Normalised time  (0 = start, 1 = end)")
    ax.set_ylabel("Cumulative failure score")
    ax.set_title(title); ax.legend()
    # Pad y-axis so lines pressed against 0 or 1 are not hidden by the axis spine
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  -> {save_path}")


def plot_roc_curve(rollouts, scores, save_path):
    y_true  = np.array([1 - r.episode_success for r in rollouts])  # 1 = failure
    y_score = np.array([s[-1] for s in scores])

    if len(np.unique(y_true)) < 2:
        print("  (skipping ROC -- only one class in test set)")
        return None

    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc     = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Failure Detection -- ROC Curve")
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  -> {save_path}  (AUC={roc_auc:.3f})")
    return roc_auc


def plot_score_histogram(rollouts, scores, save_path):
    succ = [s[-1] for r, s in zip(rollouts, scores) if     r.episode_success]
    fail = [s[-1] for r, s in zip(rollouts, scores) if not r.episode_success]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bins = np.linspace(0, 1, 30)

    # Left: overlapping histogram (shared density scale)
    ax = axes[0]
    if succ:
        ax.hist(succ, bins=bins, color="seagreen", alpha=0.6,
                label=f"Success  (n={len(succ)})", density=True)
    if fail:
        ax.hist(fail, bins=bins, color="crimson",  alpha=0.6,
                label=f"Failure  (n={len(fail)})",  density=True)
    ax.axvline(0.5, color="orange", linestyle="--", lw=1.5, label="Threshold 0.5")
    ax.set_xlabel("Final failure score"); ax.set_ylabel("Density")
    ax.set_title("Score Distribution (shared axis)")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Right: separate y-axes so both distributions are readable even when one
    # is much taller than the other (common with imbalanced classes)
    ax2 = axes[1]
    ax2_r = ax2.twinx()
    if succ:
        ax2.hist(succ, bins=bins, color="seagreen", alpha=0.6,
                 label=f"Success  (n={len(succ)})", density=True)
    if fail:
        ax2_r.hist(fail, bins=bins, color="crimson", alpha=0.5,
                   label=f"Failure  (n={len(fail)})", density=True)
    ax2.axvline(0.5, color="orange", linestyle="--", lw=1.5)
    ax2.set_xlabel("Final failure score")
    ax2.set_ylabel("Density (success)", color="seagreen")
    ax2_r.set_ylabel("Density (failure)", color="crimson")
    ax2.tick_params(axis="y", labelcolor="seagreen")
    ax2_r.tick_params(axis="y", labelcolor="crimson")
    ax2.set_title("Score Distribution (independent axes)")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_r.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  -> {save_path}")


def plot_pca_scatter(rollouts, save_path):
    """Per-rollout PCA of mean hidden state, coloured by outcome."""
    feats  = np.stack([r.hidden_states.float().mean(0).numpy() for r in rollouts])
    labels = np.array([r.episode_success for r in rollouts])

    pca  = PCA(n_components=2)
    proj = pca.fit_transform(feats)

    fig, ax = plt.subplots(figsize=(6, 6))
    for val, color, name in [(1, "seagreen", "Success"), (0, "crimson", "Failure")]:
        m = labels == val
        ax.scatter(proj[m, 0], proj[m, 1], c=color,
                   label=f"{name}  (n={m.sum()})",
                   alpha=0.75, s=25, edgecolors="none")
    ax.set_xlabel(f"PC 1  ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC 2  ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA of Mean Hidden States per Rollout")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  -> {save_path}")


def plot_per_task_auc(rollouts, scores, save_path):
    """Bar chart: per-task ROC-AUC for failure detection."""
    task_ids = sorted(set(r.task_id for r in rollouts))
    aucs, names, counts = [], [], []

    for tid in task_ids:
        idx   = [i for i, r in enumerate(rollouts) if r.task_id == tid]
        yt    = np.array([1 - rollouts[i].episode_success for i in idx])
        ys    = np.array([scores[i][-1] for i in idx])

        if len(np.unique(yt)) < 2:
            aucs.append(float("nan"))
        else:
            aucs.append(roc_auc_score(yt, ys))
        names.append(f"T{tid}")
        counts.append(len(idx))

    valid = [(a, n, c) for a, n, c in zip(aucs, names, counts)
             if not np.isnan(a)]
    if not valid:
        return

    aucs_v, names_v, counts_v = zip(*valid)
    colors = ["seagreen" if a >= 0.5 else "crimson" for a in aucs_v]

    fig, ax = plt.subplots(figsize=(max(6, len(valid) * 0.7), 4))
    bars = ax.bar(names_v, aucs_v, color=colors, alpha=0.85)
    ax.axhline(0.5, color="gray", linestyle="--", lw=1)

    for bar, cnt in zip(bars, counts_v):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"n={cnt}", ha="center", va="bottom", fontsize=7)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("ROC-AUC  (failure detection)")
    ax.set_title("Per-Task Failure Detection AUC")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  -> {save_path}")


def plot_summary(train_r, test_r, train_s, test_s, save_path):
    """4-panel overview: score curves x2, histogram, PCA."""
    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)
    n   = 100
    x   = np.linspace(0, 1, n)

    def _score_curves(ax, rollouts, scores, title):
        succ = [_interp_to_n(s, n) for r, s in zip(rollouts, scores) if     r.episode_success]
        fail = [_interp_to_n(s, n) for r, s in zip(rollouts, scores) if not r.episode_success]
        for curves, label, color in [
            (succ, f"Success (n={len(succ)})", "seagreen"),
            (fail, f"Failure (n={len(fail)})", "crimson"),
        ]:
            if not curves:
                continue
            arr = np.stack(curves)
            # Faint individual traces so extreme-value curves are visible
            for trace in arr:
                ax.plot(x, trace, color=color, lw=0.4, alpha=0.12)
            mean, std = arr.mean(0), arr.std(0)
            ax.plot(x, mean, color=color, lw=2, label=label, zorder=3)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.2)
        ax.axhline(0.5, color="orange", linestyle="--", lw=1)
        ax.set_xlabel("Normalised time"); ax.set_ylabel("Failure score")
        ax.set_title(title); ax.legend(fontsize=8)
        ax.set_ylim(-0.05, 1.05)   # pad so lines at 0/1 are not hidden
        ax.grid(True, alpha=0.3)

    _score_curves(fig.add_subplot(gs[0, 0]), train_r, train_s, "Score Curves -- Train")
    _score_curves(fig.add_subplot(gs[0, 1]), test_r,  test_s,  "Score Curves -- Test")

    # Histogram with independent y-axes so both distributions are readable
    ax_h  = fig.add_subplot(gs[1, 0])
    ax_h2 = ax_h.twinx()
    bins  = np.linspace(0, 1, 30)
    succ_f = [s[-1] for r, s in zip(test_r, test_s) if     r.episode_success]
    fail_f = [s[-1] for r, s in zip(test_r, test_s) if not r.episode_success]
    if succ_f:
        ax_h.hist(succ_f, bins=bins, color="seagreen", alpha=0.6,
                  label=f"Success (n={len(succ_f)})", density=True)
    if fail_f:
        ax_h2.hist(fail_f, bins=bins, color="crimson", alpha=0.5,
                   label=f"Failure (n={len(fail_f)})", density=True)
    ax_h.axvline(0.5, color="orange", linestyle="--", lw=1.5)
    ax_h.set_xlabel("Final failure score")
    ax_h.set_ylabel("Density (success)", color="seagreen")
    ax_h2.set_ylabel("Density (failure)", color="crimson")
    ax_h.tick_params(axis="y", labelcolor="seagreen")
    ax_h2.tick_params(axis="y", labelcolor="crimson")
    ax_h.set_title("Final Score Distribution -- Test")
    lines1, lbl1 = ax_h.get_legend_handles_labels()
    lines2, lbl2 = ax_h2.get_legend_handles_labels()
    ax_h.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)
    ax_h.grid(True, alpha=0.3)

    # PCA (all rollouts, train=circle, test=triangle)
    all_r   = train_r + test_r
    feats   = np.stack([r.hidden_states.float().mean(0).numpy() for r in all_r])
    labels  = np.array([r.episode_success for r in all_r])
    split   = np.array(["train"] * len(train_r) + ["test"] * len(test_r))
    pca     = PCA(n_components=2)
    proj    = pca.fit_transform(feats)

    ax_p = fig.add_subplot(gs[1, 1])
    for val, color, name in [(1, "seagreen", "Success"), (0, "crimson", "Failure")]:
        m = labels == val
        ax_p.scatter(proj[m & (split == "train"), 0],
                     proj[m & (split == "train"), 1],
                     c=color, marker="o", alpha=0.5, s=18,
                     edgecolors="none", label=f"{name} (train)")
        ax_p.scatter(proj[m & (split == "test"),  0],
                     proj[m & (split == "test"),  1],
                     c=color, marker="^", alpha=0.85, s=28,
                     edgecolors="none", label=f"{name} (test)")
    ax_p.set_xlabel(f"PC 1  ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax_p.set_ylabel(f"PC 2  ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax_p.set_title("PCA of Mean Hidden States")
    ax_p.legend(fontsize=7); ax_p.grid(True, alpha=0.3)

    fig.suptitle("Failure Detector -- Summary", fontsize=14, fontweight="bold")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {save_path}")


def plot_early_detection(rollouts, scores, threshold: float, save_path: str):
    """
    For every true-failure episode plot where in the episode (normalised 0→1)
    the running-mean score first crosses `threshold`.

    Left panel:  histogram of detection times (failures only)
    Right panel: cumulative detection rate vs. normalised episode time
    """
    detection_times = []
    undetected_fail = 0
    fp_times = []

    for r, s in zip(rollouts, scores):
        arr = np.array(s)
        crossings = np.where(arr >= threshold)[0]
        if r.episode_success:
            if len(crossings) > 0:
                fp_times.append(crossings[0] / max(len(s) - 1, 1))
        else:
            if len(crossings) > 0:
                detection_times.append(crossings[0] / max(len(s) - 1, 1))
            else:
                undetected_fail += 1

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    bins = np.linspace(0, 1, 25)

    ax = axes[0]
    if detection_times:
        ax.hist(detection_times, bins=bins, color="crimson", alpha=0.7,
                label=f"Failure detected (n={len(detection_times)})")
    if fp_times:
        ax.hist(fp_times, bins=bins, color="seagreen", alpha=0.5,
                label=f"False alarm / success (n={len(fp_times)})")
    ax.set_xlabel("Normalised detection time  (0=start, 1=end)")
    ax.set_ylabel("Count")
    ax.set_title(f"When Does Detector Trigger?  (thresh={threshold})")
    if undetected_fail:
        ax.text(0.98, 0.97, f"{undetected_fail} failures never detected",
                transform=ax.transAxes, ha="right", va="top",
                color="gray", fontsize=8)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    if detection_times:
        t_sorted  = np.sort(detection_times)
        cum_frac  = np.arange(1, len(t_sorted) + 1) / \
                    (len(detection_times) + undetected_fail)
        ax2.step(np.concatenate([[0], t_sorted, [1]]),
                 np.concatenate([[0], cum_frac, [cum_frac[-1]]]),
                 color="crimson", lw=2, label="Failure detection rate")
    if fp_times:
        t_fp_s = np.sort(fp_times)
        n_succ = max(sum(r.episode_success for r in rollouts), 1)
        cum_fp = np.arange(1, len(t_fp_s) + 1) / n_succ
        ax2.step(np.concatenate([[0], t_fp_s, [1]]),
                 np.concatenate([[0], cum_fp, [cum_fp[-1]]]),
                 color="seagreen", lw=2, linestyle="--",
                 label="False-alarm rate (successes)")
    ax2.set_xlabel("Normalised time"); ax2.set_ylabel("Cumulative fraction")
    ax2.set_title("Cumulative Detection Rate Over Episode Time")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)
    print(f"  -> {save_path}")


# ───────────────────────────── Annotated video ─────────────────────────────────

def _read_video(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame[:, :, ::-1])   # BGR -> RGB
    cap.release()
    return frames, fps


def _task_context(rollout, all_rollouts, all_scores, n=80):
    """Return mean +/- std score curves for all rollouts of the same task."""
    tid  = rollout.task_id
    succ, fail = [], []
    for r, s in zip(all_rollouts, all_scores):
        if r.task_id != tid:
            continue
        (succ if r.episode_success else fail).append(_interp_to_n(s, n))
    xs    = np.linspace(0, 1, n)
    m_s   = np.stack(succ).mean(0) if succ else None
    std_s = np.stack(succ).std(0)  if succ else None
    m_f   = np.stack(fail).mean(0) if fail else None
    std_f = np.stack(fail).std(0)  if fail else None
    return xs, m_s, std_s, m_f, std_f


def make_annotated_video(rollout, scores, save_path,
                         all_rollouts=None, all_scores=None):
    """
    Side-by-side annotated MP4:
      Left  -- RGB observation with coloured border
               (green border = safe, red border = failure score > 0.5)
      Right -- live failure score plot with task-level context ribbons
    """
    if rollout.mp4_path is None:
        return
    frames, fps = _read_video(rollout.mp4_path)
    if not frames:
        return

    T_frames = len(frames)
    T_scores = len(scores)

    xs_ctx = m_s = std_s = m_f = std_f = None
    if all_rollouts and all_scores:
        xs_ctx, m_s, std_s, m_f, std_f = \
            _task_context(rollout, all_rollouts, all_scores)

    gt_label  = "SUCCESS" if rollout.episode_success else "FAILURE"
    out_frames = []

    for i, frame in enumerate(frames):
        s_idx     = min(int(i * T_scores / T_frames), T_scores - 1)
        cur_score = scores[s_idx]
        alert     = cur_score > 0.5

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=90)

        # Left: RGB with border ─────────────────────────────────────────────
        border = (220, 30, 30) if alert else (30, 180, 30)
        bordered = cv2.copyMakeBorder(frame, 8, 8, 8, 8,
                                      cv2.BORDER_CONSTANT, value=border)
        axes[0].imshow(bordered)
        axes[0].axis("off")
        status = "FAILURE DETECTED" if alert else "OK"
        axes[0].set_title(f"Frame {i}    {status}",
                          color="red" if alert else "green",
                          fontweight="bold", fontsize=10)

        # Right: score plot ──────────────────────────────────────────────────
        ax = axes[1]
        x_norm = np.linspace(0, 1, 80)
        if m_s is not None:
            ax.plot(x_norm, m_s, color="seagreen", lw=1, alpha=0.6,
                    label="Task success avg")
            ax.fill_between(x_norm, m_s - std_s, m_s + std_s,
                            color="seagreen", alpha=0.12)
        if m_f is not None:
            ax.plot(x_norm, m_f, color="crimson", lw=1, alpha=0.6,
                    label="Task failure avg")
            ax.fill_between(x_norm, m_f - std_f, m_f + std_f,
                            color="crimson", alpha=0.12)

        x_now = np.linspace(0, s_idx / max(T_scores - 1, 1), s_idx + 1)
        ax.plot(x_now, scores[:s_idx + 1],
                color="royalblue", lw=2.5, label="This rollout")
        ax.axhline(0.5, color="orange", linestyle="--", lw=1,
                   label="Threshold 0.5")

        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("Normalised time"); ax.set_ylabel("Failure score")
        ax.set_title(f"Failure score  (final = {scores[-1]:.3f})")
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"{rollout.task_description}\n"
            f"Ep {rollout.episode_idx}  |  GT: {gt_label}",
            fontsize=9,
        )
        fig.tight_layout()
        fig.canvas.draw()
        out_frames.append(np.array(fig.canvas.renderer.buffer_rgba()))
        plt.close(fig)

    imageio.mimsave(save_path, out_frames, fps=fps)


# ──────────────────────────────────── CLI ──────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="OpenVLA LIBERO failure detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path", required=True,
                        help="Directory with task*--ep*--succ*.csv rollout files")
    parser.add_argument("--output_dir",  default="./failure_detection_results")
    parser.add_argument("--train_ratio", type=float, default=0.7,
                        help="Fraction of rollouts used for training")
    parser.add_argument("--n_epochs",    type=int,   default=300)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--lambda_reg",  type=float, default=1e-2)
    parser.add_argument("--hidden_dim",  type=int,   default=256)
    parser.add_argument("--n_layers",    type=int,   default=2)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--save_videos", action="store_true",
                        help="Generate annotated MP4 for test rollouts")
    parser.add_argument("--n_videos",    type=int,   default=10,
                        help="Max annotated videos (half success, half failure)")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_model",  action="store_true",
                        help="Save the trained model to output_dir/detector.pth")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    if args.save_videos:
        os.makedirs(os.path.join(out, "videos"), exist_ok=True)

    # ── 1. Load ────────────────────────────────────────────────────────────
    print("\n[1/5] Loading rollouts ...")
    all_rollouts = load_rollouts(args.data_path)

    # Stratified split: preserve success-rate in both train and test
    rng       = np.random.RandomState(args.seed)
    succ_idx  = [i for i, r in enumerate(all_rollouts) if     r.episode_success]
    fail_idx  = [i for i, r in enumerate(all_rollouts) if not r.episode_success]
    rng.shuffle(succ_idx); rng.shuffle(fail_idx)

    n_s_train = max(1, int(len(succ_idx) * args.train_ratio))
    n_f_train = max(1, int(len(fail_idx) * args.train_ratio))
    train_idx = succ_idx[:n_s_train] + fail_idx[:n_f_train]
    test_idx  = succ_idx[n_s_train:] + fail_idx[n_f_train:]

    train_rollouts = [all_rollouts[i] for i in train_idx]
    test_rollouts  = [all_rollouts[i] for i in test_idx]

    print(f"  Train: {len(train_rollouts)}  "
          f"({sum(r.episode_success for r in train_rollouts)} succ / "
          f"{sum(1 - r.episode_success for r in train_rollouts)} fail)")
    print(f"  Test:  {len(test_rollouts)}  "
          f"({sum(r.episode_success for r in test_rollouts)} succ / "
          f"{sum(1 - r.episode_success for r in test_rollouts)} fail)")

    # ── 2. Train ───────────────────────────────────────────────────────────
    print("\n[2/5] Training failure detector ...")
    input_dim = all_rollouts[0].hidden_states.shape[-1]
    print(f"  Feature dim: {input_dim}")
    model = FailureDetector(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(args.device)

    losses = train_model(
        model, train_rollouts,
        n_epochs=args.n_epochs, lr=args.lr,
        lambda_reg=args.lambda_reg, batch_size=args.batch_size,
        device=args.device,
    )

    # ── 3. Predict ─────────────────────────────────────────────────────────
    print("\n[3/5] Running predictions ...")
    train_scores = predict(model, train_rollouts, device=args.device)
    test_scores  = predict(model, test_rollouts,  device=args.device)
    all_scores   = predict(model, all_rollouts,   device=args.device)

    # ── 4. Static plots ────────────────────────────────────────────────────
    print("\n[4/5] Generating static visualisations ...")
    plot_training_loss(losses,
        os.path.join(out, "training_loss.png"))
    plot_score_curves(train_rollouts, train_scores,
        "Failure Score Curves -- Train",
        os.path.join(out, "score_curves_train.png"))
    plot_score_curves(test_rollouts, test_scores,
        "Failure Score Curves -- Test",
        os.path.join(out, "score_curves_test.png"))
    plot_roc_curve(test_rollouts, test_scores,
        os.path.join(out, "roc_curve.png"))
    plot_score_histogram(test_rollouts, test_scores,
        os.path.join(out, "score_histogram.png"))
    plot_pca_scatter(all_rollouts,
        os.path.join(out, "pca_scatter.png"))
    plot_per_task_auc(test_rollouts, test_scores,
        os.path.join(out, "per_task_auc.png"))
    plot_summary(train_rollouts, test_rollouts, train_scores, test_scores,
        os.path.join(out, "summary.png"))

    # ── 5. Videos ──────────────────────────────────────────────────────────
    if args.save_videos:
        print(f"\n[5/5] Generating annotated videos ...")
        s_idx = [i for i, r in enumerate(test_rollouts) if     r.episode_success]
        f_idx = [i for i, r in enumerate(test_rollouts) if not r.episode_success]
        n_each = max(1, args.n_videos // 2)
        to_render = s_idx[:n_each] + f_idx[:n_each]

        for idx in tqdm(to_render, desc="Videos"):
            r     = test_rollouts[idx]
            tag   = "success" if r.episode_success else "failure"
            fname = (f"task{r.task_id}_ep{r.episode_idx}"
                     f"_gt{tag}_score{test_scores[idx][-1]:.2f}.mp4")
            make_annotated_video(
                r, test_scores[idx],
                os.path.join(out, "videos", fname),
                all_rollouts, all_scores,
            )
            print(f"    {fname}")
    else:
        print("\n[5/5] Videos skipped  (pass --save_videos to enable)")

    # ── Summary stats ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    if test_rollouts:
        y_true  = np.array([1 - r.episode_success for r in test_rollouts])
        y_score = np.array([s[-1] for s in test_scores])
        if len(np.unique(y_true)) > 1:
            print(f"  Test AUC (failure detection):  "
                  f"{roc_auc_score(y_true, y_score):.4f}")
        acc = np.mean((y_score > 0.5) == y_true)
        print(f"  Accuracy @ threshold 0.5:      {acc:.4f}")

        # ── Early detection time ─────────────────────────────────────────
        # For each true-failure episode: at what fraction of its length does
        # the running-mean score first exceed 0.5?  Lower = earlier warning.
        thresh = 0.5
        early_times = []
        for r, s in zip(test_rollouts, test_scores):
            if r.episode_success:
                continue           # only measure on actual failures
            crossings = np.where(np.array(s) >= thresh)[0]
            if len(crossings) > 0:
                t_norm = crossings[0] / max(len(s) - 1, 1)
                early_times.append(t_norm)

        if early_times:
            print(f"\n  Early-detection on failures:")
            print(f"    Detected {len(early_times)} / "
                  f"{sum(1 for r in test_rollouts if not r.episode_success)} "
                  f"failure episodes")
            print(f"    Mean detection time:  "
                  f"{np.mean(early_times):.3f}  (0=start, 1=end of episode)")
            print(f"    Median detection time:{np.median(early_times):.3f}")
            # Also generate the plot
            plot_early_detection(test_rollouts, test_scores,
                                 thresh, os.path.join(out, "early_detection.png"))

    print(f"\n  All outputs saved to:  {os.path.abspath(out)}/")
    print("=" * 60)

    # ── Save model checkpoint ───────────────────────────────────────────────
    if args.save_model:
        ckpt_path = os.path.join(out, "detector.pth")
        torch.save({
            "model_state_dict": model.state_dict(),
            "input_dim":  all_rollouts[0].hidden_states.shape[-1],
            "hidden_dim": args.hidden_dim,
            "n_layers":   args.n_layers,
        }, ckpt_path)
        print(f"  Model checkpoint saved to: {ckpt_path}")


if __name__ == "__main__":
    main()
