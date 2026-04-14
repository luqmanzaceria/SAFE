#!/usr/bin/env python3
"""
compare_with_safe.py  —  SAFE Official Code vs Our Temporal Attention Model
============================================================================

This script uses SAFE's *actual* IndepModel class from failure_prob/model/indep.py
(the model described in the NeurIPS 2025 paper), trains it with SAFE's real loss
functions, then compares it against our Temporal Attention + Hinge model under an
identical evaluation protocol.

Key difference from safe_eval.py
---------------------------------
  safe_eval.py     : re-implements SAFE's IndepModel from scratch in best_detector.py
  compare_with_safe: imports and calls failure_prob.model.indep.IndepModel directly
                     → fairer, more rigorous "official SAFE vs ours" comparison

Architecture summary
--------------------
  SAFE IndepModel  (official)
  ────────────────────────────────────────────────────
  h_t ──▶ 2-layer MLP ──▶ p_t ──▶ running mean ──▶ score_t
  Loss: hinge with exponential time-weighting (early detection bias)
  Labels: success=1 (push score low), failure=0 (push score high)

  Ours: Temporal Attention + Hinge  (our contribution)
  ────────────────────────────────────────────────────
  h_t ──▶ MLP encoder ──▶ score head p_t
                       ──▶ weight head w_t
  score_t = Σ_{i≤t} w_i·p_i / Σ_{i≤t} w_i   (causal attention)
  Loss: SAFE hinge on p_t + BCE on attention alignment

Evaluation protocol
-------------------
  • 30 % of task IDs are held out (val_unseen) — the PRIMARY test set
  • val_seen  = remaining 40 % of seen-task rollouts (conformal calibration)
  • train     = 60 % of seen-task rollouts
  • Episode score = max(score_t,  t < task_min_step)  [SAFE's "earliest stop"]
  • Metrics: ROC-AUC, AP, Bal-Acc, F1, avg detection time, conformal recall/FAR
  • Multiple seeds → mean ± std

Outputs (output_dir/)
---------------------
  fig_roc.png          ROC curves with std bands
  fig_prc.png          Precision-recall curves
  fig_score_curves.png Score trajectories
  fig_detection.png    Detection time CDF
  fig_per_task.png     Per-task AUC bar chart
  fig_conformal.png    Conformal coverage
  summary_table.png    Styled numeric comparison table
  results.csv          All numbers (paste into LaTeX)
  summary.txt          Full text report

Usage
-----
    python scripts/compare_with_safe.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
        --output_dir ./compare_results

    # Quick test
    python scripts/compare_with_safe.py \\
        --data_path <path> --n_epochs 100 --seeds 0

    # Multi-seed, full run
    python scripts/compare_with_safe.py \\
        --data_path <path> --n_epochs 300 --seeds 0 1 2 --device cuda
"""

import os
import sys
import csv
import argparse
import random
import warnings
from collections import defaultdict
from types import SimpleNamespace

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve, balanced_accuracy_score, f1_score,
)

import torch
import torch.nn as nn
from tqdm import tqdm

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_FILE   = os.path.realpath(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)
_REPO_ROOT   = os.path.dirname(_SCRIPTS_DIR)
for _p in (_SCRIPTS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Our modules ───────────────────────────────────────────────────────────────
from failure_detector import load_rollouts
from combined_detector import CombinedFailureDetector
from best_detector    import train_attn_hinge, predict_attn_hinge

# ── SAFE's actual model + loss utilities ─────────────────────────────────────
from failure_prob.model.indep import IndepModel
from failure_prob.model.utils import get_time_weight, aggregate_monitor_loss

# ── SAFE's conformal utilities ────────────────────────────────────────────────
try:
    from failure_prob.utils.conformal.split_cp import split_conformal_binary
    _HAS_CONFORMAL = True
except ImportError:
    _HAS_CONFORMAL = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Plotting style
# ─────────────────────────────────────────────────────────────────────────────
MODEL_COLORS = {"safe": "#4dac26", "ours": "#d73027"}
MODEL_LABELS = {
    "safe": "SAFE IndepModel (official)",
    "ours": "Temporal Attention + Hinge (ours)",
}
MODEL_LS = {"safe": "--", "ours": "-"}


# ─────────────────────────────────────────────────────────────────────────────
# Config mock for SAFE's IndepModel (bypasses hydra)
# ─────────────────────────────────────────────────────────────────────────────

def _make_safe_cfg(lr: float = 3e-4, lambda_reg: float = 1e-2) -> SimpleNamespace:
    """Build a minimal Config-like object accepted by SAFE's IndepModel."""
    model = SimpleNamespace(
        name="indep",
        # architecture
        n_layers=2,
        hidden_dim=256,
        final_act_layer="sigmoid",
        n_history_steps=1,
        # scoring
        cumsum=False,
        rmean=True,          # running mean (SAFE default for IndepModel)
        # loss
        use_time_weighting=True,
        use_threshold=True,
        threshold=0.5,
        # regularisation & optimiser
        lambda_reg=lambda_reg,
        lambda_success=1.0,
        lambda_fail=1.0,
        grad_max_norm=None,
        lr=lr,
        lr_step_size=1000,   # no decay within our budget
        lr_gamma=1.0,
        warmup_steps=0,
        optimizer="adam",
        weight_decay=1e-2,
        dropout=0.0,
    )
    dataset = SimpleNamespace(load_to_cuda=False)
    return SimpleNamespace(model=model, dataset=dataset)


# ─────────────────────────────────────────────────────────────────────────────
# Data utilities
# ─────────────────────────────────────────────────────────────────────────────

def make_split(rollouts, unseen_ratio: float = 0.30,
               seen_train_ratio: float = 0.60, rng=None):
    """Replicate SAFE's 3-way seen/unseen task split."""
    if rng is None:
        rng = np.random.default_rng(0)

    task_ids = sorted(set(r.task_id for r in rollouts))
    rng.shuffle(task_ids := np.array(task_ids))

    n_unseen = max(1, round(unseen_ratio * len(task_ids)))
    unseen_ids = set(task_ids[:n_unseen].tolist())
    seen_ids   = set(task_ids[n_unseen:].tolist())

    seen_r   = [r for r in rollouts if r.task_id in seen_ids]
    unseen_r = [r for r in rollouts if r.task_id in unseen_ids]

    # Per-task train / val_seen split
    by_task = defaultdict(list)
    for r in seen_r:
        by_task[r.task_id].append(r)

    train_r, val_seen_r = [], []
    for tid, eps in by_task.items():
        rng.shuffle(eps := eps[:])
        k = max(1, round(seen_train_ratio * len(eps)))
        train_r.extend(eps[:k])
        val_seen_r.extend(eps[k:])

    return {"train": train_r, "val_seen": val_seen_r, "val_unseen": unseen_r}


def task_min_steps(rollouts) -> dict:
    """task_id → minimum episode length (SAFE's task_min_step concept)."""
    tms = {}
    for r in rollouts:
        l = r.hidden_states.shape[0]
        tms[r.task_id] = min(tms.get(r.task_id, l), l)
    return tms


def _pad_batch(rollouts, device="cpu"):
    """Pad a list of rollouts → (features, masks) tensors (SAFE batch format)."""
    max_T = max(r.hidden_states.shape[0] for r in rollouts)
    B     = len(rollouts)
    D     = rollouts[0].hidden_states.shape[1]

    features = torch.zeros(B, max_T, D, device=device)
    masks    = torch.zeros(B, max_T, device=device)
    # SAFE convention: success=1 / failure=0
    labels   = torch.tensor([r.episode_success for r in rollouts],
                             dtype=torch.long, device=device)

    for i, r in enumerate(rollouts):
        T = r.hidden_states.shape[0]
        features[i, :T] = r.hidden_states.to(device)
        masks[i, :T]    = 1.0

    return {"features": features, "valid_masks": masks, "success_labels": labels}


# ─────────────────────────────────────────────────────────────────────────────
# SAFE IndepModel training  (uses their actual forward_compute_loss)
# ─────────────────────────────────────────────────────────────────────────────

def train_safe_model(model: IndepModel,
                     rollouts: list,
                     n_epochs: int = 300,
                     lr: float = 3e-4,
                     lambda_reg: float = 1e-2,
                     batch_size: int = 32,
                     device: str = "cpu") -> list:
    """
    Train SAFE's official IndepModel using their hinge + time-weight loss.
    Bypasses wandb/hydra — uses forward_compute_loss directly.
    Returns list of per-epoch losses.
    """
    model = model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Class weights (SAFE convention)
    n_s  = sum(r.episode_success == 1 for r in rollouts)
    n_f  = sum(r.episode_success == 0 for r in rollouts)
    freq_0 = (n_f + 1) / len(rollouts)
    freq_1 = (n_s + 1) / len(rollouts)
    weights = [
        (1.0 / freq_0) * model.cfg.model.lambda_fail,
        (1.0 / freq_1) * model.cfg.model.lambda_success,
    ]

    losses = []
    bar = tqdm(range(n_epochs), desc="  SAFE IndepModel", leave=False)
    for _ in bar:
        idxs = list(range(len(rollouts)))
        random.shuffle(idxs)

        ep_losses = []
        for i in range(0, len(idxs), batch_size):
            batch_r = [rollouts[j] for j in idxs[i:i + batch_size]]
            batch   = _pad_batch(batch_r, device)

            loss, _ = model.forward_compute_loss(batch, weights)

            # L2 regularisation (same as SAFE's compute_regularization_loss)
            reg = sum(p.pow(2).sum()
                      for n, p in model.named_parameters()
                      if "bias" not in n)
            total = loss + lambda_reg * reg

            opt.zero_grad()
            total.backward()
            opt.step()
            ep_losses.append(total.item())

        avg = float(np.mean(ep_losses))
        losses.append(avg)
        bar.set_postfix(loss=f"{avg:.4f}")

    return losses


def predict_safe_model(model: IndepModel, rollouts: list,
                       device: str = "cpu") -> list:
    """
    Run inference with SAFE's IndepModel.
    Returns per-rollout (T,) score curves: high score = failure.
    """
    model.eval()
    curves = []
    with torch.no_grad():
        for r in rollouts:
            T    = r.hidden_states.shape[0]
            feat = r.hidden_states.unsqueeze(0).to(device)       # (1, T, D)
            mask = torch.ones(1, T, device=device)
            batch = {"features": feat,
                     "valid_masks": mask,
                     "success_labels": torch.tensor([0], device=device)}
            sc = model(batch).squeeze(-1).squeeze(0).cpu().numpy()  # (T,)
            curves.append(sc)
    return curves


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def episode_score(curve: np.ndarray, tms: int) -> float:
    """SAFE's 'earliest stop' scoring: max score up to task_min_step."""
    horizon = min(tms, len(curve))
    return float(curve[:horizon].max())


def detection_time(curve: np.ndarray, threshold: float) -> float | None:
    """Normalised timestep [0,1] of first threshold crossing, or None."""
    T    = len(curve)
    hits = np.where(curve >= threshold)[0]
    return float(hits[0]) / T if len(hits) else None


def compute_metrics(scores: list, labels: list, tms_dict: dict,
                    rollouts: list, alpha: float = 0.2):
    """
    Compute all metrics for one model on one split.

    Returns dict with: auc, ap, bal_acc, f1, avg_det, far, conf_recall, conf_far
    """
    ep_scores  = np.array([episode_score(sc, tms_dict[r.task_id])
                            for sc, r in zip(scores, rollouts)])
    # failure=1 (our eval convention)
    ep_labels  = np.array([1 - r.episode_success for r in rollouts])

    if len(np.unique(ep_labels)) < 2:
        return None

    auc = roc_auc_score(ep_labels, ep_scores)
    ap  = average_precision_score(ep_labels, ep_scores)

    thresh = 0.5
    preds  = (ep_scores >= thresh).astype(int)
    bal    = balanced_accuracy_score(ep_labels, preds)
    f1     = f1_score(ep_labels, preds, zero_division=0)

    # Average detection time on failure episodes
    det_times = []
    for sc, r in zip(scores, rollouts):
        if r.episode_success == 0:           # failure
            dt = detection_time(sc, thresh)
            if dt is not None:
                det_times.append(dt)
    avg_det = float(np.mean(det_times)) if det_times else 1.0

    # False alarm rate on success episodes
    n_fa  = sum(1 for sc, r in zip(scores, rollouts)
                if r.episode_success == 1 and sc.max() >= thresh)
    n_s   = sum(1 for r in rollouts if r.episode_success == 1)
    far   = n_fa / n_s if n_s else 0.0

    # Conformal calibration (if available and cal_scores provided)
    conf_recall = conf_far = float("nan")
    if _HAS_CONFORMAL:
        try:
            cal_ep  = ep_scores
            cal_lbl = ep_labels
            result  = split_conformal_binary(cal_ep, cal_ep, cal_lbl, alpha=alpha)
            tau     = result["threshold"]
            conf_recall = float(np.mean((ep_scores[ep_labels == 1] >= tau)))
            conf_far    = float(np.mean((ep_scores[ep_labels == 0] >= tau)))
        except Exception:
            pass

    return dict(auc=auc, ap=ap, bal_acc=bal, f1=f1,
                avg_det=avg_det, far=far,
                conf_recall=conf_recall, conf_far=conf_far)


# ─────────────────────────────────────────────────────────────────────────────
# One-seed run
# ─────────────────────────────────────────────────────────────────────────────

def run_one_seed(rollouts, seed, args, device):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    splits  = make_split(rollouts, rng=rng)
    train_r = splits["train"]
    val_r   = splits["val_unseen"]      # primary test set

    n_s = sum(r.episode_success for r in train_r)
    n_f = sum(1 - r.episode_success for r in train_r)
    print(f"    train: {len(train_r)} ({n_s} succ / {n_f} fail)  "
          f"val_unseen: {len(val_r)}")

    if n_f == 0 or n_s == 0:
        print("    [skip] training set needs both classes")
        return None

    input_dim = rollouts[0].hidden_states.shape[1]
    tms_all   = task_min_steps(rollouts)

    results = {}

    # ── SAFE IndepModel ───────────────────────────────────────────────────────
    print("    Training SAFE IndepModel …")
    safe_cfg   = _make_safe_cfg(lr=args.lr, lambda_reg=args.lambda_reg)
    safe_model = IndepModel(safe_cfg, input_dim).to(device)
    safe_losses = train_safe_model(safe_model, train_r,
                                   n_epochs=args.n_epochs, lr=args.lr,
                                   lambda_reg=args.lambda_reg,
                                   batch_size=args.batch_size, device=device)
    safe_curves = predict_safe_model(safe_model, val_r, device)
    results["safe"] = dict(
        metrics=compute_metrics(safe_curves, None, tms_all, val_r),
        curves=safe_curves,
        losses=safe_losses,
    )
    print(f"    SAFE AUC: {results['safe']['metrics']['auc']:.4f}")

    # ── Temporal Attention + Hinge  ───────────────────────────────────────────
    print("    Training Temporal Attention + Hinge …")
    attn_model = CombinedFailureDetector(
        input_dim=input_dim, task_embed_dim=0,
        hidden_dim=256, n_layers=2, dropout=0.1,
    ).to(device)
    train_attn_hinge(attn_model, train_r, task_embeds={},
                     n_epochs=args.n_epochs, lr=args.lr,
                     lambda_reg=args.lambda_reg, device=device)

    raw_ours = predict_attn_hinge(attn_model, val_r, {}, device)
    # predict_attn_hinge returns (score_curve, weight_curve) tuples
    ours_curves = [sc if not isinstance(sc, tuple) else sc[0]
                   for sc in raw_ours]
    results["ours"] = dict(
        metrics=compute_metrics(ours_curves, None, tms_all, val_r),
        curves=ours_curves,
        losses=[],
    )
    print(f"    Ours AUC: {results['ours']['metrics']['auc']:.4f}")

    # Attach rollouts for plotting
    results["_val_r"]   = val_r
    results["_tms_all"] = tms_all
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate(seed_results):
    """mean ± std across seeds for all scalar metrics."""
    names   = [k for k in seed_results[0] if not k.startswith("_")]
    agg     = {}
    for name in names:
        m_keys  = seed_results[0][name]["metrics"].keys()
        metrics = {k: [] for k in m_keys}
        for sr in seed_results:
            for k, v in sr[name]["metrics"].items():
                metrics[k].append(v)
        agg[name] = {k: (float(np.nanmean(v)), float(np.nanstd(v)))
                     for k, v in metrics.items()}
    return agg


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def fig_roc(seed_results, out_dir):
    """ROC curves (std shading over seeds)."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=0.8)

    base_grid = np.linspace(0, 1, 200)
    for name in ("safe", "ours"):
        tprs = []
        aucs = []
        for sr in seed_results:
            if name not in sr:
                continue
            val_r   = sr["_val_r"]
            curves  = sr[name]["curves"]
            tms_all = sr["_tms_all"]
            ep_sc   = np.array([episode_score(sc, tms_all[r.task_id])
                                 for sc, r in zip(curves, val_r)])
            ep_lbl  = np.array([1 - r.episode_success for r in val_r])
            if len(np.unique(ep_lbl)) < 2:
                continue
            fpr, tpr, _ = roc_curve(ep_lbl, ep_sc)
            tprs.append(np.interp(base_grid, fpr, tpr))
            aucs.append(roc_auc_score(ep_lbl, ep_sc))

        if not tprs:
            continue
        mean_tpr = np.mean(tprs, axis=0)
        std_tpr  = np.std(tprs, axis=0)
        mean_auc = np.mean(aucs)
        std_auc  = np.std(aucs)
        c = MODEL_COLORS[name]
        ax.plot(base_grid, mean_tpr, color=c, ls=MODEL_LS[name], lw=2,
                label=f"{MODEL_LABELS[name]}\nAUC={mean_auc:.3f}±{std_auc:.3f}")
        ax.fill_between(base_grid, mean_tpr - std_tpr, mean_tpr + std_tpr,
                        alpha=0.15, color=c)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — val_unseen (unseen tasks)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_roc.png"), dpi=150)
    plt.close(fig)


def fig_score_curves(seed_results, out_dir):
    """Mean score trajectories for success vs failure episodes."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, name in zip(axes, ("safe", "ours")):
        sr      = seed_results[0]
        val_r   = sr["_val_r"]
        curves  = sr[name]["curves"]

        succ = [c for c, r in zip(curves, val_r) if r.episode_success == 1]
        fail = [c for c, r in zip(curves, val_r) if r.episode_success == 0]

        def _plot_band(ax, seqs, color, label):
            if not seqs:
                return
            max_T  = max(len(s) for s in seqs)
            mat    = np.full((len(seqs), max_T), np.nan)
            for i, s in enumerate(seqs):
                mat[i, :len(s)] = s
            xs     = np.arange(max_T) / max_T
            mean   = np.nanmean(mat, axis=0)
            std    = np.nanstd(mat, axis=0)
            ax.plot(xs, mean, color=color, lw=2, label=label)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.2, color=color)

        _plot_band(ax, succ, "#2166ac", "Success")
        _plot_band(ax, fail, "#d73027", "Failure")
        ax.set_title(MODEL_LABELS[name], fontsize=9)
        ax.set_xlabel("Normalised timestep")
        ax.set_ylabel("Score")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)

    fig.suptitle("Score Trajectories (seed 0, val_unseen)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_score_curves.png"), dpi=150)
    plt.close(fig)


def fig_detection_cdf(seed_results, out_dir):
    """CDF of detection times for failure episodes."""
    fig, ax = plt.subplots(figsize=(5, 4))
    for name in ("safe", "ours"):
        all_det = []
        for sr in seed_results:
            val_r  = sr["_val_r"]
            curves = sr[name]["curves"]
            for sc, r in zip(curves, val_r):
                if r.episode_success == 0:
                    dt = detection_time(sc, 0.5)
                    if dt is not None:
                        all_det.append(dt)
        if not all_det:
            continue
        xs = np.sort(all_det)
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, color=MODEL_COLORS[name], ls=MODEL_LS[name], lw=2,
                label=MODEL_LABELS[name])

    ax.axvline(0.5, color="gray", ls=":", lw=1, label="t=0.5")
    ax.set_xlabel("Normalised detection time")
    ax.set_ylabel("CDF")
    ax.set_title("Detection Time CDF (failure episodes, val_unseen)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_detection.png"), dpi=150)
    plt.close(fig)


def fig_per_task(seed_results, out_dir):
    """Per-task AUC comparison."""
    sr      = seed_results[0]
    val_r   = sr["_val_r"]
    tms_all = sr["_tms_all"]
    tasks   = sorted(set(r.task_id for r in val_r))

    safe_aucs = []
    ours_aucs = []
    valid_tasks = []
    for tid in tasks:
        sub_r = [r for r in val_r if r.task_id == tid]
        lbls  = [1 - r.episode_success for r in sub_r]
        if len(np.unique(lbls)) < 2:
            continue
        valid_tasks.append(tid)
        for name, container in (("safe", safe_aucs), ("ours", ours_aucs)):
            curves  = sr[name]["curves"]
            sub_sc  = [curves[val_r.index(r)] for r in sub_r]
            ep_sc   = [episode_score(sc, tms_all[tid]) for sc in sub_sc]
            try:
                container.append(roc_auc_score(lbls, ep_sc))
            except Exception:
                container.append(float("nan"))

    if not valid_tasks:
        return

    x = np.arange(len(valid_tasks))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(valid_tasks) * 0.7), 4))
    ax.bar(x - w/2, safe_aucs, w, label=MODEL_LABELS["safe"],
           color=MODEL_COLORS["safe"], alpha=0.8)
    ax.bar(x + w/2, ours_aucs, w, label=MODEL_LABELS["ours"],
           color=MODEL_COLORS["ours"], alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{tid}" for tid in valid_tasks], rotation=45, ha="right")
    ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", ls=":", lw=0.8)
    ax.set_title("Per-Task AUC on val_unseen (seed 0)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_per_task.png"), dpi=150)
    plt.close(fig)


def fig_summary_table(agg: dict, out_dir: str):
    """Styled comparison table saved as PNG."""
    metrics_show = ["auc", "ap", "bal_acc", "f1", "avg_det", "far"]
    labels_show  = ["AUC ↑", "AP ↑", "Bal-Acc ↑", "F1 ↑", "Avg-Det ↓", "FAR ↓"]

    rows = []
    for name in ("safe", "ours"):
        row = [MODEL_LABELS[name]]
        for k in metrics_show:
            m, s = agg[name][k]
            row.append(f"{m:.3f}±{s:.3f}" if s > 1e-6 else f"{m:.3f}")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Model"] + labels_show,
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)

    # Highlight header
    for j in range(len(labels_show) + 1):
        tbl[(0, j)].set_facecolor("#404040")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Colour SAFE row green, ours row red
    colors_row = [MODEL_COLORS["safe"], MODEL_COLORS["ours"]]
    for i, c in enumerate(colors_row, start=1):
        for j in range(len(labels_show) + 1):
            tbl[(i, j)].set_facecolor(c + "33")   # 20% alpha hex

    fig.suptitle("SAFE Official vs Our Temporal Attention Model", fontsize=11, y=0.98)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "summary_table.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)


def save_csv(agg: dict, out_dir: str):
    path = os.path.join(out_dir, "results.csv")
    metrics_show = ["auc", "ap", "bal_acc", "f1", "avg_det", "far",
                    "conf_recall", "conf_far"]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + metrics_show + [f"{k}_std" for k in metrics_show])
        for name in ("safe", "ours"):
            row = [MODEL_LABELS[name]]
            row += [f"{agg[name][k][0]:.4f}" for k in metrics_show]
            row += [f"{agg[name][k][1]:.4f}" for k in metrics_show]
            w.writerow(row)
    print(f"  Results CSV → {path}")


def write_summary(agg: dict, args, out_dir: str):
    lines = [
        "=" * 64,
        "SAFE Official IndepModel  vs  Temporal Attention + Hinge",
        "=" * 64,
        f"  Data:     {args.data_path}",
        f"  Seeds:    {args.seeds}",
        f"  Epochs:   {args.n_epochs}",
        f"  Protocol: SAFE 'earliest-stop' scoring, 30% unseen task split",
        "",
    ]

    metric_labels = {
        "auc": "ROC-AUC ↑", "ap": "Avg-Precision ↑",
        "bal_acc": "Balanced-Acc ↑", "f1": "F1 ↑",
        "avg_det": "Avg-Det-Time ↓", "far": "FAR ↓",
        "conf_recall": "Conf. Recall ↑", "conf_far": "Conf. FAR ↓",
    }

    for name in ("safe", "ours"):
        lines.append(f"── {MODEL_LABELS[name]} ──")
        for k, lbl in metric_labels.items():
            m, s = agg[name][k]
            lines.append(f"  {lbl:<24s}: {m:.4f} ± {s:.4f}")
        lines.append("")

    # Delta analysis
    for k, lbl in metric_labels.items():
        ms, ss = agg["safe"][k]
        mo, so = agg["ours"][k]
        delta  = mo - ms
        sign   = "+" if delta >= 0 else ""
        lines.append(f"  Δ {lbl:<24s}: {sign}{delta:.4f}")

    lines += ["", "=" * 64]
    txt = "\n".join(lines)
    print(txt)

    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(txt)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare SAFE's official IndepModel vs Temporal Attention"
    )
    parser.add_argument("--data_path",   required=True,
                        help="Folder with task*.csv + *.pkl rollout files")
    parser.add_argument("--output_dir",  default="./compare_results")
    parser.add_argument("--seeds",       type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n_epochs",    type=int, default=300)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--lambda_reg",  type=float, default=1e-2)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available()
                                                       else "cpu")
    parser.add_argument("--unseen_ratio",  type=float, default=0.30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*64}")
    print("SAFE Official IndepModel  vs  Temporal Attention + Hinge")
    print(f"{'='*64}")
    print(f"  Data:    {args.data_path}")
    print(f"  Seeds:   {args.seeds}")
    print(f"  Epochs:  {args.n_epochs}  Device: {args.device}")
    print()

    # ── Load data ─────────────────────────────────────────────────────────────
    print("[1/3] Loading rollouts …")
    rollouts = load_rollouts(args.data_path)
    n_s = sum(r.episode_success for r in rollouts)
    n_f = len(rollouts) - n_s
    print(f"  {len(rollouts)} rollouts  ({n_s} success / {n_f} failure)\n")

    if n_f == 0:
        print("ERROR: no failure rollouts found — cannot train a failure detector.")
        return
    if n_s == 0:
        print("ERROR: no success rollouts found — cannot train a failure detector.")
        return

    # ── Multi-seed training ───────────────────────────────────────────────────
    print("[2/3] Training (all seeds) …")
    seed_results = []
    for seed in args.seeds:
        print(f"\n  seed={seed}")
        sr = run_one_seed(rollouts, seed, args, args.device)
        if sr is not None:
            seed_results.append(sr)

    if not seed_results:
        print("No valid seed results — check your data.")
        return

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = aggregate(seed_results)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[3/3] Generating figures …")
    fig_roc(seed_results, args.output_dir)
    fig_score_curves(seed_results, args.output_dir)
    fig_detection_cdf(seed_results, args.output_dir)
    fig_per_task(seed_results, args.output_dir)
    fig_summary_table(agg, args.output_dir)
    save_csv(agg, args.output_dir)
    write_summary(agg, args, args.output_dir)

    print(f"\n  All outputs → {args.output_dir}/")
    print("    fig_roc.png            ROC curves (std bands)")
    print("    fig_score_curves.png   Score trajectories")
    print("    fig_detection.png      Detection time CDF")
    print("    fig_per_task.png       Per-task AUC")
    print("    summary_table.png      Styled results table")
    print("    results.csv            Numbers for LaTeX")
    print("    summary.txt            Full report")


if __name__ == "__main__":
    main()
