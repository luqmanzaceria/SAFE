#!/usr/bin/env python3
"""
SAFE-Protocol Evaluation  —  Fair Apples-to-Apples Comparison
==============================================================

Trains SAFE IndepModel and our Temporal Attention + Hinge model, then
evaluates BOTH using SAFE's official evaluation protocol exactly as it
appears in the SAFE paper (failure_prob/train.py + utils/metrics.py).

SAFE evaluation protocol
------------------------
  Data splits:
    train      — seen tasks, 60 % of seen-task rollouts
    val_seen   — seen tasks, remaining 40 %  (used for conformal calibration)
    val_unseen — unseen tasks (30 % of all task IDs)  ← PRIMARY test set

  Scoring (how a full score-curve → one scalar per episode):
    "by earliest stop": s_ep = max(score_t, t < task_min_step)
    "by final end":     s_ep = max(score_t, all t)

    task_min_step = shortest episode length among ALL episodes of that task.
    Using this makes every episode of a task stop at the same horizon, so
    longer successes cannot "inflate" the max score.

  Conformal calibration:
    Split conformal (SAFE's split_cp.py) calibrated on val_seen, tested on
    val_unseen.  Threshold τ_α guarantees recall ≥ 1−α on seen tasks.

  Key evaluation functions from SAFE's codebase (no wandb, no hydra):
    eval_det_time_vs_classification  — detection time vs balanced accuracy
    eval_fixed_threshold             — TPR/TNR/Acc/F1/AUC at τ=0.5
    split_conformal_binary           — conformal τ from val_seen

Outputs (output_dir/)
---------------------
  fig_det_vs_acc.png     Detection time vs balanced accuracy  ← SAFE's Fig 3
  fig_roc_prc.png        ROC + PRC with std bands
  fig_score_curves.png   Score trajectories (success vs failure)
  fig_per_task_auc.png   Per-task AUC on val_unseen
  fig_conformal.png      Conformal recall coverage vs target
  table_results.png      Styled comparison table
  table_results.csv      CSV version (paste into LaTeX)
  summary.txt            Full numeric report

Usage
-----
    # Primary run — uses SAFE's protocol, 3 seeds, 300 epochs
    python scripts/safe_eval.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
        --output_dir ./safe_eval_results

    # Quick sanity check
    python scripts/safe_eval.py \\
        --data_path ... --n_epochs 100 --seeds 0
"""

import os
import sys
import csv
import argparse
import warnings
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, roc_curve, average_precision_score,
    precision_recall_curve,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Path setup ───────────────────────────────────────────────────────────────
_SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
_REPO_ROOT   = os.path.abspath(os.path.join(_SCRIPTS_DIR, ".."))
sys.path.insert(0, _SCRIPTS_DIR)
sys.path.insert(0, _REPO_ROOT)

# ── Our existing modules ─────────────────────────────────────────────────────
from failure_detector import (
    load_rollouts, FailureDetector,
    RolloutDataset, _pad_collate,
)
from combined_detector import CombinedFailureDetector
from best_detector import (
    _safe_time_weights, _safe_hinge_loss,
    train_safe_model, predict_safe,
    train_attn_hinge, predict_attn_hinge,
    MODEL_COLORS, MODEL_LABELS, MODEL_LS,
)

# ── SAFE's evaluation functions (imported directly — no wandb/hydra needed) ──
from failure_prob.utils.metrics import (
    eval_det_time_vs_classification,
    eval_fixed_threshold as _safe_eval_fixed,
)
from failure_prob.utils.conformal.split_cp import split_conformal_binary


# ══════════════════════════════════════════════════════════════════════════════
#  Matplotlib style
# ══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150,
})

_MODELS = ("safe", "attn_h")
_C = lambda n: MODEL_COLORS.get(n, "gray")
_L = lambda n: MODEL_LABELS.get(n, n)
_LS = lambda n: MODEL_LS.get(n, "-")


# ══════════════════════════════════════════════════════════════════════════════
#  SAFE-style task_min_step computation
# ══════════════════════════════════════════════════════════════════════════════

def set_task_min_step(rollouts):
    """Attach task_min_step to every rollout (min episode length for that task).

    This mirrors failure_prob/data/utils.py::set_task_min_step exactly.
    We store the value in a separate dict because our Rollout uses __slots__.
    """
    task_min = {}
    for r in rollouts:
        tid = r.task_id
        L   = r.hidden_states.shape[0]
        task_min[tid] = min(task_min.get(tid, L), L)
    return task_min   # {task_id: min_step}


# ══════════════════════════════════════════════════════════════════════════════
#  SAFE-style data split (train / val_seen / val_unseen)
# ══════════════════════════════════════════════════════════════════════════════

def safe_split(all_r, unseen_task_ratio=0.30, seen_train_ratio=0.60, seed=0):
    """Replicate SAFE's split_rollouts_by_seen_unseen logic.

    Returns dict: {"train": [...], "val_seen": [...], "val_unseen": [...]}
    """
    rng       = np.random.RandomState(seed)
    task_ids  = sorted(set(r.task_id for r in all_r))
    shuffled  = task_ids.copy(); rng.shuffle(shuffled)
    n_unseen  = max(1, round(unseen_task_ratio * len(task_ids)))
    unseen    = set(shuffled[:n_unseen])
    seen      = set(shuffled[n_unseen:])

    train_r, val_seen_r, val_unseen_r = [], [], []
    for tid in seen:
        task_r   = [r for r in all_r if r.task_id == tid]
        perm     = rng.permutation(len(task_r))
        n_train  = max(1, int(seen_train_ratio * len(task_r)))
        train_r    += [task_r[i] for i in perm[:n_train]]
        val_seen_r += [task_r[i] for i in perm[n_train:]]
    val_unseen_r = [r for r in all_r if r.task_id in unseen]

    splits = {"train": train_r, "val_seen": val_seen_r, "val_unseen": val_unseen_r}
    for k, v in splits.items():
        ns = sum(r.episode_success for r in v)
        print(f"  {k}: {len(v)} rollouts  ({ns} success / {len(v)-ns} failure)")
    return splits, seen, unseen


# ══════════════════════════════════════════════════════════════════════════════
#  Scoring: SAFE's max-before-task_min_step  ("by earliest stop")
# ══════════════════════════════════════════════════════════════════════════════

def score_by_earliest_stop(score_curves, rollouts, task_min):
    """SAFE's primary scoring: max(score_t, t < task_min_step).

    Returns 1-D array of per-episode scalars.
    """
    out = []
    for r, sc in zip(rollouts, score_curves):
        curve = sc[0] if isinstance(sc, tuple) else sc  # unwrap (score, weight) tuples
        ms    = task_min.get(r.task_id, len(curve))
        out.append(float(np.max(curve[:ms])))
    return np.array(out, dtype=np.float32)


def score_by_final_end(score_curves, rollouts):
    """SAFE's secondary scoring: max(score_t, all t)."""
    out = []
    for r, sc in zip(rollouts, score_curves):
        curve = sc[0] if isinstance(sc, tuple) else sc
        out.append(float(np.max(curve)))
    return np.array(out, dtype=np.float32)


def full_score_curves(score_curves):
    """Strip tuple wrappers → list of 1-D arrays."""
    return [sc[0] if isinstance(sc, tuple) else sc for sc in score_curves]


# ══════════════════════════════════════════════════════════════════════════════
#  Training one model for one seed
# ══════════════════════════════════════════════════════════════════════════════

def train_and_predict(name, splits, input_dim, args):
    """Returns (score_curves_by_split, losses)."""
    train_r = splits["train"]
    device  = args.device

    if name == "safe":
        m = FailureDetector(input_dim, hidden_dim=args.hidden_dim,
                            n_layers=args.n_layers).to(device)
        ll = train_safe_model(m, train_r, n_epochs=args.n_epochs, lr=args.lr,
                              lambda_reg=args.lambda_reg, threshold=0.5,
                              batch_size=args.batch_size, device=device)
        predict_fn = predict_safe
    elif name == "attn_h":
        m = CombinedFailureDetector(
            hidden_state_dim=input_dim, task_embed_dim=0,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        ).to(device)
        ll = train_attn_hinge(m, train_r, n_epochs=args.n_epochs, lr=args.lr,
                              lambda_reg=args.lambda_reg,
                              lambda_attn=args.lambda_attn, threshold=0.5,
                              batch_size=args.batch_size, device=device)
        predict_fn = predict_attn_hinge
    else:
        raise ValueError(name)

    sc_by_split = {k: predict_fn(m, v, device=device)
                   for k, v in splits.items()}
    return sc_by_split, ll


# ══════════════════════════════════════════════════════════════════════════════
#  Per-seed metrics using SAFE's protocol
# ══════════════════════════════════════════════════════════════════════════════

def compute_safe_metrics(rollouts, score_curves, task_min,
                         cal_rollouts, cal_score_curves, alpha=0.10):
    """Compute SAFE-protocol metrics for one model on one split.

    Returns dict with keys: auc, ap, bal_acc, f1, tpr, fpr, avg_det,
                             tau_cp, recall_cp, far_cp
    """
    labels = np.array([1 - r.episode_success for r in rollouts])
    scores = score_by_earliest_stop(score_curves, rollouts, task_min)
    curves = full_score_curves(score_curves)

    if len(np.unique(labels)) < 2:
        return {k: None for k in ("auc","ap","bal_acc","f1","tpr","fpr",
                                   "avg_det","tau_cp","recall_cp","far_cp")}

    auc = roc_auc_score(labels, scores)
    ap  = average_precision_score(labels, scores)

    # Fixed threshold = 0.5
    preds05  = (scores >= 0.5).astype(int)
    tp = ((preds05 == 1) & (labels == 1)).sum()
    fp = ((preds05 == 1) & (labels == 0)).sum()
    tn = ((preds05 == 0) & (labels == 0)).sum()
    fn = ((preds05 == 0) & (labels == 1)).sum()
    tpr = tp / (tp + fn + 1e-8)
    tnr = tn / (tn + fp + 1e-8)
    fpr = fp / (fp + tn + 1e-8)
    bal_acc = (tpr + tnr) / 2
    prec    = tp / (tp + fp + 1e-8)
    f1      = 2 * prec * tpr / (prec + tpr + 1e-8)

    # Detection time at fixed τ=0.5 (SAFE's eval_detection_time)
    det_times = []
    for s_curve, lbl in zip(curves, labels):
        if lbl == 1:
            cross = np.where(s_curve >= 0.5)[0]
            det_times.append(cross[0] / len(s_curve) if len(cross) > 0 else 1.0)
    avg_det = float(np.mean(det_times)) if det_times else float("nan")

    # Split conformal (calibrate on val_seen, test here)
    cal_labels = np.array([1 - r.episode_success for r in cal_rollouts])
    cal_scores = score_by_earliest_stop(cal_score_curves, cal_rollouts, task_min)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred_sets, thresholds = split_conformal_binary(
            cal_scores.tolist(), cal_labels.tolist(), scores.tolist(), alpha
        )
    tau_cp     = 1.0 - thresholds[1]  # threshold for positive class
    cp_preds   = np.array([1 if 1 in ps else 0 for ps in pred_sets])
    recall_cp  = float(cp_preds[labels == 1].mean()) if labels.sum() > 0 else 0.0
    far_cp     = float(cp_preds[labels == 0].mean()) if (labels == 0).sum() > 0 else 0.0

    # Detection time at conformal τ
    det_cp = []
    for s_curve, lbl in zip(curves, labels):
        if lbl == 1:
            cross = np.where(s_curve >= tau_cp)[0]
            det_cp.append(cross[0] / len(s_curve) if len(cross) > 0 else 1.0)

    return {
        "auc": auc, "ap": ap, "bal_acc": bal_acc, "f1": f1,
        "tpr": tpr, "fpr": fpr, "avg_det": avg_det,
        "tau_cp": tau_cp, "recall_cp": recall_cp, "far_cp": far_cp,
        "avg_det_cp": float(np.mean(det_cp)) if det_cp else float("nan"),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Figure: Detection time vs balanced accuracy  ← SAFE's Fig 3 equivalent
# ══════════════════════════════════════════════════════════════════════════════

def fig_det_vs_acc(model_curves_by_seed, model_rollouts_by_seed, out_dir):
    """Replicates SAFE's detection-time vs. balanced-accuracy Pareto figure.

    Uses SAFE's own eval_det_time_vs_classification under the hood.
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    for name in _MODELS:
        col = _C(name); ls = _LS(name); lbl = _L(name)
        all_dets, all_accs = [], []
        for sc_list, rollouts in zip(model_curves_by_seed[name],
                                     model_rollouts_by_seed[name]):
            labels = np.array([1 - r.episode_success for r in rollouts])
            curves = full_score_curves(sc_list)
            results = eval_det_time_vs_classification(rollouts, curves, labels)
            dets = np.array([r["avg_det_time"] for r in results])
            accs = np.array([r["bal_acc"]      for r in results])
            # Sort by detection time for a clean Pareto curve
            order = np.argsort(dets)
            all_dets.append(dets[order]); all_accs.append(accs[order])

        # Interpolate to a common x-grid for mean/std
        x_grid   = np.linspace(0, 1, 200)
        interped = [np.interp(x_grid, d, a) for d, a in zip(all_dets, all_accs)]
        mu  = np.mean(interped, axis=0)
        std = np.std(interped,  axis=0)

        ax.plot(x_grid, mu, color=col, lw=2.5, ls=ls, label=lbl)
        if len(interped) > 1:
            ax.fill_between(x_grid, mu - std, mu + std, color=col, alpha=0.15)

    ax.set_xlabel("Average normalised detection time  (0 = instant, 1 = end)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Detection Time vs. Balanced Accuracy\n"
                 "(SAFE Fig.3 equivalent — upper-left corner = best)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="gray", ls=":", lw=1, alpha=0.5)
    ax.legend(); ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_det_vs_acc.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure: ROC + PRC with std bands
# ══════════════════════════════════════════════════════════════════════════════

def fig_roc_prc(model_scores_by_seed, model_rollouts_by_seed, task_min, out_dir):
    """ROC and PRC using SAFE's max-before-task_min_step scoring."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    common_fpr = np.linspace(0, 1, 200)
    common_rec = np.linspace(0, 1, 200)

    for name in _MODELS:
        col = _C(name); ls = _LS(name); lbl = _L(name)
        tprs, precs, aucs, aps = [], [], [], []
        for sc_list, rollouts in zip(model_scores_by_seed[name],
                                     model_rollouts_by_seed[name]):
            labels = np.array([1 - r.episode_success for r in rollouts])
            scores = score_by_earliest_stop(sc_list, rollouts, task_min)
            if len(np.unique(labels)) < 2: continue
            fpr, tpr, _ = roc_curve(labels, scores)
            pre, rec, _ = precision_recall_curve(labels, scores)
            tprs.append(np.interp(common_fpr, fpr, tpr))
            precs.append(np.interp(common_rec, rec[::-1], pre[::-1]))
            aucs.append(roc_auc_score(labels, scores))
            aps.append(average_precision_score(labels, scores))

        mu_tpr = np.mean(tprs, 0); sd_tpr = np.std(tprs, 0)
        mu_pre = np.mean(precs, 0); sd_pre = np.std(precs, 0)
        n = len(aucs)
        auc_s = f"{np.mean(aucs):.3f}±{np.std(aucs):.3f}" if n > 1 else f"{np.mean(aucs):.3f}"
        ap_s  = f"{np.mean(aps):.3f}±{np.std(aps):.3f}"  if n > 1 else f"{np.mean(aps):.3f}"

        axes[0].plot(common_fpr, mu_tpr, color=col, lw=2, ls=ls,
                     label=f"{lbl}  AUC={auc_s}")
        axes[0].fill_between(common_fpr, mu_tpr-sd_tpr, mu_tpr+sd_tpr,
                             color=col, alpha=0.15)
        axes[1].plot(common_rec, mu_pre, color=col, lw=2, ls=ls,
                     label=f"{lbl}  AP={ap_s}")
        axes[1].fill_between(common_rec, mu_pre-sd_pre, mu_pre+sd_pre,
                             color=col, alpha=0.15)

    axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=0.4)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC  (scored: max before task_min_step)")
    axes[0].legend(); axes[0].grid(True, alpha=0.25)

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("PRC  (scored: max before task_min_step)")
    axes[1].legend(); axes[1].set_xlim(0,1); axes[1].grid(True, alpha=0.25)

    fig.suptitle("Discriminative Performance on val_unseen — SAFE Scoring Protocol",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_roc_prc.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure: Score trajectories
# ══════════════════════════════════════════════════════════════════════════════

def fig_score_curves(model_scores_by_seed, model_rollouts_by_seed, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), squeeze=False)
    x = np.linspace(0, 1, 100)

    def _interp(arr, n=100):
        return np.interp(np.linspace(0, len(arr)-1, n), np.arange(len(arr)), arr)

    for ax, name in zip(axes[0], _MODELS):
        succ, fail = [], []
        for sc_list, rollouts in zip(model_scores_by_seed[name],
                                     model_rollouts_by_seed[name]):
            for r, sc in zip(rollouts, sc_list):
                curve = sc[0] if isinstance(sc, tuple) else sc
                (succ if r.episode_success else fail).append(_interp(curve))
        for curves, lbl2, col in [
            (succ, f"Success (n={len(succ)})", "#2166ac"),
            (fail, f"Failure (n={len(fail)})", "#d73027"),
        ]:
            if not curves: continue
            arr  = np.stack(curves)
            for trace in arr[:min(30, len(arr))]:
                ax.plot(x, trace, color=col, lw=0.3, alpha=0.15)
            ax.plot(x, arr.mean(0), color=col, lw=2.5, label=lbl2, zorder=3)
            ax.fill_between(x, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0),
                            color=col, alpha=0.2)
        ax.axhline(0.5, color="orange", ls="--", lw=1, alpha=0.7)
        ax.set_title(_L(name)); ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Normalised time"); ax.set_ylabel("Score")
        ax.legend(); ax.grid(True, alpha=0.25)

    fig.suptitle("Score Trajectories (pooled across seeds)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_score_curves.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure: Per-task AUC on val_unseen
# ══════════════════════════════════════════════════════════════════════════════

def fig_per_task_auc(model_scores_by_seed, model_rollouts_by_seed, task_min, out_dir):
    all_tasks = sorted(set(r.task_id
                           for runs in model_rollouts_by_seed.values()
                           for rollouts in runs for r in rollouts))
    fig, ax = plt.subplots(figsize=(max(8, len(all_tasks)), 4))
    w = 0.8 / len(_MODELS)

    for mi, name in enumerate(_MODELS):
        task_aucs = defaultdict(list)
        for sc_list, rollouts in zip(model_scores_by_seed[name],
                                     model_rollouts_by_seed[name]):
            for tid in all_tasks:
                idx = [i for i, r in enumerate(rollouts) if r.task_id == tid]
                if not idx: continue
                yt = np.array([1 - rollouts[i].episode_success for i in idx])
                yf = score_by_earliest_stop(
                    [sc_list[i] for i in idx],
                    [rollouts[i] for i in idx], task_min
                )
                if len(np.unique(yt)) < 2: continue
                task_aucs[tid].append(roc_auc_score(yt, yf))

        x  = np.arange(len(all_tasks))
        mu = [np.mean(task_aucs[t]) if task_aucs[t] else 0 for t in all_tasks]
        sd = [np.std(task_aucs[t])  if len(task_aucs[t]) > 1 else 0 for t in all_tasks]
        offset = (mi - (len(_MODELS)-1)/2) * w
        ax.bar(x + offset, mu, w*0.9, color=_C(name), alpha=0.85,
               label=_L(name), yerr=sd if any(s>0 for s in sd) else None,
               capsize=3, error_kw={"elinewidth": 1})

    ax.set_xticks(np.arange(len(all_tasks)))
    ax.set_xticklabels([f"Task {t}" for t in all_tasks], rotation=30, ha="right")
    ax.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.7)
    ax.set_ylim(0, 1.15); ax.set_ylabel("ROC-AUC (by earliest stop)")
    ax.set_title("Per-Task AUC on val_unseen  (SAFE scoring)")
    ax.legend(); ax.grid(True, alpha=0.25, axis="y")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_per_task_auc.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Figure: Conformal recall coverage
# ══════════════════════════════════════════════════════════════════════════════

def fig_conformal(model_scores_by_seed, model_rollouts_by_seed,
                  model_cal_scores_by_seed, model_cal_rollouts_by_seed,
                  task_min, out_dir):
    """Empirical recall vs target recall using SAFE's split_conformal_binary."""
    alphas  = np.linspace(0.02, 0.45, 40)
    targets = 1 - alphas
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for name in _MODELS:
        col = _C(name); ls = _LS(name); lbl = _L(name)
        emp_rec_all, emp_far_all = [], []
        for sc_te, te_r, sc_ca, ca_r in zip(
                model_scores_by_seed[name], model_rollouts_by_seed[name],
                model_cal_scores_by_seed[name], model_cal_rollouts_by_seed[name]):
            emp_rec_seed, emp_far_seed = [], []
            te_labels = np.array([1 - r.episode_success for r in te_r])
            te_scores  = score_by_earliest_stop(sc_te, te_r, task_min)
            ca_labels  = np.array([1 - r.episode_success for r in ca_r])
            ca_scores  = score_by_earliest_stop(sc_ca, ca_r, task_min)
            for a in alphas:
                try:
                    _, thresholds = split_conformal_binary(
                        ca_scores.tolist(), ca_labels.tolist(),
                        te_scores.tolist(), a)
                    tau = 1.0 - thresholds[1]
                    preds = (te_scores >= tau).astype(int)
                    rec = preds[te_labels==1].mean() if te_labels.sum() > 0 else np.nan
                    far = preds[te_labels==0].mean() if (te_labels==0).sum() > 0 else np.nan
                except Exception:
                    rec = far = np.nan
                emp_rec_seed.append(rec); emp_far_seed.append(far)
            emp_rec_all.append(emp_rec_seed); emp_far_all.append(emp_far_seed)

        mr = np.nanmean(emp_rec_all, 0); sr = np.nanstd(emp_rec_all, 0)
        mf = np.nanmean(emp_far_all, 0); sf = np.nanstd(emp_far_all, 0)
        axes[0].plot(targets, mr, color=col, lw=2, ls=ls, label=lbl)
        axes[0].fill_between(targets, mr-sr, mr+sr, color=col, alpha=0.15)
        axes[1].plot(targets, mf, color=col, lw=2, ls=ls, label=lbl)
        axes[1].fill_between(targets, mf-sf, mf+sf, color=col, alpha=0.15)

    axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=0.5,label="Perfect coverage")
    axes[0].set_xlabel("Target recall (1−α)"); axes[0].set_ylabel("Empirical recall")
    axes[0].set_title("Conformal Recall Coverage\n(calibrated on val_seen, tested on val_unseen)")
    axes[0].set_xlim(0.5, 1.02); axes[0].legend(); axes[0].grid(True, alpha=0.25)
    axes[1].set_xlabel("Target recall (1−α)"); axes[1].set_ylabel("FAR (cost)")
    axes[1].set_title("False-Alarm Rate at Each Coverage Level")
    axes[1].set_xlim(0.5, 1.02); axes[1].legend(); axes[1].grid(True, alpha=0.25)
    fig.suptitle("Conformal Prediction: SAFE's Split-CP Protocol",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "fig_conformal.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Table 1 — styled comparison
# ══════════════════════════════════════════════════════════════════════════════

def make_table(agg, n_seeds, out_dir):
    metrics    = ["auc", "ap", "bal_acc", "f1", "avg_det", "recall_cp", "far_cp"]
    col_labels = ["Model", "AUC ↑", "AP ↑", "Bal-Acc ↑", "F1 ↑",
                  "Det.Time ↓", "Recall@τ ↑", "FAR@τ ↓"]
    rows = []
    for name in _MODELS:
        ms  = agg[name]
        row = [_L(name)]
        for k in metrics:
            mu, sd = ms[k]
            if mu is None:   row.append("—")
            elif n_seeds > 1: row.append(f"{mu:.3f}±{sd:.3f}")
            else:             row.append(f"{mu:.4f}")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(16, 1.8 + 0.9*len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 2.8)
    for j in range(len(col_labels)):
        tbl[0,j].set_facecolor("#1a3a5c")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i, name in enumerate(_MODELS, 1):
        tbl[i,0].set_facecolor(_C(name))
        tbl[i,0].set_text_props(color="white", fontweight="bold")
        for j in range(1, len(col_labels)):
            tbl[i,j].set_facecolor("#f5f9ff" if i%2==0 else "white")
    seed_str = f"mean±std {n_seeds} seeds" if n_seeds > 1 else "single seed"
    ax.set_title(
        f"Table 1: val_unseen — SAFE evaluation protocol  ({seed_str}, "
        f"scored: max before task_min_step)",
        fontsize=11, fontweight="bold", pad=14, y=0.98)
    fig.tight_layout()
    p = os.path.join(out_dir, "table_results.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {p}")

    csv_p = os.path.join(out_dir, "table_results.csv")
    with open(csv_p, "w", newline="") as f:
        csv.writer(f).writerows([col_labels] + rows)
    print(f"  -> {csv_p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Text summary
# ══════════════════════════════════════════════════════════════════════════════

def write_summary(agg, n_seeds, args, out_dir):
    lines = [
        "="*72, "SAFE-PROTOCOL EVALUATION — RESULTS SUMMARY", "="*72,
        f"  Data:           {args.data_path}",
        f"  Seeds:          {args.seeds}",
        f"  Epochs:         {args.n_epochs}",
        f"  Unseen ratio:   {args.unseen_task_ratio*100:.0f}% of task IDs",
        f"  Seen-train:     {args.seen_train_ratio*100:.0f}% of seen-task rollouts",
        f"  SAFE scoring:   max(score[:task_min_step])  ('by earliest stop')",
        f"  Conformal α:    {1-args.target_recall:.2f}  (target recall {args.target_recall*100:.0f}%)",
        "", "NUMERIC RESULTS (val_unseen)", "-"*72,
    ]
    for name in _MODELS:
        ms = agg[name]
        def _v(k):
            mu, sd = ms[k]
            if mu is None: return "—"
            return f"{mu:.4f}" + (f" ± {sd:.4f}" if n_seeds > 1 else "")
        lines += [
            f"\n  ── {_L(name)} ──",
            f"  AUC  (by earliest stop): {_v('auc')}",
            f"  AP:                       {_v('ap')}",
            f"  Balanced Acc @ 0.5:       {_v('bal_acc')}",
            f"  F1 @ 0.5:                 {_v('f1')}",
            f"  Avg det time (τ=0.5):     {_v('avg_det')}",
            f"  Recall @ conformal τ:     {_v('recall_cp')}",
            f"  FAR @ conformal τ:        {_v('far_cp')}",
        ]

    best_auc  = max(_MODELS, key=lambda n: agg[n]["auc"][0] or 0)
    best_det  = min(_MODELS, key=lambda n: agg[n]["avg_det"][0] or 1)
    best_bacc = max(_MODELS, key=lambda n: agg[n]["bal_acc"][0] or 0)
    lines += [
        "", "="*72, "SUMMARY", "-"*72,
        f"  Best AUC (by earliest stop): {_L(best_auc)}  "
        f"({agg[best_auc]['auc'][0]:.4f})",
        f"  Best detection time:         {_L(best_det)}  "
        f"({agg[best_det]['avg_det'][0]:.4f})",
        f"  Best balanced accuracy:      {_L(best_bacc)}  "
        f"({agg[best_bacc]['bal_acc'][0]:.4f})",
        "", "="*72,
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    p = os.path.join(out_dir, "summary.txt")
    with open(p, "w") as f: f.write(txt)
    print(f"\n  -> {p}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="SAFE-protocol evaluation — fair comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",          required=True)
    parser.add_argument("--output_dir",         default="./safe_eval_results")
    parser.add_argument("--seeds",              nargs="+", type=int, default=[0,1,2])
    parser.add_argument("--unseen_task_ratio",  type=float, default=0.30)
    parser.add_argument("--seen_train_ratio",   type=float, default=0.60,
                        help="SAFE's seen_train_ratio (fraction of seen tasks used for training)")
    parser.add_argument("--target_recall",      type=float, default=0.90)
    parser.add_argument("--n_epochs",           type=int,   default=300)
    parser.add_argument("--lr",                 type=float, default=1e-3)
    parser.add_argument("--lambda_reg",         type=float, default=1e-2)
    parser.add_argument("--lambda_attn",        type=float, default=0.10)
    parser.add_argument("--hidden_dim",         type=int,   default=256)
    parser.add_argument("--n_layers",           type=int,   default=2)
    parser.add_argument("--batch_size",         type=int,   default=32)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\nModels:  {list(_MODELS)}")
    print(f"Seeds:   {args.seeds}")
    print(f"Epochs:  {args.n_epochs}   Device: {args.device}")
    print(f"Scoring: max(score[:task_min_step])  ('by earliest stop' — SAFE protocol)")
    print(f"Output:  {os.path.abspath(args.output_dir)}/")

    # ── Load ──────────────────────────────────────────────────────────────────
    print("\n[1/3] Loading rollouts ...")
    all_r     = load_rollouts(args.data_path)
    input_dim = all_r[0].hidden_states.shape[-1]
    task_min  = set_task_min_step(all_r)
    ns = sum(r.episode_success for r in all_r)
    print(f"  {len(all_r)} rollouts  ({ns} success / {len(all_r)-ns} failure)")
    print(f"  task_min_step range: {min(task_min.values())} – {max(task_min.values())}")

    # ── Train × seeds ─────────────────────────────────────────────────────────
    print("\n[2/3] Training ...")
    all_seed_scores_unseen  = defaultdict(list)   # for ROC/PRC/table
    all_seed_rollouts_unseen= defaultdict(list)
    all_seed_scores_seen    = defaultdict(list)   # for conformal calibration
    all_seed_rollouts_seen  = defaultdict(list)
    all_seed_curves_unseen  = defaultdict(list)   # full curves for det-time
    all_seed_curves_rollouts= defaultdict(list)
    per_seed_metrics        = []

    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        splits, seen_ids, unseen_ids = safe_split(
            all_r, args.unseen_task_ratio, args.seen_train_ratio, seed
        )
        # (safe_split returns splits, seen, unseen)

        seed_metrics = {}
        for name in _MODELS:
            print(f"\n  [seed={seed}] Training {_L(name)} ...")
            sc_by_split, ll = train_and_predict(name, splits, input_dim, args)

            # Accumulate score curves per split
            all_seed_scores_unseen[name].append(sc_by_split["val_unseen"])
            all_seed_rollouts_unseen[name].append(splits["val_unseen"])
            all_seed_scores_seen[name].append(sc_by_split["val_seen"])
            all_seed_rollouts_seen[name].append(splits["val_seen"])
            # Full curves for detection-time analysis
            all_seed_curves_unseen[name].append(sc_by_split["val_unseen"])
            all_seed_curves_rollouts[name].append(splits["val_unseen"])

            # Metrics on val_unseen using SAFE's scoring
            m = compute_safe_metrics(
                rollouts    = splits["val_unseen"],
                score_curves= sc_by_split["val_unseen"],
                task_min    = task_min,
                cal_rollouts= splits["val_seen"],
                cal_score_curves= sc_by_split["val_seen"],
                alpha       = 1.0 - args.target_recall,
            )
            seed_metrics[name] = m

        per_seed_metrics.append(seed_metrics)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = {}
    for name in _MODELS:
        agg[name] = {}
        for key in per_seed_metrics[0][name]:
            vals = [d[name][key] for d in per_seed_metrics
                    if d[name][key] is not None]
            if vals:
                agg[name][key] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                agg[name][key] = (None, None)

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n[3/3] Generating figures ...")
    fig_det_vs_acc(all_seed_curves_unseen, all_seed_curves_rollouts, args.output_dir)
    fig_roc_prc(all_seed_scores_unseen, all_seed_rollouts_unseen, task_min, args.output_dir)
    fig_score_curves(all_seed_curves_unseen, all_seed_curves_rollouts, args.output_dir)
    fig_per_task_auc(all_seed_scores_unseen, all_seed_rollouts_unseen, task_min, args.output_dir)
    fig_conformal(all_seed_scores_unseen, all_seed_rollouts_unseen,
                  all_seed_scores_seen,   all_seed_rollouts_seen,
                  task_min, args.output_dir)
    make_table(agg, len(args.seeds), args.output_dir)
    write_summary(agg, len(args.seeds), args, args.output_dir)

    print(f"\n  All outputs saved to: {os.path.abspath(args.output_dir)}/")
    print("\n  Key figures:")
    print("    fig_det_vs_acc.png   ← SAFE's Fig.3 equivalent (detection time vs bal-acc)")
    print("    fig_roc_prc.png      ROC + PRC  (max-before-task_min_step scoring)")
    print("    fig_conformal.png    Conformal coverage guarantee")
    print("    table_results.csv    Numbers for LaTeX table")


if __name__ == "__main__":
    main()
