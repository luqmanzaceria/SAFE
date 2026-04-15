#!/usr/bin/env python3
"""
analyze_comparison.py  —  Deep Analysis: SAFE vs Our Temporal Attention Model
==============================================================================

Reads the outputs of compare_with_safe.py and generates a comprehensive
publication-quality analysis showing every dimension where our model wins.

Figures generated
-----------------
  01_metric_bars.png       Grouped bar chart (all metrics, error bars)
  02_improvement.png       Per-metric % improvement with confidence interval
  03_roc_overlay.png       ROC curves — std bands, AUC callout
  04_prc_overlay.png       Precision-Recall curves
  05_score_dist.png        Violin plots — score distributions
  06_detection_cdf.png     Detection time CDF (failure episodes only)
  07_score_trajectories.png Mean ± std score curves for success vs failure
  08_per_task_delta.png    Per-task AUC: SAFE vs Ours side-by-side + delta
  09_radar.png             Radar / spider chart — all normalised metrics
  10_significance.png      Statistical test table (t-test p-values, Cohen's d)
  11_confusion_heatmap.png Confusion matrices (SAFE vs Ours) at τ=0.5
  12_dashboard.png         Single-page summary combining key panels

Stats written to
----------------
  stats_report.txt    Full statistical significance report
  per_seed_table.csv  Raw per-seed numbers for supplementary material

Usage
-----
    # Quick — reads existing compare_results/ folder
    python scripts/analyze_comparison.py \\
        --results_dir ./compare_results

    # Full re-run (trains models again, then analyses)
    python scripts/analyze_comparison.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
        --rerun --seeds 0 1 2 --n_epochs 300 --device cuda

    # Just the dashboard (fast)
    python scripts/analyze_comparison.py \\
        --results_dir ./compare_results --dashboard_only
"""

import os
import sys
import csv
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS_FILE   = os.path.realpath(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)
_REPO_ROOT   = os.path.dirname(_SCRIPTS_DIR)
for _p in (_SCRIPTS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Style constants ───────────────────────────────────────────────────────────
C_SAFE = "#4dac26"   # green — SAFE
C_OURS = "#d73027"   # red   — ours (better)
C_GREY = "#aaaaaa"

SAFE_LABEL = "SAFE IndepModel\n(official, NeurIPS 2025)"
OURS_LABEL = "Temporal Attention + Hinge\n(ours)"

plt.rcParams.update({
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv_results(results_dir: str) -> dict | None:
    """Load the results.csv written by compare_with_safe.py."""
    path = Path(results_dir) / "results.csv"
    if not path.exists():
        return None
    rows = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = "safe" if "SAFE" in row["model"] else "ours"
            rows[name] = {k: float(v) for k, v in row.items() if k != "model"}
    return rows


def rerun_comparison(args) -> dict:
    """Re-run compare_with_safe.py and return seed_results + agg."""
    from compare_with_safe import (
        load_rollouts, make_split, task_min_steps,
        train_safe_model, predict_safe_model, _make_safe_cfg,
        compute_metrics, aggregate,
    )
    from failure_prob.model.indep import IndepModel
    from combined_detector import CombinedFailureDetector
    from best_detector import train_attn_hinge, predict_attn_hinge
    import random, torch

    rollouts = load_rollouts(args.data_path)
    seed_results = []

    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        random.seed(seed); torch.manual_seed(seed)

        splits  = make_split(rollouts, rng=rng)
        train_r = splits["train"]
        val_r   = splits["val_unseen"]
        if not (sum(r.episode_success for r in train_r) > 0 and
                sum(1-r.episode_success for r in train_r) > 0):
            continue

        input_dim = rollouts[0].hidden_states.shape[1]
        tms_all   = task_min_steps(rollouts)

        # SAFE
        safe_cfg   = _make_safe_cfg(lr=args.lr, lambda_reg=args.lambda_reg)
        safe_model = IndepModel(safe_cfg, input_dim).to(args.device)
        train_safe_model(safe_model, train_r, n_epochs=args.n_epochs,
                         lr=args.lr, lambda_reg=args.lambda_reg,
                         batch_size=args.batch_size, device=args.device)
        safe_curves = predict_safe_model(safe_model, val_r, args.device)

        # Ours
        attn_model = CombinedFailureDetector(
            hidden_state_dim=input_dim, task_embed_dim=0,
            hidden_dim=256, n_layers=2, dropout=0.1,
        ).to(args.device)
        train_attn_hinge(attn_model, train_r, n_epochs=args.n_epochs,
                         lr=args.lr, lambda_reg=args.lambda_reg,
                         device=args.device)
        raw_ours   = predict_attn_hinge(attn_model, val_r, args.device)
        ours_curves = [sc if not isinstance(sc, tuple) else sc[0]
                       for sc in raw_ours]

        sr = dict(
            safe=dict(metrics=compute_metrics(safe_curves, None, tms_all, val_r),
                      curves=safe_curves),
            ours=dict(metrics=compute_metrics(ours_curves, None, tms_all, val_r),
                      curves=ours_curves),
            _val_r=val_r, _tms_all=tms_all,
        )
        seed_results.append(sr)

    return seed_results, aggregate(seed_results)


def extract_per_seed(seed_results: list, metric: str) -> tuple[list, list]:
    """Return (safe_vals, ours_vals) per seed for a given metric."""
    sv = [sr["safe"]["metrics"][metric] for sr in seed_results
          if metric in sr["safe"]["metrics"]]
    ov = [sr["ours"]["metrics"][metric] for sr in seed_results
          if metric in sr["ours"]["metrics"]]
    return sv, ov


# ═══════════════════════════════════════════════════════════════════════════════
#  Statistical tests
# ═══════════════════════════════════════════════════════════════════════════════

def cohens_d(a, b):
    """Cohen's d effect size."""
    na, nb = len(a), len(b)
    pooled = np.sqrt(((na - 1) * np.std(a, ddof=1)**2 +
                      (nb - 1) * np.std(b, ddof=1)**2) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / (pooled + 1e-9)


def run_stats(seed_results: list) -> dict:
    """Run paired t-tests and compute effect sizes for all metrics."""
    metrics = ["auc", "ap", "bal_acc", "f1", "avg_det", "far"]
    results = {}
    for m in metrics:
        sv, ov = extract_per_seed(seed_results, m)
        if len(sv) < 2:
            results[m] = dict(p=float("nan"), d=float("nan"),
                              mean_diff=float("nan"), ci95=float("nan"))
            continue
        t, p    = scipy_stats.ttest_rel(ov, sv)   # ours vs safe
        d       = cohens_d(ov, sv)
        diffs   = np.array(ov) - np.array(sv)
        se      = scipy_stats.sem(diffs)
        ci95    = se * scipy_stats.t.ppf(0.975, df=len(diffs)-1)
        results[m] = dict(p=float(p), d=float(d),
                          mean_diff=float(np.mean(diffs)), ci95=float(ci95))
    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 01 — Grouped bar chart
# ═══════════════════════════════════════════════════════════════════════════════

METRIC_META = {
    #  key        display label        higher=better?
    "auc":      ("ROC-AUC",           True),
    "ap":       ("Avg Precision",     True),
    "bal_acc":  ("Balanced Acc",      True),
    "f1":       ("F1 Score",          True),
    "avg_det":  ("Avg Det Time",      False),
    "far":      ("False Alarm Rate",  False),
}


def fig_metric_bars(seed_results, out_dir):
    metrics = list(METRIC_META.keys())
    n = len(metrics)
    x = np.arange(n)
    w = 0.32

    fig, ax = plt.subplots(figsize=(13, 5))

    for i, name in enumerate(("safe", "ours")):
        vals = []; errs = []
        for m in metrics:
            sv, ov = extract_per_seed(seed_results, m)
            data = sv if name == "safe" else ov
            vals.append(np.nanmean(data))
            errs.append(np.nanstd(data))
        col   = C_SAFE if name == "safe" else C_OURS
        lbl   = "SAFE IndepModel (official)" if name == "safe" \
                else "Temporal Attention + Hinge (ours)"
        offset = -w/2 if name == "safe" else w/2
        bars = ax.bar(x + offset, vals, w, yerr=errs, capsize=4,
                      color=col, alpha=0.85, label=lbl, error_kw=dict(lw=1.5))

    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_META[m][0] for m in metrics], fontsize=10)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("All Metrics: SAFE vs Ours  (mean ± std, 3 seeds, val_unseen)")

    # Mark which direction is better
    for i, m in enumerate(metrics):
        _, higher_better = METRIC_META[m]
        sym = "↑" if higher_better else "↓"
        ax.text(x[i], -0.06, sym, ha="center", fontsize=10, color="#555")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "01_metric_bars.png"), bbox_inches="tight")
    plt.close(fig)
    print("  01_metric_bars.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 02 — % improvement with CI
# ═══════════════════════════════════════════════════════════════════════════════

def fig_improvement(seed_results, out_dir):
    metrics = list(METRIC_META.keys())
    pcts = []; cis = []; colors = []

    for m in metrics:
        sv, ov = extract_per_seed(seed_results, m)
        _, higher_better = METRIC_META[m]
        base = np.nanmean(sv)
        if base == 0:
            pcts.append(0); cis.append(0); colors.append(C_GREY)
            continue
        diffs = (np.array(ov) - np.array(sv)) / (np.abs(np.array(sv)) + 1e-8) * 100
        if not higher_better:
            diffs = -diffs    # flip: lower is better → positive improvement
        pcts.append(float(np.nanmean(diffs)))
        se  = scipy_stats.sem(diffs)
        ci  = se * scipy_stats.t.ppf(0.975, df=len(diffs)-1) if len(diffs) > 1 else 0
        cis.append(float(ci))
        colors.append(C_OURS if np.nanmean(diffs) > 0 else C_SAFE)

    y = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(y, pcts, xerr=cis, capsize=4,
                   color=colors, alpha=0.85, height=0.55)
    ax.axvline(0, color="k", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([METRIC_META[m][0] for m in metrics])
    ax.set_xlabel("% improvement over SAFE  (positive = ours wins)")
    ax.set_title("Relative Improvement: Temporal Attention vs SAFE\n"
                 "(error bars = 95% CI across 3 seeds)")

    for i, (pct, ci) in enumerate(zip(pcts, cis)):
        sign = "+" if pct >= 0 else ""
        ax.text(pct + (ci + 0.3) * np.sign(pct),
                i, f"{sign}{pct:.1f}%", va="center", fontsize=9,
                color=C_OURS if pct > 0 else C_SAFE, fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "02_improvement.png"), bbox_inches="tight")
    plt.close(fig)
    print("  02_improvement.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 03+04 — ROC and PRC
# ═══════════════════════════════════════════════════════════════════════════════

def _episode_scores(curves, val_r, tms_all):
    from compare_with_safe import episode_score
    ep_sc  = np.array([episode_score(sc, tms_all[r.task_id])
                        for sc, r in zip(curves, val_r)])
    ep_lbl = np.array([1 - r.episode_success for r in val_r])
    return ep_sc, ep_lbl


def fig_roc_prc(seed_results, out_dir):
    from sklearn.metrics import (roc_curve, roc_auc_score,
                                 precision_recall_curve, average_precision_score)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    base_fpr = np.linspace(0, 1, 300)
    base_rec = np.linspace(0, 1, 300)

    for name, col, lbl in (("safe", C_SAFE, "SAFE IndepModel"),
                            ("ours", C_OURS, "Temporal Attention (ours)")):
        tprs, precs, aucs, aps = [], [], [], []
        for sr in seed_results:
            ep_sc, ep_lbl = _episode_scores(
                sr[name]["curves"], sr["_val_r"], sr["_tms_all"])
            if len(np.unique(ep_lbl)) < 2:
                continue
            fpr, tpr, _ = roc_curve(ep_lbl, ep_sc)
            tprs.append(np.interp(base_fpr, fpr, tpr))
            aucs.append(roc_auc_score(ep_lbl, ep_sc))
            pr, rc, _   = precision_recall_curve(ep_lbl, ep_sc)
            precs.append(np.interp(base_rec, rc[::-1], pr[::-1]))
            aps.append(average_precision_score(ep_lbl, ep_sc))

        # ROC
        m_tpr = np.mean(tprs, 0); s_tpr = np.std(tprs, 0)
        axes[0].plot(base_fpr, m_tpr, color=col, lw=2,
                     label=f"{lbl}  AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}")
        axes[0].fill_between(base_fpr, m_tpr-s_tpr, m_tpr+s_tpr,
                             alpha=0.15, color=col)
        # PRC
        m_pr = np.mean(precs, 0); s_pr = np.std(precs, 0)
        axes[1].plot(base_rec, m_pr, color=col, lw=2,
                     label=f"{lbl}  AP={np.mean(aps):.3f}±{np.std(aps):.3f}")
        axes[1].fill_between(base_rec, m_pr-s_pr, m_pr+s_pr,
                             alpha=0.15, color=col)

    # ROC diagonal
    axes[0].plot([0,1],[0,1],"k:",lw=0.8)
    axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC Curve — val_unseen (unseen tasks)")
    axes[0].legend(fontsize=8, loc="lower right")
    axes[0].set_xlim(0,1); axes[0].set_ylim(0,1)

    # Chance baseline for PRC
    fail_rate = np.mean([1-r.episode_success for sr in seed_results
                         for r in sr["_val_r"]])
    axes[1].axhline(fail_rate, color="k", ls=":", lw=0.8,
                    label=f"Chance ({fail_rate:.2f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve — val_unseen")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_xlim(0,1); axes[1].set_ylim(0,1)

    fig.suptitle("Detection Performance on Completely Unseen Tasks",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "03_roc_prc.png"), bbox_inches="tight")
    plt.close(fig)
    print("  03_roc_prc.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 04 — Score distributions (violin)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_score_violin(seed_results, out_dir):
    from compare_with_safe import episode_score

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax, (name, col, lbl) in zip(
            axes,
            [("safe", C_SAFE, "SAFE IndepModel"),
             ("ours", C_OURS, "Temporal Attention (ours)")]):

        succ_scores, fail_scores = [], []
        for sr in seed_results:
            tms = sr["_tms_all"]
            for sc, r in zip(sr[name]["curves"], sr["_val_r"]):
                s = episode_score(sc, tms[r.task_id])
                if r.episode_success:
                    succ_scores.append(s)
                else:
                    fail_scores.append(s)

        data   = [succ_scores, fail_scores]
        vparts = ax.violinplot(data, positions=[0, 1], showmedians=True,
                               showextrema=True)
        for pc, c in zip(vparts["bodies"], [C_SAFE, C_OURS]):
            pc.set_facecolor(col)
            pc.set_alpha(0.6)
        vparts["cmedians"].set_color("white")
        vparts["cmedians"].set_linewidth(2)

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Success\n(should be LOW)", "Failure\n(should be HIGH)"])
        ax.set_ylabel("Episode score")
        ax.set_title(lbl)
        ax.axhline(0.5, color="k", ls=":", lw=0.8, label="threshold τ=0.5")
        ax.legend(fontsize=8)

        # Separation metric (mean difference)
        sep = np.mean(fail_scores) - np.mean(succ_scores)
        ax.text(0.5, 0.97, f"Separation = {sep:+.3f}",
                transform=ax.transAxes, ha="center", va="top",
                fontsize=10, fontweight="bold",
                color=C_OURS if col == C_OURS else C_SAFE)

    fig.suptitle("Score Distributions: Success vs Failure Episodes  (val_unseen)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "04_score_violin.png"), bbox_inches="tight")
    plt.close(fig)
    print("  04_score_violin.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 05 — Detection time CDF
# ═══════════════════════════════════════════════════════════════════════════════

def fig_detection_cdf(seed_results, out_dir):
    from compare_with_safe import detection_time

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: CDF of detection times
    ax = axes[0]
    for name, col, lbl, ls in (
            ("safe", C_SAFE, "SAFE IndepModel", "--"),
            ("ours", C_OURS, "Temporal Attention (ours)", "-")):
        all_det = []
        for sr in seed_results:
            for sc, r in zip(sr[name]["curves"], sr["_val_r"]):
                if r.episode_success == 0:
                    dt = detection_time(sc, 0.5)
                    if dt is not None:
                        all_det.append(dt)
        if not all_det:
            continue
        xs = np.sort(all_det)
        ys = np.arange(1, len(xs)+1) / len(xs)
        ax.plot(xs, ys, color=col, lw=2.5, ls=ls, label=lbl)
        med = np.median(xs)
        ax.axvline(med, color=col, ls=":", lw=1.2, alpha=0.7)
        ax.text(med+0.01, 0.05, f"med={med:.2f}", color=col, fontsize=8)

    ax.axvline(0.5, color="k", ls=":", lw=0.8, alpha=0.5, label="t=0.5 (midpoint)")
    ax.set_xlabel("Normalised detection time  (0=start, 1=end)")
    ax.set_ylabel("Cumulative fraction of failure episodes")
    ax.set_title("Detection Time CDF\n(higher = earlier detection)")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    # Right: Box plot of detection times per seed
    ax2 = axes[1]
    safe_dets, ours_dets = [], []
    for sr in seed_results:
        for name, container in (("safe", safe_dets), ("ours", ours_dets)):
            dts = [detection_time(sc, 0.5)
                   for sc, r in zip(sr[name]["curves"], sr["_val_r"])
                   if r.episode_success == 0
                   and detection_time(sc, 0.5) is not None]
            container.extend(dts)

    bp = ax2.boxplot([safe_dets, ours_dets], patch_artist=True,
                     notch=True, widths=0.45,
                     medianprops=dict(color="white", lw=2))
    for patch, col in zip(bp["boxes"], [C_SAFE, C_OURS]):
        patch.set_facecolor(col); patch.set_alpha(0.7)
    ax2.set_xticks([1, 2])
    ax2.set_xticklabels(["SAFE IndepModel", "Ours (Attn)"])
    ax2.set_ylabel("Normalised detection time")
    ax2.set_title("Detection Time Box Plot\n(lower = better, earlier detection)")

    if safe_dets and ours_dets:
        _, p = scipy_stats.mannwhitneyu(safe_dets, ours_dets, alternative="greater")
        ax2.text(0.5, 0.97, f"Mann-Whitney U  p={p:.3f}",
                 transform=ax2.transAxes, ha="center", va="top", fontsize=9,
                 color=("green" if p < 0.05 else "gray"))

    fig.suptitle("Failure Detection Timing  (earlier = safer intervention)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "05_detection_cdf.png"), bbox_inches="tight")
    plt.close(fig)
    print("  05_detection_cdf.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 06 — Score trajectories
# ═══════════════════════════════════════════════════════════════════════════════

def fig_trajectories(seed_results, out_dir):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)

    models = [("safe", C_SAFE, "SAFE IndepModel"),
              ("ours", C_OURS, "Temporal Attention (ours)")]

    for col_i, (name, col, lbl) in enumerate(models):
        # collect all success / failure curves from all seeds
        succ_curves, fail_curves = [], []
        for sr in seed_results:
            for sc, r in zip(sr[name]["curves"], sr["_val_r"]):
                T    = len(sc)
                xs   = np.linspace(0, 1, T)
                (succ_curves if r.episode_success else fail_curves).append((xs, sc))

        for row_i, (curves, outcome_lbl, outcome_col) in enumerate(
                [(succ_curves, "Success episodes", "#2166ac"),
                 (fail_curves, "Failure episodes", "#d73027")]):
            ax = axes[row_i][col_i]
            if not curves:
                ax.set_visible(False); continue

            # Interpolate to common grid
            grid = np.linspace(0, 1, 200)
            mat  = np.vstack([np.interp(grid, xs, sc) for xs, sc in curves])
            mean = mat.mean(0); std = mat.std(0)

            ax.plot(grid, mean, color=outcome_col, lw=2.5)
            ax.fill_between(grid, mean-std, mean+std, alpha=0.2, color=outcome_col)
            # Individual trajectories (light)
            for xs, sc in curves[:20]:
                ax.plot(xs, np.interp(xs, xs, sc), color=outcome_col,
                        alpha=0.07, lw=0.8)

            ax.axhline(0.5, color="k", ls=":", lw=0.8)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xlabel("Normalised timestep")
            ax.set_ylabel("Failure score")
            ax.set_title(f"{lbl}\n{outcome_lbl}  (n={len(curves)})")

            # Annotate separation
            final_mean = mean[-1]
            ax.text(0.97, 0.05 if outcome_lbl.startswith("S") else 0.95,
                    f"final={final_mean:.3f}",
                    transform=ax.transAxes, ha="right",
                    va="bottom" if outcome_lbl.startswith("S") else "top",
                    fontsize=9, color=outcome_col)

    fig.suptitle("Score Trajectories: SAFE vs Ours\n"
                 "(Success = stay low, Failure = rise high → early)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "06_score_trajectories.png"), bbox_inches="tight")
    plt.close(fig)
    print("  06_score_trajectories.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 07 — Per-task AUC delta
# ═══════════════════════════════════════════════════════════════════════════════

def fig_per_task(seed_results, out_dir):
    from sklearn.metrics import roc_auc_score
    from compare_with_safe import episode_score

    sr      = seed_results[0]
    val_r   = sr["_val_r"]
    tms_all = sr["_tms_all"]
    tasks   = sorted(set(r.task_id for r in val_r))

    safe_aucs, ours_aucs, valid_tasks = [], [], []
    for tid in tasks:
        sub_r = [r for r in val_r if r.task_id == tid]
        lbls  = [1 - r.episode_success for r in sub_r]
        if len(np.unique(lbls)) < 2:
            continue
        valid_tasks.append(tid)
        for name, container in (("safe", safe_aucs), ("ours", ours_aucs)):
            sub_sc = [sr[name]["curves"][val_r.index(r)] for r in sub_r]
            ep_sc  = [episode_score(sc, tms_all[tid]) for sc in sub_sc]
            try:
                container.append(roc_auc_score(lbls, ep_sc))
            except Exception:
                container.append(float("nan"))

    if not valid_tasks:
        return

    safe_a = np.array(safe_aucs); ours_a = np.array(ours_aucs)
    delta  = ours_a - safe_a
    x      = np.arange(len(valid_tasks))
    w      = 0.35

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(valid_tasks)*0.9), 9),
                                   gridspec_kw={"height_ratios": [2, 1]})

    # Top: grouped bars
    ax1.bar(x - w/2, safe_a, w, color=C_SAFE, alpha=0.85,
            label="SAFE IndepModel")
    ax1.bar(x + w/2, ours_a, w, color=C_OURS, alpha=0.85,
            label="Temporal Attention (ours)")
    ax1.axhline(0.5, color="k", ls=":", lw=0.8)
    ax1.set_ylabel("AUC"); ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"T{t}" for t in valid_tasks], rotation=45, ha="right")
    ax1.set_title("Per-Task AUC on val_unseen (seed 0)")
    ax1.legend(fontsize=9)

    # Bottom: delta
    colors_d = [C_OURS if d >= 0 else C_SAFE for d in delta]
    ax2.bar(x, delta, color=colors_d, alpha=0.85)
    ax2.axhline(0, color="k", lw=1)
    ax2.set_ylabel("Δ AUC (ours − SAFE)")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"T{t}" for t in valid_tasks], rotation=45, ha="right")
    wins = int((delta > 0).sum())
    ax2.set_title(f"Our model wins on {wins}/{len(valid_tasks)} tasks  "
                  f"(mean Δ = {delta.mean():+.3f})")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "07_per_task_auc.png"), bbox_inches="tight")
    plt.close(fig)
    print("  07_per_task_auc.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 08 — Radar chart
# ═══════════════════════════════════════════════════════════════════════════════

def fig_radar(seed_results, out_dir):
    metrics = ["auc", "ap", "bal_acc", "f1", "avg_det", "far"]
    labels  = ["AUC", "Avg\nPrec", "Bal\nAcc", "F1", "Det\nTime↓", "FAR↓"]

    safe_vals, ours_vals = [], []
    for m in metrics:
        sv, ov = extract_per_seed(seed_results, m)
        _, higher_better = METRIC_META[m]
        sm, om = np.nanmean(sv), np.nanmean(ov)
        # Normalise so higher = better on radar
        if not higher_better:
            sm, om = 1 - sm, 1 - om
        safe_vals.append(sm)
        ours_vals.append(om)

    N   = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]
    safe_vals += safe_vals[:1]
    ours_vals += ours_vals[:1]

    fig, ax = plt.subplots(figsize=(6, 6),
                           subplot_kw=dict(polar=True))
    ax.plot(angles, safe_vals, color=C_SAFE, lw=2, ls="--",
            label="SAFE IndepModel")
    ax.fill(angles, safe_vals, color=C_SAFE, alpha=0.15)
    ax.plot(angles, ours_vals, color=C_OURS, lw=2,
            label="Temporal Attention (ours)")
    ax.fill(angles, ours_vals, color=C_OURS, alpha=0.20)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7)
    ax.set_title("Performance Profile\n(all axes: higher = better)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=9)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "08_radar.png"), bbox_inches="tight")
    plt.close(fig)
    print("  08_radar.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 09 — Statistical significance table
# ═══════════════════════════════════════════════════════════════════════════════

def fig_stats_table(seed_results, out_dir):
    stat = run_stats(seed_results)
    metrics = list(METRIC_META.keys())

    rows = []
    for m in metrics:
        lbl, hb = METRIC_META[m]
        s = stat[m]
        p_str = f"{s['p']:.3f}" if not np.isnan(s['p']) else "n/a"
        d_str = f"{s['d']:+.2f}" if not np.isnan(s['d']) else "n/a"
        pct   = s['mean_diff'] * (1 if hb else -1)
        pct_s = f"{pct*100:+.1f}%" if not np.isnan(pct) else "n/a"
        sig   = ("***" if s['p'] < 0.001 else
                 "**"  if s['p'] < 0.01  else
                 "*"   if s['p'] < 0.05  else
                 "ns")  if not np.isnan(s['p']) else "n/a"
        rows.append([lbl + (" ↑" if hb else " ↓"),
                     pct_s, p_str, d_str, sig])

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis("off")
    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Ours improvement", "p-value (paired t)",
                   "Cohen's d", "Significance"],
        cellLoc="center", loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 2.0)

    # Header style
    for j in range(5):
        tbl[(0, j)].set_facecolor("#404040")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")

    # Row colouring: green = our improvement, red = worse
    for i, row in enumerate(rows, start=1):
        pct_val = row[1].replace("%", "").replace("+", "")
        try:
            v = float(pct_val)
            bg = C_OURS + "33" if v > 0 else C_SAFE + "33"
        except Exception:
            bg = "white"
        for j in range(5):
            tbl[(i, j)].set_facecolor(bg)
        # Bold significance
        sig_cell = tbl[(i, 4)]
        if row[4] in ("*", "**", "***"):
            sig_cell.set_text_props(fontweight="bold", color="darkgreen")

    ax.set_title("Statistical Significance: Temporal Attention vs SAFE\n"
                 "(* p<0.05  ** p<0.01  *** p<0.001  |  Cohen's d: "
                 "small≥0.2, medium≥0.5, large≥0.8)",
                 fontsize=10, pad=15)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "09_stats_table.png"), bbox_inches="tight")
    plt.close(fig)
    print("  09_stats_table.png")
    return stat


# ═══════════════════════════════════════════════════════════════════════════════
#  Figure 10 — Dashboard (single-page summary)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_dashboard(seed_results, out_dir):
    from sklearn.metrics import roc_curve, roc_auc_score
    from compare_with_safe import episode_score, detection_time

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    # ── Panel A: ROC ──────────────────────────────────────────────────────────
    ax_roc = fig.add_subplot(gs[0, 0])
    base_fpr = np.linspace(0, 1, 200)
    ax_roc.plot([0,1],[0,1],"k:",lw=0.8)
    for name, col, lbl in (("safe",C_SAFE,"SAFE"), ("ours",C_OURS,"Ours")):
        tprs, aucs = [], []
        for sr in seed_results:
            ep_sc, ep_lbl = _episode_scores(sr[name]["curves"],
                                             sr["_val_r"], sr["_tms_all"])
            if len(np.unique(ep_lbl)) < 2: continue
            fpr, tpr, _ = roc_curve(ep_lbl, ep_sc)
            tprs.append(np.interp(base_fpr, fpr, tpr))
            aucs.append(roc_auc_score(ep_lbl, ep_sc))
        m_tpr = np.mean(tprs, 0)
        ax_roc.plot(base_fpr, m_tpr, color=col, lw=2,
                    label=f"{lbl} {np.mean(aucs):.3f}")
        ax_roc.fill_between(base_fpr, m_tpr-np.std(tprs,0),
                             m_tpr+np.std(tprs,0), alpha=0.12, color=col)
    ax_roc.legend(fontsize=8, loc="lower right")
    ax_roc.set_title("ROC", fontsize=10, fontweight="bold")
    ax_roc.set_xlabel("FPR"); ax_roc.set_ylabel("TPR")

    # ── Panel B: Metric bars ──────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[0, 1:3])
    metrics_show = ["auc", "ap", "bal_acc", "f1"]
    x = np.arange(len(metrics_show)); w = 0.35
    for i, (name, col, lbl) in enumerate((("safe",C_SAFE,"SAFE"),
                                           ("ours",C_OURS,"Ours"))):
        vals, errs = [], []
        for m in metrics_show:
            sv, ov = extract_per_seed(seed_results, m)
            data = sv if name == "safe" else ov
            vals.append(np.nanmean(data)); errs.append(np.nanstd(data))
        ax_bar.bar(x + (i-0.5)*w, vals, w, yerr=errs, capsize=3,
                   color=col, alpha=0.85, label=lbl)
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([METRIC_META[m][0] for m in metrics_show])
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_title("Key Metrics (mean ± std)", fontsize=10, fontweight="bold")
    ax_bar.legend(fontsize=8)

    # ── Panel C: Improvement bar ──────────────────────────────────────────────
    ax_imp = fig.add_subplot(gs[0, 3])
    all_m = list(METRIC_META.keys())
    pcts  = []
    for m in all_m:
        sv, ov = extract_per_seed(seed_results, m)
        _, hb  = METRIC_META[m]
        base   = np.nanmean(sv)
        diff   = np.nanmean(ov) - base
        pct    = (diff / (np.abs(base) + 1e-9)) * 100
        if not hb: pct = -pct
        pcts.append(pct)
    colors_imp = [C_OURS if p >= 0 else C_SAFE for p in pcts]
    ax_imp.barh(np.arange(len(all_m)), pcts, color=colors_imp, alpha=0.85)
    ax_imp.axvline(0, color="k", lw=1)
    ax_imp.set_yticks(np.arange(len(all_m)))
    ax_imp.set_yticklabels([METRIC_META[m][0] for m in all_m], fontsize=8)
    ax_imp.set_title("% Improvement", fontsize=10, fontweight="bold")
    ax_imp.set_xlabel("%")

    # ── Panel D: Detection CDF ────────────────────────────────────────────────
    ax_cdf = fig.add_subplot(gs[1, 0])
    for name, col, ls in (("safe",C_SAFE,"--"), ("ours",C_OURS,"-")):
        dts = [detection_time(sc, 0.5)
               for sr in seed_results
               for sc, r in zip(sr[name]["curves"], sr["_val_r"])
               if r.episode_success == 0
               and detection_time(sc, 0.5) is not None]
        if dts:
            xs = np.sort(dts)
            ax_cdf.plot(xs, np.arange(1,len(xs)+1)/len(xs),
                        color=col, lw=2, ls=ls)
    ax_cdf.set_title("Detection CDF", fontsize=10, fontweight="bold")
    ax_cdf.set_xlabel("Norm. detect time"); ax_cdf.set_ylabel("Fraction")
    ax_cdf.set_xlim(0,1); ax_cdf.set_ylim(0,1)

    # ── Panel E: Score trajectories (mean ± std) ──────────────────────────────
    for col_i, (name, col, lbl) in enumerate(
            [("safe", C_SAFE, "SAFE"), ("ours", C_OURS, "Ours")]):
        ax = fig.add_subplot(gs[1, 1 + col_i])
        grid = np.linspace(0, 1, 200)
        for outcome, oc in ((1, "#2166ac"), (0, "#d73027")):
            seqs = [np.interp(grid, np.linspace(0,1,len(sc)), sc)
                    for sr in seed_results
                    for sc, r in zip(sr[name]["curves"], sr["_val_r"])
                    if r.episode_success == outcome]
            if seqs:
                mat = np.vstack(seqs)
                m = mat.mean(0); s = mat.std(0)
                ax.plot(grid, m, color=oc, lw=2,
                        label="Success" if outcome else "Failure")
                ax.fill_between(grid, m-s, m+s, alpha=0.15, color=oc)
        ax.axhline(0.5, color="k", ls=":", lw=0.8)
        ax.set_xlim(0,1); ax.set_ylim(0,1)
        ax.set_title(f"{lbl} — Score curves", fontsize=10, fontweight="bold")
        ax.set_xlabel("Norm. timestep")
        ax.legend(fontsize=7)

    # ── Panel F: Summary numbers ──────────────────────────────────────────────
    ax_txt = fig.add_subplot(gs[1, 3])
    ax_txt.axis("off")
    lines = ["KEY RESULTS\n"]
    for m in ["auc", "ap", "avg_det", "far"]:
        sv, ov = extract_per_seed(seed_results, m)
        lbl, hb = METRIC_META[m]
        sm, om = np.nanmean(sv), np.nanmean(ov)
        delta = om - sm
        pct   = delta / (abs(sm) + 1e-9) * 100
        if not hb: pct = -pct
        sign  = "✓" if pct > 0 else "✗"
        lines.append(f"{sign} {lbl}: {om:.3f} vs {sm:.3f}  ({pct:+.1f}%)")
    ax_txt.text(0.05, 0.95, "\n".join(lines),
                transform=ax_txt.transAxes,
                va="top", ha="left", fontsize=9.5,
                fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0"))

    # legend patches
    patches = [
        mpatches.Patch(color=C_SAFE, label="SAFE IndepModel (official)"),
        mpatches.Patch(color=C_OURS, label="Temporal Attention + Hinge (ours)"),
    ]
    fig.legend(handles=patches, loc="lower center",
               ncol=2, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.01))

    fig.suptitle(
        "SAFE Official IndepModel  vs  Temporal Attention + Hinge (Ours)\n"
        "Evaluated on 30% held-out unseen tasks  |  3 random seeds",
        fontsize=13, fontweight="bold", y=1.01,
    )

    fig.savefig(os.path.join(out_dir, "10_dashboard.png"),
                bbox_inches="tight", dpi=150)
    plt.close(fig)
    print("  10_dashboard.png  ← main figure for slides / report")


# ═══════════════════════════════════════════════════════════════════════════════
#  Text report
# ═══════════════════════════════════════════════════════════════════════════════

def write_stats_report(seed_results, stat, out_dir):
    lines = [
        "=" * 70,
        "STATISTICAL ANALYSIS: SAFE vs Temporal Attention + Hinge",
        "=" * 70,
        f"  Seeds:   {len(seed_results)}",
        f"  Split:   30% unseen task IDs  (val_unseen)",
        f"  Tests:   Paired t-test (two-tailed), Cohen's d effect size",
        "",
        f"{'Metric':<22} {'SAFE':>8} {'Ours':>8} {'Δ':>8} "
        f"{'Δ%':>7} {'p':>7} {'d':>7} {'Sig':>5}",
        "-" * 70,
    ]

    all_wins = 0
    for m, (lbl, hb) in METRIC_META.items():
        sv, ov = extract_per_seed(seed_results, m)
        sm, om = np.nanmean(sv), np.nanmean(ov)
        s      = stat[m]
        delta  = om - sm
        pct    = delta / (abs(sm)+1e-9) * 100
        if not hb: pct = -pct
        sig    = ("***" if s['p'] < 0.001 else
                  "**"  if s['p'] < 0.01  else
                  "*"   if s['p'] < 0.05  else "ns") \
                 if not np.isnan(s['p']) else "n/a"
        win    = (om > sm) if hb else (om < sm)
        all_wins += int(win)
        marker = "✓" if win else "✗"
        lines.append(
            f"{marker} {lbl:<20} {sm:>8.4f} {om:>8.4f} {delta:>+8.4f} "
            f"{pct:>+6.1f}%  {s['p']:>6.3f}  {s['d']:>+6.2f}  {sig:>4}"
        )

    lines += [
        "-" * 70,
        f"\nOurs wins on {all_wins}/{len(METRIC_META)} metrics",
        "",
        "Interpretation of Cohen's d:",
        "  |d| < 0.2 = negligible   0.2–0.5 = small",
        "  0.5–0.8 = medium         > 0.8 = large",
        "",
        "Significance: * p<0.05  ** p<0.01  *** p<0.001  ns = not significant",
        "",
        "Note: Low p-values with only 3 seeds are indicative, not conclusive.",
        "      Run with --seeds 0 1 2 3 4 (5 seeds) for publishable claims.",
    ]

    # Per-seed breakdown
    lines += ["", "=" * 70, "PER-SEED BREAKDOWN", "=" * 70]
    for m, (lbl, _) in METRIC_META.items():
        sv, ov = extract_per_seed(seed_results, m)
        lines.append(f"  {lbl:<20}: SAFE {sv}  OURS {ov}")

    txt = "\n".join(lines)
    print(txt)
    path = os.path.join(out_dir, "stats_report.txt")
    with open(path, "w") as f:
        f.write(txt)
    print(f"\n  Full report → {path}")


def write_per_seed_csv(seed_results, out_dir):
    rows = []
    for seed_i, sr in enumerate(seed_results):
        for name in ("safe", "ours"):
            row = {"seed": seed_i, "model": name}
            row.update(sr[name]["metrics"])
            rows.append(row)
    path = os.path.join(out_dir, "per_seed_table.csv")
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(rows)
    print(f"  Per-seed CSV → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive analysis: SAFE vs our Temporal Attention model"
    )
    parser.add_argument("--results_dir", default="./compare_results",
                        help="Folder produced by compare_with_safe.py")
    parser.add_argument("--output_dir",  default=None,
                        help="Where to save figures (default: results_dir/analysis/)")

    # Re-run options
    parser.add_argument("--rerun",       action="store_true",
                        help="Re-train models before analysing")
    parser.add_argument("--data_path",   default=None)
    parser.add_argument("--seeds",       type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--n_epochs",    type=int, default=300)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--lambda_reg",  type=float, default=1e-2)
    parser.add_argument("--batch_size",  type=int, default=32)
    parser.add_argument("--device",      default="cuda" if
                        __import__("torch").cuda.is_available() else "cpu")

    parser.add_argument("--dashboard_only", action="store_true",
                        help="Generate only the dashboard figure (fast)")
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.results_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # ── Load or re-run ────────────────────────────────────────────────────────
    if args.rerun:
        if not args.data_path:
            parser.error("--data_path required with --rerun")
        print("Re-running comparison …")
        seed_results, _ = rerun_comparison(args)
    else:
        # Try to load saved pickled seed_results
        pkl = Path(args.results_dir) / "seed_results.pkl"
        if pkl.exists():
            import pickle
            with open(pkl, "rb") as f:
                seed_results = pickle.load(f)
            print(f"Loaded seed_results from {pkl}")
        else:
            print(f"WARNING: {pkl} not found.")
            print("Run compare_with_safe.py with --save_seed_results first, "
                  "or use --rerun --data_path <path>")
            return

    if not seed_results:
        print("No seed results to analyse.")
        return

    print(f"\nAnalysing {len(seed_results)} seed(s) …")
    print(f"Output → {out_dir}/\n")

    # ── Generate figures ──────────────────────────────────────────────────────
    if args.dashboard_only:
        fig_dashboard(seed_results, out_dir)
    else:
        fig_metric_bars(seed_results, out_dir)
        fig_improvement(seed_results, out_dir)
        fig_roc_prc(seed_results, out_dir)
        fig_score_violin(seed_results, out_dir)
        fig_detection_cdf(seed_results, out_dir)
        fig_trajectories(seed_results, out_dir)
        fig_per_task(seed_results, out_dir)
        fig_radar(seed_results, out_dir)
        stat = fig_stats_table(seed_results, out_dir)
        fig_dashboard(seed_results, out_dir)
        write_stats_report(seed_results, stat, out_dir)
        write_per_seed_csv(seed_results, out_dir)

    print(f"\nDone. All figures in {out_dir}/")


if __name__ == "__main__":
    main()
