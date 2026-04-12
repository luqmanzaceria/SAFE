#!/usr/bin/env python3
"""
Conformal Failure Detector
==========================

Adds *statistical coverage guarantees* on top of the trained MLP failure
detector via Inductive Conformal Prediction (ICP).

Standard thresholding (threshold = 0.5) gives no formal guarantee about how
many real failures you'll actually catch.  Conformal prediction lets you set
a *target recall* (e.g. "catch at least 90% of failures") and derives the
exact threshold from held-out calibration data — with a provable bound:

    P( score >= τ_α  |  episode is a failure ) ≥ 1 − α

where α is your chosen miss-rate budget and τ_α is the calibrated threshold.

The script:
  1. Loads rollouts and splits them into train / calibrate / test
     (default 60% / 20% / 20%, stratified by outcome).
  2. Trains the base MLP detector on the train split.
  3. Computes nonconformity scores on the calibration split to find τ_α.
  4. Evaluates on the test split and reports:
       - Empirical coverage (should be ≥ 1−α)
       - False-alarm rate at τ_α
       - Comparison with fixed threshold = 0.5
  5. Generates coverage_curve.png  (coverage & false-alarm vs α)
     and conformal_histogram.png   (score distribution with both thresholds)

Usage
-----
    python scripts/conformal_detector.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
        --target_recall 0.90

    # or sweep several targets at once
    python scripts/conformal_detector.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
        --target_recalls 0.80 0.85 0.90 0.95
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Pull shared utilities from failure_detector.py (same directory)
sys.path.insert(0, os.path.dirname(__file__))
from failure_detector import (
    load_rollouts,
    FailureDetector,
    train_model,
    predict,
)

import torch
from sklearn.metrics import roc_auc_score


# ──────────────────────────── Conformal calibration ────────────────────────

def calibrate_threshold(
    calibration_rollouts: list,
    calibration_scores: list,
    alpha: float,
) -> float:
    """
    Compute the (1-α)-quantile threshold from calibration *failure* episodes.

    Nonconformity score for failure episodes = 1 - final_score.
    A high nonconformity means the episode looks *less* like a failure
    according to the detector.

    The threshold τ is chosen so that ≥ (1-α) fraction of calibration
    failures have final_score ≥ τ.

    α = 0.10 → catch ≥ 90% of calibration failures → formal test guarantee.
    """
    failure_scores = [
        s[-1]
        for r, s in zip(calibration_rollouts, calibration_scores)
        if not r.episode_success
    ]
    if not failure_scores:
        raise ValueError("No failure episodes in calibration set — "
                         "increase --calib_ratio or collect more failures.")

    n   = len(failure_scores)
    # Finite-sample correction: use ceil((1-α)(n+1))/n quantile
    q_idx = int(np.ceil((1 - alpha) * (n + 1))) - 1
    q_idx = max(0, min(n - 1, q_idx))
    tau   = float(np.sort(failure_scores)[q_idx])
    return tau


def evaluate_at_threshold(rollouts, scores, tau: float):
    """Returns (recall, false_alarm_rate, accuracy) at a given threshold."""
    y_true  = np.array([1 - r.episode_success for r in rollouts])  # 1=fail
    y_score = np.array([s[-1] for s in scores])
    y_pred  = (y_score >= tau).astype(int)

    n_fail = y_true.sum()
    n_succ = (1 - y_true).sum()

    recall    = (y_pred[y_true == 1] == 1).sum() / max(n_fail, 1)
    far       = (y_pred[y_true == 0] == 1).sum() / max(n_succ, 1)
    accuracy  = (y_pred == y_true).mean()
    return float(recall), float(far), float(accuracy)


# ──────────────────────────── Visualisations ───────────────────────────────

def plot_coverage_curve(
    calib_rollouts, calib_scores,
    test_rollouts,  test_scores,
    alphas: np.ndarray,
    out_dir: str,
):
    """
    For each α in alphas, compute τ_α from calibration and measure empirical
    recall / false-alarm-rate on the test set.
    Plot the resulting curves.
    """
    calib_recalls, calib_fars = [], []
    test_recalls,  test_fars  = [], []
    taus = []

    for alpha in alphas:
        try:
            tau = calibrate_threshold(calib_rollouts, calib_scores, alpha)
        except ValueError:
            taus.append(np.nan); calib_recalls.append(np.nan)
            calib_fars.append(np.nan); test_recalls.append(np.nan)
            test_fars.append(np.nan)
            continue

        taus.append(tau)
        cr, cf, _ = evaluate_at_threshold(calib_rollouts, calib_scores, tau)
        tr, tf, _ = evaluate_at_threshold(test_rollouts,  test_scores,  tau)
        calib_recalls.append(cr); calib_fars.append(cf)
        test_recalls.append(tr);  test_fars.append(tf)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    target_recalls = 1 - alphas

    # Panel 1: empirical recall vs target recall
    ax = axes[0]
    ax.plot(target_recalls, calib_recalls, "b-o", ms=4, label="Calibration")
    ax.plot(target_recalls, test_recalls,  "r-o", ms=4, label="Test")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect")
    ax.set_xlabel("Target recall  (1 − α)")
    ax.set_ylabel("Empirical recall")
    ax.set_title("Coverage: Target vs Empirical")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 2: false-alarm rate vs target recall
    ax = axes[1]
    ax.plot(target_recalls, calib_fars, "b-o", ms=4, label="Calibration FAR")
    ax.plot(target_recalls, test_fars,  "r-o", ms=4, label="Test FAR")
    ax.set_xlabel("Target recall  (1 − α)")
    ax.set_ylabel("False-alarm rate")
    ax.set_title("False-alarm Rate vs Target Recall")
    ax.legend(); ax.grid(True, alpha=0.3)

    # Panel 3: calibrated threshold τ_α
    ax = axes[2]
    ax.plot(target_recalls, taus, "g-o", ms=4)
    ax.axhline(0.5, color="orange", linestyle="--", lw=1, label="Fixed τ=0.5")
    ax.set_xlabel("Target recall  (1 − α)")
    ax.set_ylabel("Calibrated threshold τ_α")
    ax.set_title("Calibrated Threshold vs Target Recall")
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "coverage_curve.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")


def plot_conformal_histogram(
    test_rollouts, test_scores,
    tau_conformal: float,
    out_dir: str,
):
    """
    Score histogram with both the fixed (0.5) and conformal threshold marked.
    """
    succ = [s[-1] for r, s in zip(test_rollouts, test_scores) if     r.episode_success]
    fail = [s[-1] for r, s in zip(test_rollouts, test_scores) if not r.episode_success]

    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0, 1, 30)
    ax_r = ax.twinx()

    if succ:
        ax.hist(succ,  bins=bins, color="seagreen", alpha=0.6,
                label=f"Success (n={len(succ)})", density=True)
    if fail:
        ax_r.hist(fail, bins=bins, color="crimson", alpha=0.5,
                  label=f"Failure (n={len(fail)})", density=True)

    ax.axvline(0.5,           color="orange", linestyle="--",  lw=2,
               label="Fixed τ=0.5")
    ax.axvline(tau_conformal, color="purple",  linestyle="-.", lw=2,
               label=f"Conformal τ={tau_conformal:.3f}")

    ax.set_xlabel("Final failure score"); ax.set_ylabel("Density (success)", color="seagreen")
    ax_r.set_ylabel("Density (failure)", color="crimson")
    ax.tick_params(axis="y", labelcolor="seagreen")
    ax_r.tick_params(axis="y", labelcolor="crimson")
    ax.set_title("Score Distribution with Thresholds")
    lines1, lbl1 = ax.get_legend_handles_labels()
    lines2, lbl2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lbl1 + lbl2, fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "conformal_histogram.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Conformal failure detector with coverage guarantee",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",      required=True)
    parser.add_argument("--output_dir",     default="./conformal_results")
    parser.add_argument("--train_ratio",    type=float, default=0.60)
    parser.add_argument("--calib_ratio",    type=float, default=0.20,
                        help="Fraction for conformal calibration (rest = test)")
    parser.add_argument("--target_recall",  type=float, default=0.90,
                        help="Primary target recall for conformal threshold")
    parser.add_argument("--target_recalls", type=float, nargs="+", default=None,
                        help="Sweep of target recalls for the coverage curve "
                             "(overrides --target_recall for the curve; "
                             "primary threshold still uses --target_recall)")
    parser.add_argument("--n_epochs",   type=int,   default=300)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--lambda_reg", type=float, default=1e-2)
    parser.add_argument("--hidden_dim", type=int,   default=256)
    parser.add_argument("--n_layers",   type=int,   default=2)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load & 3-way split ───────────────────────────────────────────────
    print("[1/5] Loading rollouts ...")
    all_rollouts = load_rollouts(args.data_path)

    rng      = np.random.RandomState(args.seed)
    succ_idx = [i for i, r in enumerate(all_rollouts) if     r.episode_success]
    fail_idx = [i for i, r in enumerate(all_rollouts) if not r.episode_success]
    rng.shuffle(succ_idx); rng.shuffle(fail_idx)

    def _split3(idx, r_train, r_calib):
        n  = len(idx)
        n_tr = max(1, int(n * r_train))
        n_ca = max(1, int(n * r_calib))
        return idx[:n_tr], idx[n_tr:n_tr+n_ca], idx[n_tr+n_ca:]

    s_tr, s_ca, s_te = _split3(succ_idx, args.train_ratio, args.calib_ratio)
    f_tr, f_ca, f_te = _split3(fail_idx, args.train_ratio, args.calib_ratio)

    train_r = [all_rollouts[i] for i in s_tr + f_tr]
    calib_r = [all_rollouts[i] for i in s_ca + f_ca]
    test_r  = [all_rollouts[i] for i in s_te + f_te]

    def _counts(rollouts):
        ns = sum(r.episode_success for r in rollouts)
        return f"{len(rollouts)} ({ns} succ / {len(rollouts)-ns} fail)"

    print(f"  Train: {_counts(train_r)}")
    print(f"  Calib: {_counts(calib_r)}")
    print(f"  Test:  {_counts(test_r)}")

    # ── 2. Train ────────────────────────────────────────────────────────────
    print("\n[2/5] Training base MLP detector ...")
    input_dim = all_rollouts[0].hidden_states.shape[-1]
    model = FailureDetector(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(args.device)
    train_model(model, train_r, n_epochs=args.n_epochs, lr=args.lr,
                lambda_reg=args.lambda_reg, batch_size=args.batch_size,
                device=args.device)

    # ── 3. Score all splits ─────────────────────────────────────────────────
    print("\n[3/5] Scoring ...")
    calib_scores = predict(model, calib_r, device=args.device)
    test_scores  = predict(model, test_r,  device=args.device)

    # ── 4. Calibrate ────────────────────────────────────────────────────────
    print("\n[4/5] Calibrating conformal threshold ...")
    alpha      = 1.0 - args.target_recall
    tau        = calibrate_threshold(calib_r, calib_scores, alpha)
    print(f"  Target recall:      {args.target_recall:.0%}")
    print(f"  Calibrated τ_α:     {tau:.4f}  (fixed τ=0.50)")

    calib_rec, calib_far, calib_acc = evaluate_at_threshold(calib_r, calib_scores, tau)
    test_rec,  test_far,  test_acc  = evaluate_at_threshold(test_r,  test_scores,  tau)
    _, _, fixed_acc = evaluate_at_threshold(test_r, test_scores, 0.5)

    print(f"\n  ─── Conformal threshold τ={tau:.4f} ───")
    print(f"  Calib recall (coverage): {calib_rec:.1%}  "
          f"(target ≥ {args.target_recall:.0%})")
    print(f"  Test  recall (coverage): {test_rec:.1%}")
    print(f"  Test  false-alarm rate:  {test_far:.1%}")
    print(f"  Test  accuracy:          {test_acc:.1%}")
    print(f"\n  ─── Fixed threshold τ=0.5 ───")
    print(f"  Test  accuracy:          {fixed_acc:.1%}")

    y_true  = np.array([1 - r.episode_success for r in test_r])
    y_score = np.array([s[-1] for s in test_scores])
    if len(np.unique(y_true)) > 1:
        print(f"  Test  AUC:               {roc_auc_score(y_true, y_score):.4f}")

    # ── 5. Plots ────────────────────────────────────────────────────────────
    print("\n[5/5] Generating plots ...")
    alphas_sweep = 1.0 - np.array(
        args.target_recalls if args.target_recalls
        else np.linspace(0.70, 0.99, 30)
    )
    plot_coverage_curve(calib_r, calib_scores, test_r, test_scores,
                        alphas_sweep, args.output_dir)
    plot_conformal_histogram(test_r, test_scores, tau, args.output_dir)

    print(f"\n  Outputs saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
