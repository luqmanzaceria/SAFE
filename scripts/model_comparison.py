#!/usr/bin/env python3
"""
Multi-Model Multi-Seed Comparison
===================================

Trains every detector variant across N random seeds and reports mean ± std.
This is the script that generates paper-ready numbers.

Models evaluated
----------------
  base          Base MLP + running mean  (your re-implementation of SAFE's IndepModel)
  attn          MLP + causal attention   (your best model so far)
  lstm          LSTM + running mean      (SAFE's primary architecture)
  clip          MLP + CLIP task cond + attention   (requires transformers)
  combined      task(LSA) + attention + conformal  (your combined_detector)

For each model × seed:
  1. Apply seen/unseen task split  (--unseen_task_ratio)
  2. Train
  3. Score calibration and test sets
  4. Conformal calibrate threshold
  5. Record AUC, AP, Acc@0.5, Recall@τ, FAR@τ, Avg detection time

Aggregate across seeds → mean ± std table.

Outputs (output_dir/)
---------------------
  results_table.png     Styled mean±std table  (paper-ready)
  results_table.csv     Same as CSV for easy import
  roc_comparison.png    Mean ROC ± std band for each model
  detection_time.png    Mean detection-time curve ± std band
  training_loss.png     All training curves (all models, all seeds)
  summary.txt           Full numeric results

Usage
-----
    # Full run (3 seeds, all models, 300 epochs each)  — ~45–90 min on GPU
    python scripts/model_comparison.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/

    # Quick sanity check (1 seed, 100 epochs)
    python scripts/model_comparison.py \\
        --data_path ... --seeds 0 --n_epochs 100 --no_clip

    # Skip CLIP (faster, no transformers needed)
    python scripts/model_comparison.py \\
        --data_path ... --no_clip
"""

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(__file__))
from failure_detector import (
    load_rollouts, FailureDetector, train_model, predict,
)
from combined_detector import (
    CombinedFailureDetector, TaskEncoder,
    _train_combined, _predict_combined,
    calibrate_threshold, compute_detection_curve, _compute_model_metrics,
)
from lstm_detector import LSTMDetector, train_lstm, predict_lstm

import torch
import torch.nn as nn

try:
    from clip_task_detector import CLIPTaskEncoder, CLIPCondFailureDetector
    from clip_task_detector import _train_clip, _predict_clip
    _CLIP_OK = True
except Exception:
    _CLIP_OK = False


# ══════════════════════════════════════════════════════════════════════════════
#  Data splitting
# ══════════════════════════════════════════════════════════════════════════════

def make_split(all_r, all_emb, unseen_task_ratio, seed, train_frac=0.75):
    """
    Returns (train_r, train_emb, calib_r, calib_emb, test_r, test_emb,
             seen_ids, unseen_ids)
    all_emb may be None (for models with no task conditioning).
    """
    rng = np.random.RandomState(seed)
    all_task_ids = sorted(set(r.task_id for r in all_r))
    shuffled = all_task_ids.copy(); rng.shuffle(shuffled)
    n_unseen = max(1, round(unseen_task_ratio * len(all_task_ids)))
    unseen_ids = set(shuffled[:n_unseen])
    seen_ids   = set(shuffled[n_unseen:])

    seen_idx   = [i for i, r in enumerate(all_r) if r.task_id in seen_ids]
    unseen_idx = [i for i, r in enumerate(all_r) if r.task_id in unseen_ids]
    rng.shuffle(seen_idx)

    seen_s = [i for i in seen_idx if     all_r[i].episode_success]
    seen_f = [i for i in seen_idx if not all_r[i].episode_success]
    n_tr_s = max(1, int(len(seen_s) * train_frac))
    n_tr_f = max(1, int(len(seen_f) * train_frac))
    tr_idx = seen_s[:n_tr_s] + seen_f[:n_tr_f]
    ca_idx = seen_s[n_tr_s:] + seen_f[n_tr_f:]
    te_idx = list(unseen_idx)

    def _g(idx):
        rs  = [all_r[i] for i in idx]
        emb = all_emb[idx] if all_emb is not None else None
        return rs, emb

    (train_r, train_emb) = _g(tr_idx)
    (calib_r, calib_emb) = _g(ca_idx)
    (test_r,  test_emb)  = _g(te_idx)
    return (train_r, train_emb, calib_r, calib_emb,
            test_r,  test_emb,  seen_ids, unseen_ids)


# ══════════════════════════════════════════════════════════════════════════════
#  Per-model training + evaluation
# ══════════════════════════════════════════════════════════════════════════════

def _zero_emb(n): return np.zeros((n, 0), np.float32)


def run_one_seed(all_r, clip_emb, lsa_emb, input_dim, args, seed):
    """
    Trains all enabled models for one seed.
    Returns dict  model_name → metrics_dict
    """
    print(f"\n{'='*60}\n  Seed {seed}\n{'='*60}")
    alpha  = 1.0 - args.target_recall
    device = args.device

    # -- Split (re-split per seed for variance estimation) -------------------
    # base / attn / lstm use no task embedding
    (train_r, _, calib_r, _, test_r, _,
     seen_ids, unseen_ids) = make_split(all_r, None, args.unseen_task_ratio, seed)
    print(f"  Seen: {sorted(seen_ids)}  Unseen: {sorted(unseen_ids)}")

    def _cnt(r): ns = sum(x.episode_success for x in r); return f"{len(r)}({ns}s/{len(r)-ns}f)"
    print(f"  Train {_cnt(train_r)}  Calib {_cnt(calib_r)}  Test {_cnt(test_r)}")

    results = {}
    losses_log = {}   # name → list of epoch losses

    # ── Base MLP ─────────────────────────────────────────────────────────────
    if "base" in args.models:
        print(f"\n  [seed={seed}] Training base MLP ...")
        m = FailureDetector(input_dim, hidden_dim=args.hidden_dim,
                            n_layers=args.n_layers).to(device)
        ll = train_model(m, train_r, n_epochs=args.n_epochs, lr=args.lr,
                         lambda_reg=args.lambda_reg, batch_size=args.batch_size,
                         device=device)
        sc_ca = predict(m, calib_r, device=device)
        sc_te = predict(m, test_r,  device=device)
        tau   = calibrate_threshold(calib_r, sc_ca, alpha)
        results["base"] = _compute_model_metrics(test_r, sc_te, tau)
        losses_log["base"] = ll

    # ── Attn only ─────────────────────────────────────────────────────────────
    if "attn" in args.models:
        print(f"\n  [seed={seed}] Training +attention (no task) ...")
        m = CombinedFailureDetector(input_dim, task_embed_dim=0,
                                    hidden_dim=args.hidden_dim,
                                    n_layers=args.n_layers).to(device)
        ll = _train_combined(m, train_r, _zero_emb(len(train_r)),
                             n_epochs=args.n_epochs, lr=args.lr,
                             lambda_reg=args.lambda_reg, lambda_attn=args.lambda_attn,
                             batch_size=args.batch_size, device=device)
        sc_ca = _predict_combined(m, calib_r, _zero_emb(len(calib_r)), device=device)
        sc_te = _predict_combined(m, test_r,  _zero_emb(len(test_r)),  device=device)
        tau   = calibrate_threshold(calib_r, sc_ca, alpha)
        results["attn"] = _compute_model_metrics(test_r, sc_te, tau)
        losses_log["attn"] = ll

    # ── LSTM ──────────────────────────────────────────────────────────────────
    if "lstm" in args.models:
        print(f"\n  [seed={seed}] Training LSTM ...")
        m = LSTMDetector(input_dim, hidden_dim=args.hidden_dim,
                         n_layers=args.n_lstm_layers).to(device)
        ll = train_lstm(m, train_r, n_epochs=args.n_epochs, lr=args.lr,
                        lambda_reg=args.lambda_reg, batch_size=args.batch_size,
                        device=device)
        sc_ca = predict_lstm(m, calib_r, device=device)
        sc_te = predict_lstm(m, test_r,  device=device)
        tau   = calibrate_threshold(calib_r, sc_ca, alpha)
        results["lstm"] = _compute_model_metrics(test_r, sc_te, tau)
        losses_log["lstm"] = ll

    # ── CLIP + attention ──────────────────────────────────────────────────────
    if "clip" in args.models and _CLIP_OK and clip_emb is not None:
        (train_r2, tr_clip, calib_r2, ca_clip, test_r2, te_clip, _, _) = \
            make_split(all_r, clip_emb, args.unseen_task_ratio, seed)
        print(f"\n  [seed={seed}] Training CLIP+attention ...")
        proj_dim = args.clip_proj_dim
        m = CLIPCondFailureDetector(
            hidden_state_dim=input_dim, clip_embed_dim=512,
            proj_dim=proj_dim, hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        ).to(device)
        ll = _train_clip(m, train_r2, tr_clip, n_epochs=args.n_epochs,
                         lr=args.lr, lambda_reg=args.lambda_reg,
                         lambda_attn=args.lambda_attn,
                         batch_size=args.batch_size, device=device)
        sc_ca = _predict_clip(m, calib_r2, ca_clip, device=device)
        sc_te = _predict_clip(m, test_r2,  te_clip, device=device)
        tau   = calibrate_threshold(calib_r2, sc_ca, alpha)
        results["clip"] = _compute_model_metrics(test_r2, sc_te, tau)
        losses_log["clip"] = ll

    # ── Combined (LSA + attn + conformal) ────────────────────────────────────
    if "combined" in args.models and lsa_emb is not None:
        (train_r3, tr_lsa, calib_r3, ca_lsa, test_r3, te_lsa, _, _) = \
            make_split(all_r, lsa_emb, args.unseen_task_ratio, seed)
        E = tr_lsa.shape[-1]
        print(f"\n  [seed={seed}] Training combined (LSA+attn) ...")
        m = CombinedFailureDetector(
            hidden_state_dim=input_dim, task_embed_dim=E,
            hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        ).to(device)
        ll = _train_combined(m, train_r3, tr_lsa, n_epochs=args.n_epochs,
                             lr=args.lr, lambda_reg=args.lambda_reg,
                             lambda_attn=args.lambda_attn,
                             batch_size=args.batch_size, device=device)
        sc_ca = _predict_combined(m, calib_r3, ca_lsa, device=device)
        sc_te = _predict_combined(m, test_r3,  te_lsa, device=device)
        tau   = calibrate_threshold(calib_r3, sc_ca, alpha)
        results["combined"] = _compute_model_metrics(test_r3, sc_te, tau)
        losses_log["combined"] = ll

    return results, losses_log


# ══════════════════════════════════════════════════════════════════════════════
#  Aggregation
# ══════════════════════════════════════════════════════════════════════════════

def aggregate(all_seed_results):
    """
    all_seed_results: list of dicts  {model_name: metrics_dict}
    Returns: {model_name: {metric: (mean, std)}}
    """
    model_names = list(all_seed_results[0].keys())
    metric_keys = list(list(all_seed_results[0].values())[0].keys())
    agg = {}
    for name in model_names:
        agg[name] = {}
        for key in metric_keys:
            vals = [r[name][key] for r in all_seed_results
                    if r[name][key] is not None]
            if vals:
                agg[name][key] = (float(np.mean(vals)), float(np.std(vals)))
            else:
                agg[name][key] = (None, None)
    return agg


# ══════════════════════════════════════════════════════════════════════════════
#  Plots
# ══════════════════════════════════════════════════════════════════════════════

MODEL_COLORS = {
    "base":     "steelblue",
    "attn":     "darkorange",
    "lstm":     "mediumseagreen",
    "clip":     "crimson",
    "combined": "mediumpurple",
}
MODEL_LABELS = {
    "base":     "Base MLP",
    "attn":     "+Attention",
    "lstm":     "LSTM",
    "clip":     "CLIP+Attn",
    "combined": "Combined (LSA+Attn)",
}


def plot_results_table(agg, n_seeds, out_dir):
    """Styled table with mean ± std for each model × metric."""
    metrics = ["auc", "ap", "acc", "recall", "far", "avg_det"]
    headers = ["Model", "AUC ↑", "AP ↑", "Acc@0.5 ↑", "Recall@τ ↑",
               "FAR@τ ↓", "Det.Time ↓"]
    rows = []
    for name, ms in agg.items():
        row = [MODEL_LABELS.get(name, name)]
        for k in metrics:
            mu, sd = ms[k]
            if mu is None:
                row.append("—")
            elif n_seeds > 1:
                row.append(f"{mu:.3f}±{sd:.3f}")
            else:
                row.append(f"{mu:.4f}")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(15, 1.8 + 0.9 * len(rows)))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=headers,
                   loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.6)

    for j in range(len(headers)):
        tbl[0, j].set_facecolor("#1a3a5c")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        bg = "#f0f5fb" if i % 2 == 0 else "white"
        col = MODEL_COLORS.get(list(agg.keys())[i-1], "white")
        tbl[i, 0].set_facecolor(col); tbl[i, 0].set_text_props(color="white", fontweight="bold")
        for j in range(1, len(headers)):
            tbl[i, j].set_facecolor(bg)

    seeds_str = f"  (mean ± std across {n_seeds} seeds)" if n_seeds > 1 else "  (single seed)"
    ax.set_title(f"Model Comparison — Unseen Task Evaluation{seeds_str}",
                 fontsize=12, fontweight="bold", pad=14, y=0.98)
    fig.tight_layout()
    p = os.path.join(out_dir, "results_table.png")
    fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); print(f"  -> {p}")


def plot_metric_bars(agg, n_seeds, out_dir):
    """Bar chart with error bars for key metrics."""
    metrics   = ["auc", "ap", "recall", "avg_det"]
    titles    = ["ROC AUC ↑", "Avg Precision ↑", "Recall@τ ↑", "Det. Time ↓"]
    names     = list(agg.keys())
    x         = np.arange(len(names))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, key, title in zip(axes, metrics, titles):
        means = [agg[n][key][0] if agg[n][key][0] is not None else 0 for n in names]
        stds  = [agg[n][key][1] if agg[n][key][1] is not None else 0 for n in names]
        colors = [MODEL_COLORS.get(n, "gray") for n in names]
        bars = ax.bar(x, means, color=colors, alpha=0.85,
                      yerr=stds if n_seeds > 1 else None,
                      capsize=4, error_kw={"elinewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(n, n) for n in names],
                           rotation=30, ha="right", fontsize=8)
        ax.set_title(title); ax.grid(True, alpha=0.3, axis="y")
        # annotate values
        for bar, mu in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{mu:.3f}", ha="center", va="bottom", fontsize=7)

    seeds_str = f"mean±std, {n_seeds} seeds" if n_seeds > 1 else "single seed"
    fig.suptitle(f"Model Comparison ({seeds_str}, unseen tasks)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "metric_bars.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def plot_training_losses(all_losses_by_seed, out_dir):
    """All training curves for all models × seeds."""
    # all_losses_by_seed: list of {model_name: [epoch_losses]}
    model_names = list(all_losses_by_seed[0].keys())
    fig, axes = plt.subplots(1, len(model_names),
                             figsize=(5 * len(model_names), 3), squeeze=False)
    for ax, name in zip(axes[0], model_names):
        col = MODEL_COLORS.get(name, "gray")
        all_curves = [d[name] for d in all_losses_by_seed if name in d]
        max_len = max(len(c) for c in all_curves)
        padded  = np.array([np.pad(c, (0, max_len - len(c)), mode='edge')
                            for c in all_curves])
        mean, std = padded.mean(0), padded.std(0)
        x = np.arange(max_len)
        ax.plot(x, mean, lw=2, color=col, label=MODEL_LABELS.get(name, name))
        ax.fill_between(x, mean-std, mean+std, alpha=0.25, color=col)
        ax.set_title(MODEL_LABELS.get(name, name)); ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3); ax.legend(fontsize=7)
    fig.suptitle("Training Loss (mean ± std across seeds)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(out_dir, "training_loss.png")
    fig.savefig(p, dpi=150); plt.close(fig); print(f"  -> {p}")


def save_csv(agg, n_seeds, out_dir):
    metrics = ["auc", "ap", "acc", "recall", "far", "avg_det"]
    path = os.path.join(out_dir, "results_table.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["model"] + [f"{m}_mean" for m in metrics] + \
                 ([f"{m}_std" for m in metrics] if n_seeds > 1 else [])
        writer.writerow(header)
        for name, ms in agg.items():
            row = [MODEL_LABELS.get(name, name)]
            row += [f"{ms[k][0]:.4f}" if ms[k][0] is not None else "" for k in metrics]
            if n_seeds > 1:
                row += [f"{ms[k][1]:.4f}" if ms[k][1] is not None else "" for k in metrics]
            writer.writerow(row)
    print(f"  -> {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed multi-model comparison (paper-ready results)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",         required=True)
    parser.add_argument("--output_dir",        default="./comparison_results")
    parser.add_argument("--seeds",             nargs="+", type=int, default=[0, 1, 2],
                        help="Seeds to run. E.g. --seeds 0 1 2")
    parser.add_argument("--models",            nargs="+",
                        default=["base", "attn", "lstm", "clip", "combined"],
                        choices=["base", "attn", "lstm", "clip", "combined"],
                        help="Models to train")
    parser.add_argument("--unseen_task_ratio", type=float, default=0.30)
    parser.add_argument("--target_recall",     type=float, default=0.90)
    parser.add_argument("--n_epochs",          type=int,   default=300,
                        help="Use --n_epochs 100 for a quick sanity check")
    parser.add_argument("--lr",                type=float, default=1e-3)
    parser.add_argument("--lambda_reg",        type=float, default=1e-2)
    parser.add_argument("--lambda_attn",       type=float, default=0.1)
    parser.add_argument("--hidden_dim",        type=int,   default=256)
    parser.add_argument("--n_layers",          type=int,   default=2)
    parser.add_argument("--n_lstm_layers",     type=int,   default=1)
    parser.add_argument("--batch_size",        type=int,   default=32)
    parser.add_argument("--lsa_dim",           type=int,   default=32)
    parser.add_argument("--clip_proj_dim",     type=int,   default=64)
    parser.add_argument("--no_clip",           action="store_true",
                        help="Skip CLIP model even if transformers is installed")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.no_clip and "clip" in args.models:
        args.models = [m for m in args.models if m != "clip"]

    if "clip" in args.models and not _CLIP_OK:
        print("WARNING: clip model requested but clip_task_detector import failed. "
              "Skipping CLIP.")
        args.models = [m for m in args.models if m != "clip"]

    os.makedirs(args.output_dir, exist_ok=True)
    n_seeds = len(args.seeds)

    print(f"\nModels:  {args.models}")
    print(f"Seeds:   {args.seeds}")
    print(f"Epochs:  {args.n_epochs}  Device: {args.device}")

    # ── Pre-compute task embeddings once ─────────────────────────────────────
    print("\n[Pre] Loading rollouts and pre-computing task embeddings ...")
    all_r     = load_rollouts(args.data_path)
    input_dim = all_r[0].hidden_states.shape[-1]

    # LSA
    lsa_emb = None
    if "combined" in args.models:
        enc = TaskEncoder(n_components=args.lsa_dim)
        enc.fit(list(dict.fromkeys(r.task_description for r in all_r)))
        lsa_emb = enc.transform([r.task_description for r in all_r])
        print(f"  LSA: {lsa_emb.shape[1]}-d")

    # CLIP
    clip_emb = None
    if "clip" in args.models and _CLIP_OK:
        print("  Loading CLIP encoder ...")
        clip_enc = CLIPTaskEncoder(device=args.device)
        clip_emb = clip_enc.encode([r.task_description for r in all_r])
        print(f"  CLIP: {clip_emb.shape[1]}-d")

    # ── Run all seeds ─────────────────────────────────────────────────────────
    all_results = []
    all_losses  = []
    for seed in args.seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        res, losses = run_one_seed(all_r, clip_emb, lsa_emb,
                                   input_dim, args, seed)
        all_results.append(res)
        all_losses.append(losses)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    print("\n\nAggregating results ...")
    agg = aggregate(all_results)

    # Print summary
    print("\n" + "=" * 70)
    print(f"MULTI-SEED RESULTS  ({n_seeds} seed{'s' if n_seeds>1 else ''})")
    print("=" * 70)
    for name, ms in agg.items():
        tag = MODEL_LABELS.get(name, name)
        mu_auc, sd_auc = ms["auc"]
        mu_ap,  sd_ap  = ms["ap"]
        mu_rec, sd_rec = ms["recall"]
        mu_det, sd_det = ms["avg_det"]
        if n_seeds > 1:
            print(f"  {tag:30s}  AUC={mu_auc:.3f}±{sd_auc:.3f}  "
                  f"AP={mu_ap:.3f}±{sd_ap:.3f}  "
                  f"Recall={mu_rec:.3f}±{sd_rec:.3f}  "
                  f"Det={mu_det:.3f}±{sd_det:.3f}")
        else:
            print(f"  {tag:30s}  AUC={mu_auc:.4f}  AP={mu_ap:.4f}  "
                  f"Recall={mu_rec:.4f}  Det={mu_det:.4f}")
    print("=" * 70)

    # ── Save results ─────────────────────────────────────────────────────────
    lines = []
    for name, ms in agg.items():
        lines.append(f"\n─── {MODEL_LABELS.get(name, name)} ───")
        for k, (mu, sd) in ms.items():
            if mu is None: continue
            lines.append(f"  {k:12s}: {mu:.4f}" +
                         (f" ± {sd:.4f}" if n_seeds > 1 else ""))
    with open(os.path.join(args.output_dir, "summary.txt"), "w") as f:
        f.write("\n".join(lines))

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_results_table(agg, n_seeds, args.output_dir)
    plot_metric_bars(agg, n_seeds, args.output_dir)
    plot_training_losses(all_losses, args.output_dir)
    save_csv(agg, n_seeds, args.output_dir)

    print(f"\n  All outputs saved to: {os.path.abspath(args.output_dir)}/")
    print(f"\nTo cite in your report:")
    print(f"  Table X: 'All models evaluated on {int(args.unseen_task_ratio*100)}% "
          f"held-out unseen tasks ({n_seeds} seed(s)).'")


if __name__ == "__main__":
    main()
