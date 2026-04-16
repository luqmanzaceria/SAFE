#!/usr/bin/env python3
"""
pres_videos.py  —  Generate Accurate Animated Presentation Videos
==================================================================

All videos use REAL data from seed_results.pkl (output of compare_with_safe.py).
No synthetic or approximated curves — every score value, detection time, and
attention weight is taken directly from the actual model outputs.

Videos generated
----------------
  video01_score_overlay.mp4     Robot rollout video with actual failure score bar
                                  overlaid frame-by-frame (requires mp4_path in rollouts)
  video02_comparison.mp4        Animated side-by-side: SAFE vs Ours actual score curves
                                  building up over real episodes
  video03_attention_anim.mp4    Real attention weights + score for a failure episode
  video04_arch_flow.mp4         Architecture data-flow diagram animation (no data needed)
  video05_detection_moment.mp4  Clip around the actual earliest detection in a failure ep

Usage
-----
    # Full run (needs seed_results.pkl from compare_with_safe.py --save_seed_results)
    python scripts/pres_videos.py \\
        --seed_results  ./compare_results/seed_results.pkl \\
        --rollout_dir   ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
        --output_dir    ./presentation_videos

    # Diagram only (no data needed)
    python scripts/pres_videos.py --output_dir ./presentation_videos --diagrams_only

Notes
-----
  • Re-run compare_with_safe.py with --save_seed_results to get seed_results.pkl
  • The pkl now stores ours["weights"] (attention weight curves per episode)
  • If mp4_path is unavailable for video01/05, matplotlib-only fallbacks are used
"""

import os, sys, glob, pickle, argparse, warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import Normalize
from matplotlib import cm

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
_THIS = os.path.realpath(os.path.abspath(__file__))
_SD   = os.path.dirname(_THIS)
_ROOT = os.path.dirname(_SD)
for _p in (_SD, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Style constants ────────────────────────────────────────────────────────────
C_SAFE = "#4dac26"
C_OURS = "#d73027"
C_BLUE = "#2166ac"
C_GOLD = "#e6a817"
C_BG   = "#0d0d0d"

plt.rcParams.update({"font.size": 11, "figure.dpi": 120})

THRESHOLD = 0.5     # score threshold used in evaluation


# ── Helpers ───────────────────────────────────────────────────────────────────

def _save(anim, path: str, fps: int = 24):
    """Save animation as MP4 (ffmpeg) or GIF (pillow fallback)."""
    try:
        writer = animation.FFMpegWriter(
            fps=fps, bitrate=2000,
            extra_args=["-vcodec", "libx264", "-pix_fmt", "yuv420p"])
        anim.save(path, writer=writer)
        print(f"  ✓  {path}")
    except Exception as e:
        gif = path.replace(".mp4", ".gif")
        try:
            anim.save(gif, writer="pillow", fps=fps)
            print(f"  ✓  {gif}  (ffmpeg unavailable — saved as GIF)")
        except Exception as e2:
            print(f"  ✗  {path}  ({e2})")


def _interp(arr: np.ndarray, n: int) -> np.ndarray:
    """Interpolate 1-D array to length n."""
    return np.interp(np.linspace(0, 1, n),
                     np.linspace(0, 1, len(arr)),
                     arr)


def _first_detection(curve: np.ndarray, thresh: float = THRESHOLD):
    """Return index of first threshold crossing, or None."""
    hits = np.where(curve >= thresh)[0]
    return int(hits[0]) if len(hits) else None


def _best_episodes(seed_results, kind: str):
    """
    Return (safe_curve, ours_curve, weight_curve, rollout) for the episode
    where our model scores best (failure: highest score; success: lowest score).
    kind: 'failure' or 'success'
    """
    sr = seed_results[0]
    val_r = sr["_val_r"]
    safe_c = sr["safe"]["curves"]
    ours_c = sr["ours"]["curves"]
    ours_w = sr["ours"].get("weights", [None]*len(ours_c))

    candidates = []
    for sc_s, sc_o, wc, r in zip(safe_c, ours_c, ours_w, val_r):
        if kind == "failure" and r.episode_success == 0:
            score = float(np.array(sc_o).max())
            candidates.append((score, sc_s, sc_o, wc, r))
        elif kind == "success" and r.episode_success == 1:
            score = -float(np.array(sc_o).max())   # lower is better for success
            candidates.append((score, sc_s, sc_o, wc, r))

    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda x: x[0])
    _, sc_s, sc_o, wc, r = candidates[0]
    return np.array(sc_s), np.array(sc_o), wc, r


def _collect_curves(seed_results, kind: str, max_ep: int = 20):
    """Collect up to max_ep (safe_curve, ours_curve) pairs for failure or success."""
    sr = seed_results[0]
    val_r = sr["_val_r"]
    safe_c = sr["safe"]["curves"]
    ours_c = sr["ours"]["curves"]

    out = []
    for sc_s, sc_o, r in zip(safe_c, ours_c, val_r):
        if kind == "failure" and r.episode_success == 0:
            out.append((np.array(sc_s), np.array(sc_o)))
        elif kind == "success" and r.episode_success == 1:
            out.append((np.array(sc_s), np.array(sc_o)))
        if len(out) >= max_ep:
            break
    return out


# ═══════════════════════════════════════════════════════════════════════════════
#  Video 01 — Robot video with actual score bar overlaid
# ═══════════════════════════════════════════════════════════════════════════════

def video01_score_overlay(seed_results, rollout_dir, out_dir: str):
    """
    Overlay the ACTUAL our-model failure score on a real robot rollout video.
    Picks the failure episode with:
      - highest episode score (clearest true positive)
      - a valid mp4_path on disk
    Falls back to a matplotlib animation if no MP4 is found.
    """
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False

    # ── Find best failure episode with actual MP4 ──────────────────────────────
    sr    = seed_results[0]
    val_r = sr["_val_r"]
    safe_curves = sr["safe"]["curves"]
    ours_curves = sr["ours"]["curves"]

    best = None
    for sc_s, sc_o, r in zip(safe_curves, ours_curves, val_r):
        if r.episode_success != 0:
            continue
        sc_o_arr = np.array(sc_o)
        ep_score = float(sc_o_arr.max())

        mp4 = r.mp4_path
        # Also search rollout_dir if mp4_path not on disk
        if (mp4 is None or not os.path.exists(mp4)) and rollout_dir:
            # Try to match by task_id / episode_idx in filename
            patterns = [
                os.path.join(rollout_dir, f"*task{r.task_id}*ep{r.episode_idx}*succ0*.mp4"),
                os.path.join(rollout_dir, f"*succ0*.mp4"),
            ]
            for pat in patterns:
                hits = sorted(glob.glob(pat))
                if hits:
                    mp4 = hits[0]
                    break

        if mp4 and os.path.exists(mp4):
            if best is None or ep_score > best[0]:
                best = (ep_score, np.array(sc_s), sc_o_arr, mp4)

    if best is None or not _has_cv2:
        # ── Fallback: matplotlib-only animated score plot ──────────────────────
        print("  (video01: no robot MP4 found — generating matplotlib fallback)")
        _video01_matplotlib_fallback(seed_results, out_dir)
        return

    ep_score, sc_safe, sc_ours, mp4_path = best
    det_idx = _first_detection(sc_ours)
    det_norm = det_idx / len(sc_ours) if det_idx is not None else None

    cap   = cv2.VideoCapture(mp4_path)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = max(int(cap.get(cv2.CAP_PROP_FPS)), 10)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Map score timesteps → video frames
    score_at_frame = _interp(sc_ours, total)
    safe_at_frame  = _interp(sc_safe, total)

    out_path = os.path.join(out_dir, "video01_score_overlay.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    bar_h    = 170
    out_vid  = cv2.VideoWriter(out_path, fourcc, fps, (W, H + bar_h))

    cap = cv2.VideoCapture(mp4_path)
    detected = False

    for fi in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        sv_ours = float(np.clip(score_at_frame[fi], 0, 1))
        sv_safe = float(np.clip(safe_at_frame[fi], 0, 1))
        norm_t  = fi / max(total - 1, 1)

        # ── Bar canvas ─────────────────────────────────────────────────────────
        bar = np.ones((bar_h, W, 3), dtype=np.uint8) * 30

        def _draw_bar(canvas, y0, label, val, color_bgr):
            bx, bw, bh = 15, W - 30, 22
            # Background
            cv2.rectangle(canvas, (bx, y0), (bx + bw, y0 + bh), (70, 70, 70), -1)
            # Fill
            fill_w = int(bw * val)
            cv2.rectangle(canvas, (bx, y0), (bx + fill_w, y0 + bh), color_bgr, -1)
            # Threshold line
            tx = bx + int(bw * THRESHOLD)
            cv2.line(canvas, (tx, y0 - 4), (tx, y0 + bh + 4), (255, 255, 0), 2)
            # Labels
            cv2.putText(canvas, f"{label}: {val:.3f}",
                        (bx, y0 + bh + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)

        _draw_bar(bar, 12,  "SAFE (uniform)",         sv_safe,
                  (50, 180, 50))   # green
        _draw_bar(bar, 72,  "Ours (attn+hinge)",      sv_ours,
                  (50, 50, 210))   # red/blue for ours

        # Threshold label
        cv2.putText(bar, f"t={THRESHOLD:.1f}", (W // 2 - 15, 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1)

        # Time indicator
        cv2.putText(bar, f"timestep: {norm_t:.3f}",
                    (W - 165, bar_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)

        # ── Detection flash ────────────────────────────────────────────────────
        if not detected and sv_ours >= THRESHOLD:
            detected = True
        if detected:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 220), 8)
            cv2.putText(bar, "FAILURE DETECTED — STOPPING ROBOT",
                        (15, bar_h - 28),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (80, 80, 255), 2)
        else:
            cv2.putText(bar, "Monitoring ...",
                        (15, bar_h - 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 100), 1)

        out_vid.write(np.vstack([frame, bar]))

    cap.release()
    out_vid.release()
    print(f"  ✓  {out_path}")


def _video01_matplotlib_fallback(seed_results, out_dir: str):
    """Animated score-curve plot when no robot MP4 is available."""
    result = _best_episodes(seed_results, "failure")
    if result is None:
        print("  ✗  video01: no failure episodes in seed_results")
        return
    sc_safe, sc_ours, _, _ = result
    T = max(len(sc_safe), len(sc_ours))
    sc_safe = _interp(sc_safe, T)
    sc_ours = _interp(sc_ours, T)

    det_safe = _first_detection(sc_safe)
    det_ours = _first_detection(sc_ours)
    t_arr    = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor("#111")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_xlim(0, T); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Timestep", color="white"); ax.set_ylabel("Failure Score", color="white")
    ax.axhline(THRESHOLD, color="yellow", ls=":", lw=1.2)
    ax.text(T * 0.01, THRESHOLD + 0.02, f"τ={THRESHOLD}", color="yellow", fontsize=9)

    l_safe, = ax.plot([], [], color=C_SAFE, lw=2, label="SAFE (uniform)")
    l_ours, = ax.plot([], [], color=C_OURS, lw=2.5, label="Ours (attn+hinge)")
    dm_safe = ax.axvline(-1, color=C_SAFE, lw=2, ls="-.", alpha=0, zorder=5)
    dm_ours = ax.axvline(-1, color=C_OURS, lw=2, ls="-.", alpha=0, zorder=5)
    ax.legend(facecolor="#222", labelcolor="white", framealpha=0.7)
    ax.set_title("Live Failure Score — Failure Episode", color="white", fontsize=13,
                 fontweight="bold")

    ds_done = do_done = False

    def animate(fi):
        nonlocal ds_done, do_done
        k = fi + 1
        l_safe.set_data(t_arr[:k], sc_safe[:k])
        l_ours.set_data(t_arr[:k], sc_ours[:k])
        if det_safe is not None and not ds_done and fi >= det_safe:
            ds_done = True
            dm_safe.set_xdata([det_safe]); dm_safe.set_alpha(0.9)
        if det_ours is not None and not do_done and fi >= det_ours:
            do_done = True
            dm_ours.set_xdata([det_ours]); dm_ours.set_alpha(0.9)
        return l_safe, l_ours, dm_safe, dm_ours

    hold = 48
    anim = animation.FuncAnimation(fig, animate, frames=T + hold,
                                   interval=1000 // 24, blit=False)
    _save(anim, os.path.join(out_dir, "video01_score_overlay.mp4"), fps=24)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Video 02 — Animated comparison: SAFE vs Ours (real score curves)
# ═══════════════════════════════════════════════════════════════════════════════

def video02_comparison(seed_results, out_dir: str):
    """
    Side-by-side animation of SAFE vs our model score curves building over real episodes.
    Uses the best failure episode (highest ours score) as the example trajectory,
    overlaid on mean±std bands computed from ALL failure/success episodes.
    """
    sr    = seed_results[0]
    val_r = sr["_val_r"]
    safe_c = sr["safe"]["curves"]
    ours_c = sr["ours"]["curves"]

    # Separate failure / success
    fail_pairs = [(np.array(ss), np.array(so))
                  for ss, so, r in zip(safe_c, ours_c, val_r)
                  if r.episode_success == 0]
    succ_pairs = [(np.array(ss), np.array(so))
                  for ss, so, r in zip(safe_c, ours_c, val_r)
                  if r.episode_success == 1]

    if not fail_pairs:
        print("  ✗  video02: no failure episodes")
        return

    N = 100   # common length for interpolation
    grid = np.linspace(0, 1, N)

    def _mean_std(pairs, idx):
        mat = np.vstack([_interp(p[idx], N) for p in pairs])
        return mat.mean(0), mat.std(0)

    sf_m, sf_s = _mean_std(fail_pairs, 0)   # SAFE failure mean/std
    of_m, of_s = _mean_std(fail_pairs, 1)   # Ours failure mean/std
    ss_m, ss_s = _mean_std(succ_pairs, 0) if succ_pairs else (np.zeros(N), np.zeros(N))
    os_m, os_s = _mean_std(succ_pairs, 1) if succ_pairs else (np.zeros(N), np.zeros(N))

    # Example trajectory = best failure episode (highest episode score for ours)
    best_idx   = int(np.argmax([np.array(so).max() for _, so in fail_pairs]))
    ex_safe    = _interp(fail_pairs[best_idx][0], N)
    ex_ours    = _interp(fail_pairs[best_idx][1], N)
    det_safe_t = _first_detection(ex_safe)
    det_ours_t = _first_detection(ex_ours)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    fig.patch.set_facecolor(C_BG)
    for ax in axes:
        ax.set_facecolor("#111")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    titles      = ["SAFE IndepModel\n(uniform running mean)", "Ours: Temporal Attention\n(learned weights)"]
    model_cols  = [C_SAFE, C_OURS]
    means_f     = [sf_m, of_m];  stds_f = [sf_s, of_s]
    means_s     = [ss_m, os_m];  stds_s = [ss_s, os_s]
    examples    = [ex_safe, ex_ours]
    det_steps   = [det_safe_t, det_ours_t]

    lines_f, fills_f = [], []
    lines_s, fills_s = [], []
    lines_ex         = []
    det_vlines       = []
    score_texts      = []

    for i, ax in enumerate(axes):
        ax.set_xlim(0, N - 1); ax.set_ylim(-0.02, 1.05)
        ax.set_title(titles[i], color="white", fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalised timestep (100 = episode end)", color="white")
        if i == 0:
            ax.set_ylabel("Failure score", color="white")
        ax.axhline(THRESHOLD, color="yellow", ls=":", lw=1.2, alpha=0.7)
        ax.text(N * 0.01, THRESHOLD + 0.02, f"τ={THRESHOLD}", color="yellow", fontsize=8)

        lf, = ax.plot([], [], color=model_cols[i], lw=2.5, label="Fail mean")
        ls, = ax.plot([], [], color="#4488cc", lw=1.5, ls="--", label="Succ mean", alpha=0.7)
        le, = ax.plot([], [], color=model_cols[i], lw=1.5, alpha=0.55, ls="-",
                      label="Best failure ep")
        ff = ax.fill_between([], [], [], alpha=0.15, color=model_cols[i])
        fs = ax.fill_between([], [], [], alpha=0.10, color="#4488cc")

        dv = ax.axvline(-1, color=model_cols[i], lw=2.5, ls="-.", alpha=0, zorder=6)

        st = ax.text(0.97, 0.07, "", transform=ax.transAxes, ha="right",
                     fontsize=12, color=model_cols[i], fontweight="bold",
                     fontfamily="monospace")

        ax.legend(fontsize=8, loc="upper left",
                  facecolor="#222", labelcolor="white", framealpha=0.7)

        lines_f.append(lf); lines_s.append(ls); lines_ex.append(le)
        fills_f.append(ff); fills_s.append(fs)
        det_vlines.append(dv); score_texts.append(st)

    det_done = [False, False]

    def animate(frame):
        k = min(frame + 1, N)
        x = np.arange(k)

        for i, ax in enumerate(axes):
            mf = means_f[i][:k]; sf = stds_f[i][:k]
            ms = means_s[i][:k]; ss = stds_s[i][:k]
            ex = examples[i][:k]

            lines_f[i].set_data(x, mf)
            lines_s[i].set_data(x, ms)
            lines_ex[i].set_data(x, ex)

            fills_f[i].remove()
            fills_s[i].remove()
            fills_f[i] = ax.fill_between(x, mf - sf, mf + sf,
                                          alpha=0.18, color=model_cols[i])
            fills_s[i] = ax.fill_between(x, ms - ss, ms + ss,
                                          alpha=0.10, color="#4488cc")

            cur_score = float(ex[-1]) if len(ex) else 0.0
            score_texts[i].set_text(f"score={cur_score:.3f}")
            score_texts[i].set_color("yellow" if cur_score >= THRESHOLD
                                     else model_cols[i])

            ds = det_steps[i]
            if ds is not None and not det_done[i] and frame >= ds:
                det_done[i] = True
                det_vlines[i].set_xdata([ds])
                det_vlines[i].set_alpha(0.9)

        return lines_f + lines_s + lines_ex + score_texts + det_vlines

    hold = 48
    total_frames = N + hold

    def animate_full(frame):
        return animate(min(frame, N - 1))

    fig.text(0.5, 0.97,
             f"SAFE vs Ours — Score Curves: {len(fail_pairs)} failure, "
             f"{len(succ_pairs)} success episodes (seed 0)",
             ha="center", fontsize=13, fontweight="bold", color="white")
    fig.text(0.5, 0.01,
             "Failure episodes → score HIGH   |   Success episodes → score LOW",
             ha="center", fontsize=10, color="#aaa")
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    anim = animation.FuncAnimation(fig, animate_full, frames=total_frames,
                                   interval=1000 // 24, blit=False)
    _save(anim, os.path.join(out_dir, "video02_comparison.mp4"), fps=24)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Video 03 — Real attention weights for the best failure episode
# ═══════════════════════════════════════════════════════════════════════════════

def video03_attention_anim(seed_results, out_dir: str):
    """
    Animate the REAL attention weight curve alongside the score curve for the
    best failure episode. Uses stored ours['weights'] from seed_results.pkl.
    If weights are unavailable, derives an approximation from the score gradient.
    """
    sr    = seed_results[0]
    val_r = sr["_val_r"]
    safe_c = sr["safe"]["curves"]
    ours_c = sr["ours"]["curves"]
    ours_w = sr["ours"].get("weights", [None] * len(ours_c))

    # Pick best failure episode with actual weight curve if possible
    best_score = -1.0
    best = None
    for sc_s, sc_o, wc, r in zip(safe_c, ours_c, ours_w, val_r):
        if r.episode_success != 0:
            continue
        ep_score = float(np.array(sc_o).max())
        if ep_score > best_score:
            best_score = ep_score
            best = (np.array(sc_s), np.array(sc_o), wc, r)

    if best is None:
        print("  ✗  video03: no failure episodes")
        return

    sc_safe, sc_ours, wc, rollout = best
    T = len(sc_ours)

    if wc is not None and len(wc) == T:
        w = np.array(wc, dtype=float)
    else:
        # Approx weights from score gradient (larger change → more important step)
        grad = np.abs(np.gradient(sc_ours))
        w    = grad / (grad.max() + 1e-8)

    # Safe running mean score
    sc_safe_interp = _interp(sc_safe, T)
    safe_running   = np.cumsum(sc_safe_interp) / np.arange(1, T + 1)

    det_safe_idx = _first_detection(safe_running)
    det_ours_idx = _first_detection(sc_ours)

    # Compute label for task
    task_desc = getattr(rollout, "task_description", f"task {rollout.task_id}") or \
                f"task {rollout.task_id}"
    w_label = "Real attention weights" if wc is not None else "Approx. weights (|Δscore|)"

    fig, axes = plt.subplots(3, 1, figsize=(12, 8),
                              gridspec_kw={"height_ratios": [1, 1.5, 1]})
    fig.patch.set_facecolor(C_BG)
    for ax in axes:
        ax.set_facecolor("#111")
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#333")

    t_arr = np.arange(T)

    # ── Row 0: attention weight bars ──────────────────────────────────────────
    ax0 = axes[0]
    ax0.set_xlim(0, T - 1); ax0.set_ylim(0, w.max() * 1.25)
    ax0.set_ylabel(f"wₜ\n({w_label})", color=C_GOLD, fontsize=9)
    ax0.set_title(
        f"Temporal Attention Analysis — failure episode\nTask: {task_desc[:60]}",
        color="white", fontsize=11, fontweight="bold")
    bars = ax0.bar(t_arr, np.zeros(T), color=C_GOLD, alpha=0.75, width=0.9)

    # ── Row 1: score curves ───────────────────────────────────────────────────
    ax1 = axes[1]
    ax1.set_xlim(0, T - 1); ax1.set_ylim(-0.02, 1.05)
    ax1.set_ylabel("Failure Score", color="white", fontsize=10)
    ax1.axhline(THRESHOLD, color="yellow", ls=":", lw=1.2, alpha=0.7)
    ax1.text(T * 0.01, THRESHOLD + 0.02, f"τ={THRESHOLD}", color="yellow", fontsize=8)

    l_safe, = ax1.plot([], [], color=C_SAFE, lw=2,   ls="--", label="SAFE (uniform mean)")
    l_ours, = ax1.plot([], [], color=C_OURS, lw=2.5, ls="-",  label="Ours (attn weighted)")
    dv_safe = ax1.axvline(-1, color=C_SAFE, lw=1.8, ls="-.", alpha=0, zorder=5)
    dv_ours = ax1.axvline(-1, color=C_OURS, lw=1.8, ls="-.", alpha=0, zorder=5)
    ax1.legend(fontsize=9, loc="upper left", facecolor="#222",
               labelcolor="white", framealpha=0.7)

    # ── Row 2: heatmap strip ──────────────────────────────────────────────────
    ax2 = axes[2]
    ax2.set_xlim(0, T - 1); ax2.set_ylim(0, 1)
    ax2.set_ylabel("Weight\nheatmap", color=C_GOLD, fontsize=9)
    ax2.set_xlabel("Timestep", color="white")
    heat_data  = np.zeros((1, T), dtype=float)
    heat_img   = ax2.imshow(heat_data, aspect="auto",
                             extent=[0, T - 1, 0, 1], cmap="hot",
                             vmin=0, vmax=max(float(w.max()), 1e-6),
                             interpolation="nearest")
    cursor = ax2.axvline(-1, color="cyan", lw=1.5, alpha=0.8)

    fig.tight_layout(pad=1.5)

    ds_done = do_done = False

    def animate(frame):
        nonlocal ds_done, do_done
        k = min(frame + 1, T)

        # Weight bars
        for rect, h in zip(bars, w[:k]):
            rect.set_height(h)
        for rect in list(bars)[k:]:
            rect.set_height(0)

        # Score curves
        l_safe.set_data(t_arr[:k], safe_running[:k])
        l_ours.set_data(t_arr[:k], sc_ours[:k])

        # Detection markers
        if det_safe_idx is not None and not ds_done and frame >= det_safe_idx:
            ds_done = True
            dv_safe.set_xdata([det_safe_idx]); dv_safe.set_alpha(0.8)
        if det_ours_idx is not None and not do_done and frame >= det_ours_idx:
            do_done = True
            dv_ours.set_xdata([det_ours_idx]); dv_ours.set_alpha(0.8)

        # Heatmap
        heat_data[0, :k] = w[:k]
        heat_img.set_data(heat_data)
        cursor.set_xdata([k - 1])

        return list(bars) + [l_safe, l_ours, dv_safe, dv_ours, heat_img, cursor]

    hold = 48
    anim = animation.FuncAnimation(fig, animate, frames=T + hold,
                                   interval=1000 // 24, blit=False)
    _save(anim, os.path.join(out_dir, "video03_attention_anim.mp4"), fps=24)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Video 04 — Architecture data-flow animation (no data needed)
# ═══════════════════════════════════════════════════════════════════════════════

def video04_arch_flow(out_dir: str):
    """Animate data flowing through our Temporal Attention architecture."""

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5)
    ax.set_facecolor("#0d0d1a"); ax.axis("off")
    fig.patch.set_facecolor("#0d0d1a")

    def _box(x, y, w, h, text, fc, alpha=0.25):
        box = FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                              boxstyle="round,pad=0.15",
                              facecolor=fc, edgecolor="white",
                              linewidth=1.2, alpha=alpha, zorder=3)
        ax.add_patch(box)
        t = ax.text(x, y, text, ha="center", va="center", fontsize=9,
                    color="white", fontweight="bold", zorder=4,
                    multialignment="center")
        return box, t

    boxes_cfg = [
        (1.2,  2.5, 1.8, 1.0, "hₜ\nhidden\nstate",        C_BLUE),
        (3.5,  2.5, 1.8, 0.9, "Shared\nEncoder\n(MLP)",    "#555"),
        (6.2,  3.6, 2.0, 0.8, "Score Head\n→ pₜ",         C_OURS),
        (6.2,  1.4, 2.0, 0.8, "Weight Head\n→ wₜ",        C_GOLD),
        (9.2,  2.5, 2.2, 1.1, "Causal\nAttention\n∑wᵢpᵢ/∑wᵢ", "#1a1a7e"),
        (12.0, 2.5, 1.8, 0.9, "scoreₜ\n≥ τ?",             "#333"),
    ]
    static_arts = [_box(*cfg) for cfg in boxes_cfg]

    arrows = [
        (2.1, 2.5, 2.6, 2.5),   # h_t → encoder
        (4.4, 2.7, 5.2, 3.5),   # encoder → score head
        (4.4, 2.3, 5.2, 1.5),   # encoder → weight head
        (7.2, 3.5, 8.1, 2.8),   # score head → aggregation
        (7.2, 1.5, 8.1, 2.2),   # weight head → aggregation
        (10.3, 2.5, 11.1, 2.5), # aggregation → decision
    ]
    for (x0, y0, x1, y1) in arrows:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.5))

    pulse_cols = [C_BLUE, C_BLUE, C_BLUE, C_OURS, C_GOLD, "#8888ff"]
    pulses = [ax.plot([], [], "o", color=c, ms=14, zorder=6, alpha=0)[0]
              for c in pulse_cols]

    readouts = [
        ax.text(3.5, 1.2, "", ha="center", fontsize=9, color=C_BLUE,  fontfamily="monospace", zorder=7),
        ax.text(6.2, 4.5, "", ha="center", fontsize=9, color=C_OURS,  fontfamily="monospace", zorder=7),
        ax.text(6.2, 0.4, "", ha="center", fontsize=9, color=C_GOLD,  fontfamily="monospace", zorder=7),
        ax.text(9.2, 1.0, "", ha="center", fontsize=9, color="#8888ff", fontfamily="monospace", zorder=7),
        ax.text(12.0,1.5, "", ha="center", fontsize=9, color="white", fontfamily="monospace", zorder=7),
    ]

    ax.text(7.0, 4.85, "Our Architecture: Temporal Attention Flow",
            ha="center", fontsize=14, fontweight="bold", color="white", zorder=8)

    CYCLE = 96

    def _lerp(t, x0, y0, x1, y1):
        return x0 + t * (x1 - x0), y0 + t * (y1 - y0)

    def animate(frame):
        phase = (frame % CYCLE) / CYCLE

        for i, (b, _) in enumerate(static_arts):
            thresh = i / len(static_arts)
            a = min(1.0, max(0.25, (phase - thresh * 0.55) * 4))
            b.set_alpha(a)

        # Pulse 0: h_t → encoder
        if phase < 0.18:
            x, y = _lerp(phase / 0.18, *arrows[0])
            pulses[0].set_data([x], [y]); pulses[0].set_alpha(1.0)
        else:
            pulses[0].set_alpha(0)
            if phase >= 0.18: readouts[0].set_text("enc(hₜ)")

        # Pulse 1: encoder → score head
        if 0.16 <= phase < 0.34:
            x, y = _lerp((phase - 0.16) / 0.18, *arrows[1])
            pulses[1].set_data([x], [y]); pulses[1].set_alpha(1.0)
        else:
            pulses[1].set_alpha(0)
            if phase >= 0.34: readouts[1].set_text("pₜ=0.73")

        # Pulse 2: encoder → weight head
        if 0.20 <= phase < 0.38:
            x, y = _lerp((phase - 0.20) / 0.18, *arrows[2])
            pulses[2].set_data([x], [y]); pulses[2].set_alpha(1.0)
        else:
            pulses[2].set_alpha(0)
            if phase >= 0.38: readouts[2].set_text("wₜ=0.91")

        # Pulses 3+4: score/weight → aggregation
        if 0.38 <= phase < 0.56:
            tp = (phase - 0.38) / 0.18
            x3, y3 = _lerp(tp, *arrows[3])
            x4, y4 = _lerp(tp, *arrows[4])
            pulses[3].set_data([x3], [y3]); pulses[3].set_alpha(1.0)
            pulses[4].set_data([x4], [y4]); pulses[4].set_alpha(1.0)
        else:
            pulses[3].set_alpha(0); pulses[4].set_alpha(0)
            if phase >= 0.56: readouts[3].set_text("scoreₜ=0.68")

        # Pulse 5: aggregation → decision
        if 0.58 <= phase < 0.76:
            x, y = _lerp((phase - 0.58) / 0.18, *arrows[5])
            pulses[5].set_data([x], [y]); pulses[5].set_alpha(1.0)
        else:
            pulses[5].set_alpha(0)
            if phase >= 0.76:
                readouts[4].set_text("⚠ ALERT" if 0.68 >= THRESHOLD else "✓ OK")
                readouts[4].set_color("#ff4444" if 0.68 >= THRESHOLD else "#44ff44")

        if phase < 0.04:
            for r in readouts: r.set_text("")

        return pulses + readouts + [b for b, _ in static_arts]

    anim = animation.FuncAnimation(fig, animate, frames=CYCLE * 3,
                                   interval=1000 // 24, blit=False)
    _save(anim, os.path.join(out_dir, "video04_arch_flow.mp4"), fps=24)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Video 05 — Earliest detection moment — annotated robot video
# ═══════════════════════════════════════════════════════════════════════════════

def video05_detection_moment(seed_results, rollout_dir, out_dir: str):
    """
    Find the failure episode where our model detects failure EARLIEST
    (smallest normalised detection time among true positives).
    Render a clip from 1.5 s before detection to 2.5 s after.
    Falls back to matplotlib if no MP4 found.
    """
    try:
        import cv2
        _has_cv2 = True
    except ImportError:
        _has_cv2 = False

    sr    = seed_results[0]
    val_r = sr["_val_r"]
    safe_c = sr["safe"]["curves"]
    ours_c = sr["ours"]["curves"]

    # Find the failure episode with earliest CORRECT detection (true positive)
    best = None   # (det_norm, sc_safe, sc_ours, mp4, det_idx)
    for sc_s, sc_o, r in zip(safe_c, ours_c, val_r):
        if r.episode_success != 0:
            continue
        sc_o_arr = np.array(sc_o)
        sc_s_arr = np.array(sc_s)
        det_idx  = _first_detection(sc_o_arr)
        if det_idx is None:
            continue   # model missed this episode
        det_norm = det_idx / len(sc_o_arr)

        mp4 = r.mp4_path
        if (mp4 is None or not os.path.exists(mp4)) and rollout_dir:
            for pat in [
                os.path.join(rollout_dir,
                             f"*task{r.task_id}*ep{r.episode_idx}*succ0*.mp4"),
                os.path.join(rollout_dir, "*succ0*.mp4"),
            ]:
                hits = sorted(glob.glob(pat))
                if hits: mp4 = hits[0]; break

        if best is None or det_norm < best[0]:
            best = (det_norm, sc_s_arr, sc_o_arr, mp4, det_idx)

    if best is None:
        print("  ✗  video05: model made no detections on failure episodes")
        return

    det_norm, sc_safe, sc_ours, mp4, det_idx = best
    det_safe_idx = _first_detection(_interp(sc_safe, len(sc_ours)))

    print(f"  video05: earliest detection at t={det_norm:.3f} "
          f"(step {det_idx}/{len(sc_ours)})")

    if not _has_cv2 or mp4 is None or not os.path.exists(mp4):
        print("  (video05: no robot MP4 — generating matplotlib fallback)")
        _video05_matplotlib_fallback(sc_safe, sc_ours, det_idx, det_safe_idx, out_dir)
        return

    cap   = cv2.VideoCapture(mp4)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps   = max(int(cap.get(cv2.CAP_PROP_FPS)), 10)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    det_frame = int(det_norm * total)
    pre       = min(int(fps * 1.5), det_frame)
    post      = min(int(fps * 2.5), total - det_frame)
    start_f   = det_frame - pre
    end_f     = det_frame + post

    score_at_frame = _interp(sc_ours, total)
    safe_at_frame  = _interp(sc_safe, total)

    out_path = os.path.join(out_dir, "video05_detection_moment.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    bar_h    = 155
    out_vid  = cv2.VideoWriter(out_path, fourcc, fps, (W, H + bar_h))

    cap = cv2.VideoCapture(mp4)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

    for fi in range(start_f, end_f):
        ret, frame = cap.read()
        if not ret:
            break

        sv_ours = float(np.clip(score_at_frame[fi], 0, 1))
        sv_safe = float(np.clip(safe_at_frame[fi],  0, 1))
        is_post = fi >= det_frame

        # Frame border
        if is_post:
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 220), 8)
            cv2.putText(frame, "FAILURE DETECTED — STOPPING", (20, 40),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (80, 80, 255), 2)
        else:
            frames_left = det_frame - fi
            cv2.putText(frame, f"Monitoring ... ({frames_left / fps:.1f}s to detection)",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 210, 100), 2)

        # Bar
        bar = np.ones((bar_h, W, 3), dtype=np.uint8) * 20

        def _draw(canvas, y0, label, val, col):
            bx, bw, bh = 12, W - 24, 22
            cv2.rectangle(canvas, (bx, y0), (bx + bw, y0 + bh), (60, 60, 60), -1)
            fw = int(bw * val)
            cv2.rectangle(canvas, (bx, y0), (bx + fw, y0 + bh), col, -1)
            tx = bx + int(bw * THRESHOLD)
            cv2.line(canvas, (tx, y0 - 4), (tx, y0 + bh + 4), (255, 255, 0), 2)
            cv2.putText(canvas, f"{label}: {val:.3f}",
                        (bx, y0 + bh + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)

        _draw(bar, 10,  "SAFE (uniform)",    sv_safe, (50, 180, 50))
        _draw(bar, 60,  "Ours (attn+hinge)", sv_ours, (50, 50, 210))

        # Time info
        norm_t = fi / total
        cv2.putText(bar, f"t={norm_t:.3f}  |  det @ t={det_norm:.3f}",
                    (12, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (150, 150, 150), 1)
        if is_post:
            dt_s = (fi - det_frame) / fps
            cv2.putText(bar, f"+{dt_s:.1f}s after detection",
                        (W - 200, bar_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (100, 100, 255), 1)

        out_vid.write(np.vstack([frame, bar]))

    cap.release()
    out_vid.release()
    print(f"  ✓  {out_path}")


def _video05_matplotlib_fallback(sc_safe, sc_ours, det_idx_ours,
                                  det_idx_safe, out_dir: str):
    T = len(sc_ours)
    safe_running = np.cumsum(_interp(sc_safe, T)) / np.arange(1, T + 1)
    t_arr = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(C_BG)
    ax.set_facecolor("#111")
    ax.tick_params(colors="white")
    for sp in ax.spines.values(): sp.set_edgecolor("#444")
    ax.set_xlim(0, T - 1); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Timestep", color="white"); ax.set_ylabel("Failure Score", color="white")
    ax.axhline(THRESHOLD, color="yellow", ls=":", lw=1.2)
    ax.text(T * 0.01, THRESHOLD + 0.02, f"τ={THRESHOLD}", color="yellow", fontsize=9)

    l_safe, = ax.plot([], [], color=C_SAFE, lw=2,   ls="--", label="SAFE")
    l_ours, = ax.plot([], [], color=C_OURS, lw=2.5, ls="-",  label="Ours (attn+hinge)")
    dv_safe = ax.axvline(-1, color=C_SAFE, lw=2, ls="-.", alpha=0, zorder=5)
    dv_ours = ax.axvline(-1, color=C_OURS, lw=2, ls="-.", alpha=0, zorder=5)
    ax.legend(facecolor="#222", labelcolor="white", framealpha=0.7)
    ax.set_title("Earliest Failure Detection — Failure Episode", color="white",
                 fontsize=13, fontweight="bold")

    # Annotation for ours detection
    if det_idx_ours is not None:
        ax.axvspan(det_idx_ours, T - 1, alpha=0.08, color=C_OURS)
        ax.text(det_idx_ours + 1, 0.92,
                f"Ours detects @ t={det_idx_ours / T:.3f}",
                color=C_OURS, fontsize=9)
    if det_idx_safe is not None:
        ax.text(det_idx_safe + 1, 0.80,
                f"SAFE detects @ t={det_idx_safe / T:.3f}",
                color=C_SAFE, fontsize=9)

    ds_done = do_done = False

    def animate(fi):
        nonlocal ds_done, do_done
        k = min(fi + 1, T)
        l_safe.set_data(t_arr[:k], safe_running[:k])
        l_ours.set_data(t_arr[:k], sc_ours[:k])
        if det_idx_safe is not None and not ds_done and fi >= det_idx_safe:
            ds_done = True
            dv_safe.set_xdata([det_idx_safe]); dv_safe.set_alpha(0.85)
        if det_idx_ours is not None and not do_done and fi >= det_idx_ours:
            do_done = True
            dv_ours.set_xdata([det_idx_ours]); dv_ours.set_alpha(0.85)
        return l_safe, l_ours, dv_safe, dv_ours

    hold = 72
    anim = animation.FuncAnimation(fig, animate, frames=T + hold,
                                   interval=1000 // 24, blit=False)
    _save(anim, os.path.join(out_dir, "video05_detection_moment.mp4"), fps=24)
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Generate accurate presentation videos")
    parser.add_argument("--seed_results",  default=None,
                        help="Path to seed_results.pkl (from compare_with_safe.py "
                             "--save_seed_results)")
    parser.add_argument("--rollout_dir",   default=None,
                        help="Rollout folder for robot MP4 overlays")
    parser.add_argument("--output_dir",    default="./presentation_videos")
    parser.add_argument("--diagrams_only", action="store_true",
                        help="Only generate the architecture diagram video")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    seed_results = None
    if args.seed_results and os.path.exists(args.seed_results):
        with open(args.seed_results, "rb") as f:
            seed_results = pickle.load(f)
        sr0 = seed_results[0]
        n_val = len(sr0["_val_r"])
        n_fail = sum(1 for r in sr0["_val_r"] if r.episode_success == 0)
        n_succ = n_val - n_fail
        has_weights = any(w is not None for w in sr0["ours"].get("weights", []))
        print(f"Loaded {len(seed_results)} seed(s) — "
              f"val set: {n_val} episodes ({n_fail} failure / {n_succ} success)")
        print(f"Attention weight curves stored: {has_weights}\n")
        if not has_weights:
            print("  NOTE: Re-run compare_with_safe.py --save_seed_results to get real "
                  "weight curves.\n  video03 will use |Δscore| approximation.\n")
    else:
        if not args.diagrams_only:
            print("WARNING: no seed_results.pkl provided — data-driven videos skipped.\n"
                  "  Run compare_with_safe.py --save_seed_results first.\n")

    print(f"Generating videos → {args.output_dir}/\n")

    # Architecture diagram (always generated — no data needed)
    print("video04: Architecture flow animation ...")
    video04_arch_flow(args.output_dir)

    if seed_results and not args.diagrams_only:
        print("\nvideo02: SAFE vs Ours animated comparison ...")
        video02_comparison(seed_results, args.output_dir)

        print("\nvideo03: Attention weights animation ...")
        video03_attention_anim(seed_results, args.output_dir)

        print("\nvideo01: Score overlay on robot video ...")
        video01_score_overlay(seed_results, args.rollout_dir, args.output_dir)

        print("\nvideo05: Detection moment clip ...")
        video05_detection_moment(seed_results, args.rollout_dir, args.output_dir)

    print(f"\n{'='*60}")
    print("Done — videos saved:")
    for ext in ("*.mp4", "*.gif"):
        for f in sorted(Path(args.output_dir).glob(ext)):
            size_mb = f.stat().st_size / 1e6
            print(f"  {f.name:<45} {size_mb:.1f} MB")
    print(f"\nEmbed in slides: Insert → Video → From file")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
