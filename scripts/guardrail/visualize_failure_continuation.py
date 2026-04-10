"""
Day 2 Video (Continuation) — visualize_failure_continuation.py
=============================================================
Modified version of create_guardrail_video.py that does NOT stop the video
when the guardrail triggers.  Instead, it shows the full episode to demonstrate
how the robot continues and eventually fails after the alert has fired.

The visual alert (Red ✗ and banner) is latched: once it appears, it stays
until the end of the video.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import zoom

# Make sure sibling package is importable when run as a script
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR.parent.parent))   # repo root

from scripts.guardrail.attention_guardrail import AttentionGuardrail, calibrate_threshold


# ---------------------------------------------------------------------------
# Rendering helpers (mostly copied from create_guardrail_video.py)
# ---------------------------------------------------------------------------

_RED   = (200,  40,  40)
_GREEN = ( 50, 180,  80)
_BLUE  = ( 60, 120, 220)
_WHITE = (255, 255, 255)
_ORANGE= (255, 140,   0)
_BLACK = (  0,   0,   0)

OBS_H, OBS_W = 448, 448


def _upscale_attn(attn_map: np.ndarray, h: int, w: int) -> np.ndarray:
    sy = h / attn_map.shape[0]
    sx = w / attn_map.shape[1]
    return zoom(attn_map, (sy, sx), order=1)


def _blend_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    cmap = plt.get_cmap("hot")
    coloured = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return (frame * (1 - alpha) + coloured * alpha).astype(np.uint8)


def _draw_x_overlay(frame: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    h, w = frame.shape[:2]
    thickness = max(w // 18, 8)
    margin = w // 12
    colour = (220, 30, 30)
    cv2.line(overlay, (margin, margin), (w - margin, h - margin), colour, thickness, cv2.LINE_AA)
    cv2.line(overlay, (w - margin, margin), (margin, h - margin), colour, thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)


def _peak_to_pixel(attn_map: np.ndarray, out_h: int, out_w: int) -> tuple[int, int]:
    patch_h, patch_w = attn_map.shape
    flat_idx = int(np.argmax(attn_map))
    row = flat_idx // patch_w
    col = flat_idx % patch_w
    px = int(col * out_w / patch_w + out_w / patch_w / 2)
    py = int(row * out_h / patch_h + out_h / patch_h / 2)
    return px, py


def build_obs_panel(
    image_rgb: np.ndarray,
    attn_map: np.ndarray,
    result,
    is_latched_triggered: bool,  # NEW: latched state
    out_h: int = OBS_H,
    out_w: int = OBS_W,
) -> np.ndarray:
    frame = cv2.resize(image_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    hmap = _upscale_attn(attn_map, out_h, out_w)
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    frame = _blend_heatmap(frame, hmap, alpha=0.45)

    px, py = _peak_to_pixel(attn_map, out_h, out_w)
    cv2.circle(frame, (px, py), radius=12, color=_BLUE, thickness=-1)
    cv2.circle(frame, (px, py), radius=13, color=_WHITE, thickness=2)

    if result.anchor_xy is not None and not result.in_anchor_phase:
        ax_px = int(result.anchor_xy[0] / 224 * out_w)
        ax_py = int(result.anchor_xy[1] / 224 * out_h)
        cv2.circle(frame, (ax_px, ax_py), radius=10, color=_WHITE, thickness=-1)
        cv2.circle(frame, (ax_px, ax_py), radius=11, color=_BLACK, thickness=2)
        cv2.line(frame, (ax_px, ax_py), (px, py), _ORANGE, 2, cv2.LINE_AA)

    # Use latched state for visual alert
    if is_latched_triggered:
        frame = _draw_x_overlay(frame)
        banner_h = out_h // 8
        cv2.rectangle(frame, (0, 0), (out_w, banner_h), (180, 0, 0), -1)
        cv2.putText(
            frame, "GUARDRAIL TRIGGERED — ALERT",
            (out_w // 16, banner_h * 3 // 4),
            cv2.FONT_HERSHEY_DUPLEX, out_w / 600,
            _WHITE, 2, cv2.LINE_AA,
        )
    elif result.in_anchor_phase:
        cv2.putText(
            frame, f"Anchor phase ({result.timestep + 1})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, _GREEN, 2, cv2.LINE_AA,
        )

    return frame


def build_plot_panel(
    results: list,
    threshold: float,
    total_steps: int,
    out_h: int = OBS_H,
    out_w: int = OBS_W,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(out_w / 100, out_h / 100), dpi=100)

    t_vals = [r.timestep for r in results]
    d_vals = [r.smoothed_distance for r in results]

    triggered_steps = [r.timestep for r in results if r.triggered]
    if triggered_steps:
        ax.axvspan(triggered_steps[0], total_steps, alpha=0.18, color="red")

    ax.plot(t_vals, d_vals, color="#1f77b4", linewidth=2, label="Smoothed distance")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.0f} px")

    trig = next((r.timestep for r in results if r.triggered), None)
    if trig is not None:
        ax.axvline(trig, color="red", linewidth=2, linestyle="-", alpha=0.6)
        ax.text(trig + 0.3, threshold * 1.05, "ALERT", color="red",
                fontsize=9, fontweight="bold")

    ax.set_xlim(0, total_steps)
    y_max = max(threshold * 1.8, max(d_vals) * 1.1) if d_vals else threshold * 2
    ax.set_ylim(0, y_max)
    ax.set_xlabel("Timestep $t$", fontsize=10)
    ax.set_ylabel("Distance from anchor (px)", fontsize=10)
    ax.set_title("Attention-Peak Distance", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(pad=0.5)

    fig.canvas.draw()
    img = np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]
    plt.close(fig)

    if img.shape[0] != out_h or img.shape[1] != out_w:
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return img


def make_episode_video(
    ep: dict,
    threshold: float,
    anchor_steps: int = 3,
    smooth_window: int = 5,
    fps: float = 10.0,
    save_path: Path | None = None,
) -> Path:
    attn_maps   = ep["attention_maps"]
    images      = ep["images"]
    success     = bool(int(ep["success"]))
    ep_idx      = int(ep.get("episode_idx", 0))
    task_desc   = str(ep.get("task_description", ""))

    T = len(attn_maps)
    img_h, img_w = images.shape[1], images.shape[2]

    guardrail = AttentionGuardrail(
        threshold=threshold,
        anchor_steps=anchor_steps,
        smooth_window=smooth_window,
        img_h=img_h,
        img_w=img_w,
    )

    all_results = []
    for t in range(T):
        r = guardrail.step(attn_maps[t])
        all_results.append(r)

    video_frames = []
    latched_triggered = False
    for t in range(T):  # Always show full episode
        if all_results[t].triggered:
            latched_triggered = True
            
        obs_panel  = build_obs_panel(images[t], attn_maps[t], all_results[t], latched_triggered)
        plot_panel = build_plot_panel(all_results[:t + 1], threshold, T)
        frame      = np.concatenate([obs_panel, plot_panel], axis=1)

        label  = "SUCCESS" if success else "FAILURE"
        colour = _GREEN if success else _RED
        bar_h  = 48
        bar    = np.zeros((bar_h, frame.shape[1], 3), dtype=np.uint8)
        bar[:] = colour
        short_desc = task_desc[:60] + ("…" if len(task_desc) > 60 else "")
        title_str  = f"Ep {ep_idx} | {label} | {short_desc} | t={t}"
        cv2.putText(bar, title_str, (10, bar_h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, _WHITE, 1, cv2.LINE_AA)
        frame = np.concatenate([bar, frame], axis=0)
        video_frames.append(frame)

    if save_path is None:
        data_dir = Path(ep.get("path", "."))
        save_path = data_dir.parent / f"ep{ep_idx:03d}_succ{int(success)}_continuation.mp4"

    imageio.mimsave(str(save_path), video_frames, fps=fps)
    print(f"  [video] ep {ep_idx:03d}  {'SUC' if success else 'FAIL'}  → {save_path.name}")
    return save_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", default="./attention_rollouts")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--anchor_steps", type=int, default=3)
    parser.add_argument("--smooth_window", type=int, default=5)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--episode_idx", type=int, default=None)
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rollout_files = sorted(data_dir.glob("*.npz"))
    if not rollout_files:
        print(f"[error] No .npz files found in {data_dir}")
        sys.exit(1)

    rollouts = []
    for p in rollout_files:
        d = dict(np.load(p, allow_pickle=True))
        d["path"] = p
        rollouts.append(d)

    if args.threshold is not None:
        threshold = args.threshold
    else:
        cal = calibrate_threshold(rollouts, anchor_steps=args.anchor_steps)
        threshold = cal["suggested_threshold"]

    if args.episode_idx is not None:
        rollouts = [ep for ep in rollouts if int(ep.get("episode_idx", -1)) == args.episode_idx]

    for ep in rollouts:
        make_episode_video(ep, threshold, args.anchor_steps, args.smooth_window, args.fps)

if __name__ == "__main__":
    main()
