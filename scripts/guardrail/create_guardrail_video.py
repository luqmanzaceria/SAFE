"""
Day 2 Video — create_guardrail_video.py
=======================================
Load saved .npz rollout files, run the AttentionGuardrail on each, and
produce an annotated MP4 video showing:

  Left panel  — RGB observation with:
    • Attention heatmap blended in (hot colourmap)
    • Blue dot  = current attention peak
    • White dot = anchor peak
    • Orange line connecting anchor → current peak
    • Red ✗ overlay (full-frame) when the guardrail triggers

  Right panel — distance-over-time plot that updates frame-by-frame, with a
    horizontal dashed threshold line and a vertical marker at the trigger step.

The video is saved alongside each .npz as  <ep>_guardrail.mp4.

Usage
-----
  # Process all rollouts in a directory (uses auto-calibrated threshold)
  python create_guardrail_video.py --data_dir ./attention_rollouts

  # Use a specific threshold
  python create_guardrail_video.py --data_dir ./attention_rollouts --threshold 30

  # Only process one episode
  python create_guardrail_video.py --data_dir ./attention_rollouts --episode_idx 3
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
# Rendering helpers
# ---------------------------------------------------------------------------

_RED   = (200,  40,  40)
_GREEN = ( 50, 180,  80)
_BLUE  = ( 60, 120, 220)
_WHITE = (255, 255, 255)
_ORANGE= (255, 140,   0)
_BLACK = (  0,   0,   0)

# Standard output size for the left (observation) panel
OBS_H, OBS_W = 448, 448    # 2× upscale from 224×224 for legibility


def _upscale_attn(attn_map: np.ndarray, h: int, w: int) -> np.ndarray:
    """Bilinear zoom of a patch attention map to (h, w)."""
    sy = h / attn_map.shape[0]
    sx = w / attn_map.shape[1]
    return zoom(attn_map, (sy, sx), order=1)


def _blend_heatmap(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a normalised heatmap on an RGB frame using the 'hot' colourmap."""
    cmap = plt.get_cmap("hot")
    coloured = (cmap(heatmap)[:, :, :3] * 255).astype(np.uint8)
    return (frame * (1 - alpha) + coloured * alpha).astype(np.uint8)


def _draw_x_overlay(frame: np.ndarray) -> np.ndarray:
    """Draw a large, semi-transparent red ✗ across the frame."""
    overlay = frame.copy()
    h, w = frame.shape[:2]
    thickness = max(w // 18, 8)
    margin = w // 12
    colour = (220, 30, 30)
    cv2.line(overlay, (margin, margin), (w - margin, h - margin), colour, thickness, cv2.LINE_AA)
    cv2.line(overlay, (w - margin, margin), (margin, h - margin), colour, thickness, cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.75, frame, 0.25, 0)


def _peak_to_pixel(attn_map: np.ndarray, out_h: int, out_w: int) -> tuple[int, int]:
    """Return (px, py) image coords of the attention argmax, scaled to (out_h, out_w)."""
    patch_h, patch_w = attn_map.shape
    flat_idx = int(np.argmax(attn_map))
    row = flat_idx // patch_w
    col = flat_idx % patch_w
    px = int(col * out_w / patch_w + out_w / patch_w / 2)
    py = int(row * out_h / patch_h + out_h / patch_h / 2)
    return px, py


def build_obs_panel(
    image_rgb: np.ndarray,      # (H, W, 3) uint8
    attn_map: np.ndarray,       # (ph, pw)
    result,                      # GuardrailResult
    out_h: int = OBS_H,
    out_w: int = OBS_W,
) -> np.ndarray:
    """Render the left observation panel."""
    # 1. Resize the raw image
    frame = cv2.resize(image_rgb, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    # 2. Blend attention heatmap
    hmap = _upscale_attn(attn_map, out_h, out_w)
    hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
    frame = _blend_heatmap(frame, hmap, alpha=0.45)

    # 3. Current peak (blue dot)
    px, py = _peak_to_pixel(attn_map, out_h, out_w)
    cv2.circle(frame, (px, py), radius=12, color=_BLUE, thickness=-1)
    cv2.circle(frame, (px, py), radius=13, color=_WHITE, thickness=2)

    # 4. Anchor (white dot) and connector line
    if result.anchor_xy is not None and not result.in_anchor_phase:
        patch_h, patch_w = attn_map.shape
        ax_px = int(result.anchor_xy[0] * out_w / (patch_w * (224 / patch_w)))
        ax_py = int(result.anchor_xy[1] * out_h / (patch_h * (224 / patch_h)))
        # Recalculate from actual stored pixel coords (img space = 224)
        ax_px = int(result.anchor_xy[0] / 224 * out_w)
        ax_py = int(result.anchor_xy[1] / 224 * out_h)
        cv2.circle(frame, (ax_px, ax_py), radius=10, color=_WHITE, thickness=-1)
        cv2.circle(frame, (ax_px, ax_py), radius=11, color=_BLACK, thickness=2)
        cv2.line(frame, (ax_px, ax_py), (px, py), _ORANGE, 2, cv2.LINE_AA)

    # 5. Red ✗ if triggered
    if result.triggered:
        frame = _draw_x_overlay(frame)
        # Text banner
        banner_h = out_h // 8
        cv2.rectangle(frame, (0, 0), (out_w, banner_h), (180, 0, 0), -1)
        cv2.putText(
            frame, "GUARDRAIL TRIGGERED — STOP",
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
    results: list,      # list of GuardrailResult up to current step
    threshold: float,
    total_steps: int,
    out_h: int = OBS_H,
    out_w: int = OBS_W,
) -> np.ndarray:
    """Render the right distance-over-time plot panel."""
    fig, ax = plt.subplots(figsize=(out_w / 100, out_h / 100), dpi=100)

    t_vals = [r.timestep for r in results]
    d_vals = [r.smoothed_distance for r in results]

    # Shade triggered region in pink
    triggered_steps = [r.timestep for r in results if r.triggered]
    if triggered_steps:
        ax.axvspan(triggered_steps[0], total_steps, alpha=0.18, color="red")

    ax.plot(t_vals, d_vals, color="#1f77b4", linewidth=2, label="Smoothed distance")
    ax.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
               label=f"Threshold = {threshold:.0f} px")

    # Mark trigger step
    trig = next((r.timestep for r in results if r.triggered), None)
    if trig is not None:
        ax.axvline(trig, color="red", linewidth=2, linestyle="-", alpha=0.6)
        ax.text(trig + 0.3, threshold * 1.05, "STOP", color="red",
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

    # Resize to match obs panel
    if img.shape[0] != out_h or img.shape[1] != out_w:
        img = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return img


# ---------------------------------------------------------------------------
# Per-episode video generation
# ---------------------------------------------------------------------------

def make_episode_video(
    ep: dict,
    threshold: float,
    anchor_steps: int = 3,
    smooth_window: int = 5,
    fps: float = 10.0,
    save_path: Path | None = None,
) -> Path:
    """
    Generate and save an annotated guardrail video for one episode.

    Returns the path to the saved video.
    """
    attn_maps   = ep["attention_maps"]    # (T, ph, pw)
    images      = ep["images"]            # (T, H, W, 3)
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

    # Pre-run the guardrail to collect all results
    all_results = []
    stopped_at = None
    for t in range(T):
        r = guardrail.step(attn_maps[t])
        all_results.append(r)
        if r.triggered and stopped_at is None:
            stopped_at = t

    # Determine video length (stop at trigger + 5 frames for dramatic effect)
    video_len = T
    if stopped_at is not None:
        video_len = min(T, stopped_at + 6)

    video_frames = []
    for t in range(video_len):
        obs_panel  = build_obs_panel(images[t], attn_maps[t], all_results[t])
        plot_panel = build_plot_panel(all_results[:t + 1], threshold, T)
        frame      = np.concatenate([obs_panel, plot_panel], axis=1)

        # Title bar (height must be a multiple of 16 for ffmpeg compatibility)
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

    # Save video
    if save_path is None:
        data_dir = Path(ep.get("path", "."))
        save_path = data_dir.parent / f"ep{ep_idx:03d}_succ{int(success)}_guardrail.mp4"

    imageio.mimsave(str(save_path), video_frames, fps=fps)
    trigger_str = f"step {stopped_at}" if stopped_at is not None else "none"
    print(f"  [video] ep {ep_idx:03d}  {'SUC' if success else 'FAIL'}  "
          f"trigger={trigger_str}  → {save_path.name}")
    return save_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", default="./attention_rollouts",
                   help="Directory with .npz rollout files")
    p.add_argument("--output_dir", default=None,
                   help="Where to save videos (default: same as data_dir)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Guardrail threshold in image pixels. "
                        "Auto-calibrated if not set.")
    p.add_argument("--anchor_steps", type=int, default=3)
    p.add_argument("--smooth_window", type=int, default=5)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--episode_idx", type=int, default=None,
                   help="Process only this episode index (default: all)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load rollouts
    rollout_files = sorted(data_dir.glob("*.npz"))
    if not rollout_files:
        print(f"[error] No .npz files found in {data_dir}")
        sys.exit(1)

    rollouts = []
    for p in rollout_files:
        d = dict(np.load(p, allow_pickle=True))
        d["path"] = p
        rollouts.append(d)
    print(f"[video] Loaded {len(rollouts)} rollouts from {data_dir}")

    # Calibrate or use provided threshold
    if args.threshold is not None:
        threshold = args.threshold
        print(f"[video] Using user-supplied threshold = {threshold:.1f} px")
    else:
        cal = calibrate_threshold(
            rollouts,
            anchor_steps=args.anchor_steps,
        )
        threshold = cal["suggested_threshold"]
        print(f"[video] Auto-calibrated threshold = {threshold:.1f} px  "
              f"(success p95={cal['success_p95']:.1f},  failure p05={cal['failure_p05']:.1f})")

    # Filter by episode_idx if requested
    if args.episode_idx is not None:
        rollouts = [ep for ep in rollouts if int(ep.get("episode_idx", -1)) == args.episode_idx]
        if not rollouts:
            print(f"[error] No rollout with episode_idx={args.episode_idx}")
            sys.exit(1)

    # Generate videos
    for ep in rollouts:
        ep_idx  = int(ep.get("episode_idx", 0))
        success = int(ep["success"])
        save_path = output_dir / f"ep{ep_idx:03d}_succ{success}_guardrail.mp4"
        make_episode_video(
            ep,
            threshold=threshold,
            anchor_steps=args.anchor_steps,
            smooth_window=args.smooth_window,
            fps=args.fps,
            save_path=save_path,
        )

    print(f"\n[done] Videos saved to {output_dir}")


if __name__ == "__main__":
    main()
