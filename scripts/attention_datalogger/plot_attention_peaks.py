"""
Day 1 Visualisation — plot_attention_peaks.py
=============================================
Load the .npz rollout files saved by collect_attention_data.py, extract the
attention peak (argmax) at each timestep, and visualise the resulting 2D
trajectory overlaid on the first-frame image.

What you will see
-----------------
  • One subplot per episode (or a compact grid).
  • SUCCESS rollouts: the peak path is drawn in green and stays near a single
    object throughout the episode.
  • FAILURE rollouts:  the peak path is drawn in red and shows a clear JUMP
    to the wrong object partway through the episode — this is the smoking-gun
    signal for the guardrail.

A second figure shows all trajectories overlaid on a single axes so the
separation between the two populations is immediately obvious.

Usage
-----
  python plot_attention_peaks.py --data_dir ./attention_rollouts
  python plot_attention_peaks.py --data_dir ./attention_rollouts --save_dir ./figures
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless-safe
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from scipy.ndimage import zoom


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_rollouts(data_dir: Path) -> list[dict]:
    """Load every .npz file in data_dir and return list of episode dicts."""
    rollouts = []
    for path in sorted(data_dir.glob("*.npz")):
        d = dict(np.load(path, allow_pickle=True))
        d["path"] = path
        rollouts.append(d)
    if not rollouts:
        raise FileNotFoundError(f"No .npz files found in {data_dir}")
    print(f"Loaded {len(rollouts)} rollouts from {data_dir}")
    return rollouts


def attention_peak(attn_map: np.ndarray) -> tuple[float, float]:
    """
    Return the (x, y) image-space coordinates of the maximum attention weight.

    attn_map: (patch_h, patch_w) float32
    Returns (x, y) in *image pixel* coordinates assuming the map is scaled to
    the first image's (H, W).  Caller must pass img_h / img_w for scaling.
    """
    flat = attn_map.flatten()
    idx = int(np.argmax(flat))
    patch_h, patch_w = attn_map.shape
    row = idx // patch_w
    col = idx % patch_w
    return row, col  # patch-space coords; caller up-scales if needed


def peaks_for_rollout(
    attn_maps: np.ndarray,   # (T, patch_h, patch_w)
    img_h: int,
    img_w: int,
) -> np.ndarray:
    """
    Compute per-timestep attention peaks in image-pixel space.
    Returns (T, 2) array of (x_pixel, y_pixel).
    """
    patch_h, patch_w = attn_maps.shape[1], attn_maps.shape[2]
    scale_y = img_h / patch_h
    scale_x = img_w / patch_w
    coords = []
    for t in range(len(attn_maps)):
        row, col = attention_peak(attn_maps[t])
        coords.append([col * scale_x, row * scale_y])   # (x, y) convention
    return np.array(coords, dtype=np.float32)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _add_coloured_path(
    ax: plt.Axes,
    xy: np.ndarray,        # (T, 2)
    success: bool,
    alpha_base: float = 0.85,
):
    """
    Draw the trajectory as a colour-coded line that fades from light to dark
    (time progression) with a distinct style for success vs failure.
    """
    T = len(xy)
    cmap = cm.Greens if success else cm.Reds
    colours = cmap(np.linspace(0.4, 0.95, max(T - 1, 1)))

    for t in range(T - 1):
        ax.plot(
            xy[t : t + 2, 0],
            xy[t : t + 2, 1],
            color=colours[t],
            linewidth=1.8,
            alpha=alpha_base,
            solid_capstyle="round",
        )

    # Start marker (circle) and end marker (star)
    ax.scatter(*xy[0],  s=60,  c="white", edgecolors="k", zorder=5, linewidths=1)
    ax.scatter(*xy[-1], s=100, c=colours[-1], marker="*", edgecolors="k", zorder=5, linewidths=0.8)


def _upscale_attn(attn_map: np.ndarray, img_h: int, img_w: int) -> np.ndarray:
    """Bilinear zoom of a patch-space map to image resolution."""
    sy = img_h / attn_map.shape[0]
    sx = img_w / attn_map.shape[1]
    return zoom(attn_map, (sy, sx), order=1)


def plot_per_episode_grid(
    rollouts: list[dict],
    save_path: Path | None = None,
):
    """
    One panel per rollout showing: first-frame image, aggregated attention
    heatmap, and the peak trajectory.
    """
    n = len(rollouts)
    ncols = min(5, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(ncols * 3.2, nrows * 3.5),
        dpi=120,
    )
    axes = np.array(axes).reshape(nrows, ncols)

    for i, ep in enumerate(rollouts):
        ax = axes[i // ncols, i % ncols]

        attn_maps = ep["attention_maps"]  # (T, ph, pw)
        images    = ep["images"]          # (T, H, W, 3)
        success   = bool(int(ep["success"]))

        img_h, img_w = images.shape[1], images.shape[2]
        first_frame = images[0]

        # --- background: first frame ---
        ax.imshow(first_frame, origin="upper")

        # --- attention heatmap overlay (average across time) ---
        mean_attn = attn_maps.mean(axis=0)
        heatmap   = _upscale_attn(mean_attn, img_h, img_w)
        ax.imshow(heatmap, origin="upper", cmap="hot", alpha=0.45,
                  vmin=0, vmax=heatmap.max())

        # --- peak trajectory ---
        xy = peaks_for_rollout(attn_maps, img_h, img_w)
        _add_coloured_path(ax, xy, success)

        label = "SUCCESS" if success else "FAILURE"
        colour = "#2ca02c" if success else "#d62728"
        ep_idx = int(ep.get("episode_idx", i))
        ax.set_title(f"Ep {ep_idx} — {label}", color=colour, fontsize=8, fontweight="bold")
        ax.axis("off")

    # Hide unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle("Attention-Peak Trajectories per Episode\n"
                 "(circle = start,  ★ = end;  green = success,  red = failure)",
                 fontsize=10)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  [fig] saved per-episode grid → {save_path}")
    return fig


def plot_overlaid(
    rollouts: list[dict],
    save_path: Path | None = None,
):
    """
    All trajectories overlaid on a shared axes.
    Uses the first frame of the first rollout as background.
    """
    images_0 = rollouts[0]["images"]
    img_h, img_w = images_0.shape[1], images_0.shape[2]
    first_frame = images_0[0]

    fig, ax = plt.subplots(figsize=(7, 7), dpi=140)
    ax.imshow(first_frame, origin="upper")

    success_patches, fail_patches = [], []
    for ep in rollouts:
        attn_maps = ep["attention_maps"]
        success   = bool(int(ep["success"]))
        xy = peaks_for_rollout(attn_maps, img_h, img_w)
        _add_coloured_path(ax, xy, success, alpha_base=0.6)
        if success:
            success_patches.append(xy)
        else:
            fail_patches.append(xy)

    # Legend proxies
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="green", lw=2, label=f"Success (n={len(success_patches)})"),
        Line2D([0], [0], color="red",   lw=2, label=f"Failure (n={len(fail_patches)})"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title("All Attention-Peak Trajectories (overlaid)\n"
                 "Note how failure paths jump to the wrong object mid-episode",
                 fontsize=10)
    ax.axis("off")
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  [fig] saved overlaid figure → {save_path}")
    return fig


def plot_peak_distance_over_time(
    rollouts: list[dict],
    save_path: Path | None = None,
):
    """
    For each rollout, plot the Euclidean distance of the attention peak from
    the episode's own anchor (mean of first 3 steps).  This is the exact
    signal the Day-2 guardrail uses.
    """
    fig, ax = plt.subplots(figsize=(9, 4), dpi=130)

    success_count, fail_count = 0, 0
    for ep in rollouts:
        attn_maps = ep["attention_maps"]
        images    = ep["images"]
        success   = bool(int(ep["success"]))
        img_h, img_w = images.shape[1], images.shape[2]

        xy = peaks_for_rollout(attn_maps, img_h, img_w)  # (T, 2)
        anchor = xy[:3].mean(axis=0)
        dists  = np.linalg.norm(xy - anchor, axis=1)     # (T,)
        t      = np.arange(len(dists))

        c = "#2ca02c" if success else "#d62728"
        lw = 1.5
        label = None
        if success and success_count == 0:
            label = "Success"
            success_count += 1
        elif not success and fail_count == 0:
            label = "Failure"
            fail_count += 1
        ax.plot(t, dists, color=c, alpha=0.55, linewidth=lw, label=label)

    ax.set_xlabel("Timestep $t$", fontsize=11)
    ax.set_ylabel("Distance from anchor (px)", fontsize=11)
    ax.set_title("Attention-Peak Distance from Anchor over Time\n"
                 "Guardrail threshold should sit between the two populations",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  [fig] saved distance-over-time → {save_path}")
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--data_dir", default="./attention_rollouts",
                   help="Directory with .npz rollout files from collect_attention_data.py")
    p.add_argument("--save_dir", default=None,
                   help="Save figures here (default: same as data_dir)")
    p.add_argument("--show", action="store_true",
                   help="Also show figures interactively (requires a display)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    save_dir = Path(args.save_dir) if args.save_dir else data_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    rollouts = load_rollouts(data_dir)

    print("[plot] Generating per-episode grid …")
    fig1 = plot_per_episode_grid(rollouts, save_dir / "attention_peak_grid.png")

    print("[plot] Generating overlaid trajectory figure …")
    fig2 = plot_overlaid(rollouts, save_dir / "attention_peak_overlaid.png")

    print("[plot] Generating distance-over-time figure …")
    fig3 = plot_peak_distance_over_time(rollouts, save_dir / "attention_peak_distance.png")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()
    else:
        plt.close("all")

    print(f"[done] All figures saved to {save_dir}")


if __name__ == "__main__":
    main()
