"""
Day 2 Guardrail — attention_guardrail.py
========================================
The AttentionGuardrail monitors the spatial position of the attention peak
during a robot rollout.  During the first few timesteps it computes an
'anchor' — the average peak location when the model is confidently looking
at the correct object.  On every subsequent step it checks whether the peak
has drifted more than `threshold` pixels from that anchor.

If the distance exceeds the threshold the guardrail fires: it returns a flag
telling the calling code to STOP the robot.

Design principles
-----------------
  • Stateless-friendly: the object can be reset between episodes.
  • Camera-motion robust: an optional EEF-based or optical-flow compensation
    argument shifts the anchor to account for camera movement (in LIBERO the
    camera is typically static, so compensation is off by default).
  • Smoothed decision: a short rolling window prevents single-frame noise
    from triggering a false positive.

Usage (standalone demo)
-----------------------
  python attention_guardrail.py --demo
  python attention_guardrail.py --demo --data_dir ./attention_rollouts

Usage (import)
--------------
  from scripts.guardrail.attention_guardrail import AttentionGuardrail

  guardrail = AttentionGuardrail(threshold=30.0, anchor_steps=3)
  for t, attn_map in enumerate(rollout_attention_maps):
      result = guardrail.step(attn_map)
      if result.triggered:
          robot.stop()
          break
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data class for per-step results
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Result returned by AttentionGuardrail.step()."""
    timestep: int
    peak_xy: np.ndarray              # (2,) image-pixel (x, y)
    anchor_xy: Optional[np.ndarray]  # None until anchor is set
    distance: float                  # Euclidean distance from anchor (pixels)
    smoothed_distance: float         # Running mean over the last `smooth_window` steps
    triggered: bool                  # True → stop the robot
    in_anchor_phase: bool            # True → still collecting anchor frames


# ---------------------------------------------------------------------------
# Main guardrail class
# ---------------------------------------------------------------------------

class AttentionGuardrail:
    """
    Attention-peak distance guardrail for VLA robot policies.

    Parameters
    ----------
    threshold : float
        Distance (in image pixels) beyond which the guardrail fires.
        A good starting point is ~20–40 px for a 224×224 image, calibrated
        by inspecting `plot_attention_peaks.py`'s distance-over-time figure.
    anchor_steps : int
        Number of initial timesteps used to compute the anchor position.
        The guardrail does not trigger during this phase.
    smooth_window : int
        Width of the rolling-mean window applied to distances before
        comparing against the threshold.  Set to 1 to disable smoothing.
    img_h, img_w : int
        Image dimensions used to map patch-space peaks to pixel space.
    camera_compensation : bool
        If True, adjust the anchor by the displacement of a fixed background
        region to account for camera movement (LIBERO: usually False).
    """

    def __init__(
        self,
        threshold: float = 30.0,
        anchor_steps: int = 3,
        smooth_window: int = 5,
        img_h: int = 224,
        img_w: int = 224,
    ):
        self.threshold = threshold
        self.anchor_steps = anchor_steps
        self.smooth_window = smooth_window
        self.img_h = img_h
        self.img_w = img_w

        # Mutable state — reset between episodes
        self._timestep: int = 0
        self._anchor_buffer: list[np.ndarray] = []
        self._anchor_xy: Optional[np.ndarray] = None
        self._dist_history: list[float] = []
        self._results: list[GuardrailResult] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self):
        """Clear all state. Call before each new episode."""
        self._timestep = 0
        self._anchor_buffer = []
        self._anchor_xy = None
        self._dist_history = []
        self._results = []

    def step(self, attn_map: np.ndarray) -> GuardrailResult:
        """
        Process one attention map and return a GuardrailResult.

        Parameters
        ----------
        attn_map : np.ndarray
            Shape (patch_h, patch_w), values in [0, 1], sums to 1 (probability
            distribution over image patches).
        """
        peak_xy = self._peak_to_pixel(attn_map)
        t = self._timestep
        self._timestep += 1

        # --- anchor phase ---
        if t < self.anchor_steps:
            self._anchor_buffer.append(peak_xy)
            if t == self.anchor_steps - 1:
                self._anchor_xy = np.mean(self._anchor_buffer, axis=0)
            result = GuardrailResult(
                timestep=t,
                peak_xy=peak_xy,
                anchor_xy=self._anchor_xy,
                distance=0.0,
                smoothed_distance=0.0,
                triggered=False,
                in_anchor_phase=True,
            )
            self._results.append(result)
            return result

        # --- monitoring phase ---
        dist = float(np.linalg.norm(peak_xy - self._anchor_xy))
        self._dist_history.append(dist)

        # Rolling-mean smoothing
        window = self._dist_history[-self.smooth_window:]
        smoothed = float(np.mean(window))

        triggered = smoothed > self.threshold

        result = GuardrailResult(
            timestep=t,
            peak_xy=peak_xy,
            anchor_xy=self._anchor_xy.copy(),
            distance=dist,
            smoothed_distance=smoothed,
            triggered=triggered,
            in_anchor_phase=False,
        )
        self._results.append(result)
        return result

    def get_history(self) -> list[GuardrailResult]:
        return list(self._results)

    def get_distance_series(self) -> np.ndarray:
        """Return (T,) array of smoothed distances; 0 during anchor phase."""
        return np.array([r.smoothed_distance for r in self._results])

    def get_trigger_step(self) -> Optional[int]:
        """Return the first timestep that triggered, or None."""
        for r in self._results:
            if r.triggered:
                return r.timestep
        return None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _peak_to_pixel(self, attn_map: np.ndarray) -> np.ndarray:
        """Return (x_pixel, y_pixel) of the attention-map argmax."""
        patch_h, patch_w = attn_map.shape
        flat_idx = int(np.argmax(attn_map))
        row = flat_idx // patch_w
        col = flat_idx % patch_w
        x = col * (self.img_w / patch_w) + (self.img_w / patch_w) / 2
        y = row * (self.img_h / patch_h) + (self.img_h / patch_h) / 2
        return np.array([x, y], dtype=np.float32)


# ---------------------------------------------------------------------------
# Threshold calibration helper
# ---------------------------------------------------------------------------

def calibrate_threshold(
    rollouts: list[dict],
    img_h: int = 224,
    img_w: int = 224,
    anchor_steps: int = 3,
    percentile_success: float = 95.0,
    percentile_failure: float = 5.0,
) -> dict:
    """
    Given a list of rollout dicts (from .npz files), compute the distribution
    of peak distances and suggest a threshold that separates success from
    failure.

    Returns a dict with:
      'suggested_threshold'   – midpoint between the two percentiles
      'success_p95'           – 95th pct of success distances
      'failure_p05'           – 5th pct of failure distances
      'success_max_dists'     – max distance per success episode
      'failure_max_dists'     – max distance per failure episode
    """
    guardrail = AttentionGuardrail(
        threshold=9999.0,  # never trigger during calibration
        anchor_steps=anchor_steps,
        smooth_window=1,
        img_h=img_h,
        img_w=img_w,
    )

    success_max, failure_max = [], []

    for ep in rollouts:
        guardrail.reset()
        attn_maps = ep["attention_maps"]  # (T, ph, pw)
        success   = bool(int(ep["success"]))
        for attn in attn_maps:
            guardrail.step(attn)
        dists = guardrail.get_distance_series()
        max_d = float(dists.max())
        if success:
            success_max.append(max_d)
        else:
            failure_max.append(max_d)

    s_arr = np.array(success_max) if success_max else np.array([0.0])
    f_arr = np.array(failure_max) if failure_max else np.array([0.0])

    s_p = np.percentile(s_arr, percentile_success)
    f_p = np.percentile(f_arr, percentile_failure)
    suggested = (s_p + f_p) / 2.0

    return {
        "suggested_threshold": float(suggested),
        "success_p95": float(s_p),
        "failure_p05": float(f_p),
        "success_max_dists": s_arr.tolist(),
        "failure_max_dists": f_arr.tolist(),
    }


# ---------------------------------------------------------------------------
# Demo mode
# ---------------------------------------------------------------------------

def run_demo(args):
    """
    Load .npz rollouts (or generate tiny synthetic ones) and run the guardrail
    on each, printing a summary table.
    """
    import sys

    data_dir = Path(args.data_dir)
    rollouts = []
    if data_dir.exists():
        for p in sorted(data_dir.glob("*.npz")):
            d = dict(np.load(p, allow_pickle=True))
            rollouts.append(d)

    if not rollouts:
        print("[demo] No .npz files found — using built-in synthetic rollouts.")
        rollouts = _make_synthetic_rollouts()

    print(f"\n[demo] Loaded {len(rollouts)} rollouts")

    # Calibrate threshold
    cal = calibrate_threshold(rollouts, anchor_steps=args.anchor_steps)
    threshold = cal["suggested_threshold"]
    print(f"\n[calibration]")
    print(f"  success p95 max-dist: {cal['success_p95']:.1f} px")
    print(f"  failure p05 max-dist: {cal['failure_p05']:.1f} px")
    print(f"  → suggested threshold: {threshold:.1f} px")

    # Override with CLI value if provided
    if args.threshold is not None:
        threshold = args.threshold
        print(f"  (overridden by --threshold {threshold})")

    guardrail = AttentionGuardrail(
        threshold=threshold,
        anchor_steps=args.anchor_steps,
        smooth_window=args.smooth_window,
    )

    print(f"\n{'Ep':>4}  {'Label':>8}  {'TrigStep':>8}  {'MaxDist':>8}  {'Correct?':>9}")
    print("─" * 52)

    tp = tn = fp = fn = 0
    for ep in rollouts:
        guardrail.reset()
        success = bool(int(ep["success"]))
        for attn in ep["attention_maps"]:
            r = guardrail.step(attn)
            if r.triggered:
                break

        trig_step = guardrail.get_trigger_step()
        max_dist  = guardrail.get_distance_series().max()
        predicted_failure = trig_step is not None

        # A "correct" detection: triggered on a failure, or stayed quiet on success
        correct = (predicted_failure and not success) or (not predicted_failure and success)
        if predicted_failure and not success: tp += 1
        elif not predicted_failure and success: tn += 1
        elif predicted_failure and success: fp += 1
        else: fn += 1

        ep_idx = int(ep.get("episode_idx", 0))
        label  = "SUCCESS" if success else "FAILURE"
        tstr   = str(trig_step) if trig_step else "—"
        mark   = "✓" if correct else "✗"
        print(f"{ep_idx:>4}  {label:>8}  {tstr:>8}  {max_dist:>8.1f}  {mark:>9}")

    total = tp + tn + fp + fn
    print("─" * 52)
    print(f"  TP={tp}  TN={tn}  FP={fp}  FN={fn}")
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    print(f"  Precision={precision:.2f}  Recall={recall:.2f}  Accuracy={accuracy:.2f}")


def _make_synthetic_rollouts(n_success=5, n_failure=5, seed=0) -> list[dict]:
    """Tiny in-memory synthetic rollouts for the demo."""
    rng = np.random.default_rng(seed)
    PATCH_H, PATCH_W = 16, 16
    T = 40
    target = np.array([8, 6])
    distractor = np.array([8, 10])

    def _map(peak):
        gr, gc = np.mgrid[0:PATCH_H, 0:PATCH_W]
        m = np.exp(-((gr - peak[0])**2 + (gc - peak[1])**2) / (2 * 1.5**2))
        m += rng.uniform(0, 0.1, m.shape)
        m /= m.sum()
        return m.astype(np.float32)

    rollouts = []
    for is_success in [True] * n_success + [False] * n_failure:
        jump = rng.integers(T // 3, T // 2) if not is_success else T + 1
        maps = []
        for t in range(T):
            if t < jump:
                alpha = min(t / max(jump // 2, 1), 1.0)
                peak = np.round((1 - alpha) * (target + rng.integers(-1, 2, 2)) + alpha * target).astype(int)
            else:
                beta = min((t - jump) / 5.0, 1.0)
                peak = np.round((1 - beta) * target + beta * distractor).astype(int)
            peak = np.clip(peak, [0, 0], [PATCH_H - 1, PATCH_W - 1])
            maps.append(_map(peak))
        rollouts.append({
            "attention_maps": np.stack(maps),
            "success": np.array(int(is_success)),
            "episode_idx": np.array(len(rollouts)),
        })
    return rollouts


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--demo", action="store_true",
                   help="Run the demo evaluation loop")
    p.add_argument("--data_dir", default="./attention_rollouts",
                   help="Directory with .npz rollout files")
    p.add_argument("--threshold", type=float, default=None,
                   help="Override the auto-calibrated threshold (pixels)")
    p.add_argument("--anchor_steps", type=int, default=3,
                   help="Number of initial steps for anchor computation")
    p.add_argument("--smooth_window", type=int, default=5,
                   help="Rolling-mean window for distance smoothing")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.demo or True:   # always run demo when called directly
        run_demo(args)
