#!/usr/bin/env python3
"""
Live failure detector for OpenVLA on LIBERO.

Runs the standard OpenVLA→LIBERO evaluation loop, but at every timestep feeds
the VLA's hidden states into a pre-trained MLP failure detector.  When the
running-mean failure score crosses a threshold the episode is *immediately
interrupted*:
  1. The robot arm lifts straight up (pure +z delta actions).
  2. The episode is marked as a detected-failure and the env is reset.

The script mirrors the CLI of openvla/experiments/robot/libero/run_libero_eval.py
so you can swap it in directly; it adds three extra flags:
  --detector_path     path to detector.pth saved by failure_detector.py
  --fd_threshold      failure score threshold that triggers intervention (0-1)
  --lift_steps        how many +z steps to execute before resetting (default 10)
  --lift_delta_z      magnitude of each upward delta (default 0.15 m)
  --no_recovery       if set, just log the detection without lifting/resetting

Usage
-----
First train and save the detector:

    python scripts/failure_detector.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
        --output_dir ./results_spatial \\
        --save_model

Then run live:

    cd ~/vlp/openvla
    conda activate safe-openvla

    python ~/vlp/SAFE/scripts/live_failure_detector.py \\
        --model_family openvla \\
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \\
        --task_suite_name libero_spatial \\
        --center_crop True \\
        --output_hidden_states True \\
        --run_id_note live-fd \\
        --use_wandb False \\
        --save_logs True \\
        --num_trials_per_task 10 \\
        --detector_path ~/vlp/SAFE/results_spatial/detector.pth \\
        --fd_threshold 0.6
"""

# ── std-lib & third-party ───────────────────────────────────────────────────
import os
import sys
import time
import json
import argparse
import datetime
import pathlib
import collections
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

# ── openvla / LIBERO imports ────────────────────────────────────────────────
# These are available when running inside the openvla conda env.
# The script expects to be run with cwd = ~/vlp/openvla (same as run_libero_eval.py)

try:
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from PIL import Image as PILImage
except ImportError:
    raise SystemExit("transformers not found — activate the openvla conda env.")

try:
    from libero.libero import benchmark as libero_benchmark
    from libero.libero.env_wrapper import OffScreenRenderEnv
except ImportError:
    raise SystemExit("libero not found — install LIBERO or set PYTHONPATH.")

try:
    import draccus
except ImportError:
    raise SystemExit("draccus not found — pip install draccus")


# ──────────────────────────────── Failure detector ─────────────────────────
# Minimal re-definition so the script is self-contained (no import from
# failure_detector.py which lives in the SAFE repo root, not on sys.path).

class FailureDetector(nn.Module):
    """Identical architecture to scripts/failure_detector.py FailureDetector."""
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_layers: int = 2):
        super().__init__()
        layers, in_d = [], input_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
            in_d = hidden_dim
        layers += [nn.Linear(in_d, 1), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """x: (1, D)  →  scalar in [0, 1]"""
        return self.mlp(x).squeeze(-1)  # (1,)


def load_detector(path: str, device: str = "cpu") -> FailureDetector:
    ckpt = torch.load(path, map_location=device)
    det  = FailureDetector(
        input_dim  = ckpt["input_dim"],
        hidden_dim = ckpt["hidden_dim"],
        n_layers   = ckpt["n_layers"],
    ).to(device)
    det.load_state_dict(ckpt["model_state_dict"])
    det.eval()
    print(f"[FD] Loaded detector  ({ckpt['input_dim']}-d → "
          f"{ckpt['hidden_dim']} hidden × {ckpt['n_layers']} layers)")
    return det


@torch.no_grad()
def detector_step(det: FailureDetector,
                  hidden_state: torch.Tensor,
                  running_sum: float,
                  step_idx: int,
                  device: str) -> tuple[float, float]:
    """
    Feed a single hidden-state vector through the detector.

    hidden_state : (N_tokens, D)  or  (D,)
    Returns (raw_prob, running_mean_score).
    """
    h = hidden_state.float()
    if h.ndim == 2:          # (N_tokens, D) → mean over tokens
        h = h.mean(dim=0)
    h = h.unsqueeze(0).to(device)   # (1, D)

    raw_prob = det.forward_raw(h).item()
    running_sum   = running_sum + raw_prob
    running_mean  = running_sum / (step_idx + 1)
    return raw_prob, running_mean, running_sum


# ──────────────────────────────── Recovery action ──────────────────────────

def execute_recovery(env, lift_steps: int = 10, lift_delta_z: float = 0.15):
    """
    Execute a pure +z recovery motion to lift the end-effector straight up,
    then return the final observation.
    """
    lift_action = np.array(
        [0.0, 0.0, lift_delta_z,   # dx, dy, dz
         0.0, 0.0, 0.0,            # droll, dpitch, dyaw
         0.0],                     # gripper (keep current)
        dtype=np.float64,
    )
    obs = None
    for _ in range(lift_steps):
        obs, _, done, _ = env.step(lift_action)
        if done:
            break
    return obs


# ──────────────────────────────── Eval config (mirrors run_libero_eval) ────

@dataclass
class EvalConfig:
    # VLA
    model_family: str = "openvla"
    pretrained_checkpoint: str = ""
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    center_crop: bool = True

    # LIBERO
    task_suite_name: str = "libero_spatial"
    num_trials_per_task: int = 20
    run_id_note: str = "live-fd"
    output_hidden_states: bool = True

    # Logging
    use_wandb: bool = False
    wandb_project: str = "vla-safe"
    save_logs: bool = True
    save_videos: bool = False

    # Failure detector
    detector_path: str = ""
    fd_threshold:  float = 0.6
    lift_steps:    int   = 10
    lift_delta_z:  float = 0.15
    no_recovery:   bool  = False


# ──────────────────────────────── Main ─────────────────────────────────────

def _make_env(task_cfg, img_size: int = 256):
    env_args = {
        "bddl_file_name":  task_cfg.bddl_file,
        "camera_heights":  img_size,
        "camera_widths":   img_size,
        "camera_names":    "agentview",
        "use_camera_obs":  True,
        "reward_shaping":  False,
    }
    return OffScreenRenderEnv(**env_args)


def _to_pil(obs_img: np.ndarray) -> PILImage.Image:
    """Convert HWC uint8 numpy → PIL (RGB)."""
    if obs_img.max() <= 1.0:
        obs_img = (obs_img * 255).astype(np.uint8)
    return PILImage.fromarray(obs_img[..., :3], "RGB")


@draccus.wrap()
def main(cfg: EvalConfig) -> None:
    device_vla = "cuda" if torch.cuda.is_available() else "cpu"
    device_fd  = device_vla

    # ── Load failure detector ───────────────────────────────────────────────
    if not cfg.detector_path:
        raise ValueError("--detector_path must be set to a detector.pth file")
    detector = load_detector(cfg.detector_path, device=device_fd)

    # ── Load VLA ────────────────────────────────────────────────────────────
    print(f"[VLA] Loading {cfg.pretrained_checkpoint} ...")
    processor = AutoProcessor.from_pretrained(
        cfg.pretrained_checkpoint, trust_remote_code=True
    )
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device_vla)
    vla.eval()
    print("[VLA] Model loaded.")

    # ── LIBERO benchmark ────────────────────────────────────────────────────
    benchmark  = libero_benchmark.get_benchmark_dict()[cfg.task_suite_name]()
    n_tasks    = benchmark.n_tasks
    task_suite_name = cfg.task_suite_name

    # Results accumulator
    results = collections.defaultdict(list)   # task_id → list of dicts
    total_interventions = 0

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = pathlib.Path(f"./live_fd_logs/{task_suite_name}/{timestamp}")
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[EVAL] {n_tasks} tasks × {cfg.num_trials_per_task} trials "
          f"| threshold={cfg.fd_threshold} | lift_steps={cfg.lift_steps}\n")

    for task_id in range(n_tasks):
        task       = benchmark.get_task(task_id)
        task_name  = task.name
        init_states = benchmark.get_task_init_states(task_id)
        env        = _make_env(task)

        task_successes = 0
        task_fd_hits   = 0   # episodes where detector fired before true failure

        print(f"  Task {task_id:02d}: {task_name}")

        for trial in range(cfg.num_trials_per_task):
            # ── Reset ───────────────────────────────────────────────────────
            init_state = init_states[trial % len(init_states)]
            obs        = env.reset()
            env.set_init_state(init_state)

            # Detector state
            running_sum  = 0.0
            score_curve  = []
            raw_curve    = []
            detected     = False
            detection_step = None
            gt_success   = False
            frames       = []   # optional video

            for step in range(benchmark.get_task_max_steps(task_id)):
                # ── Observation ─────────────────────────────────────────────
                img = obs["agentview_image"]
                pil = _to_pil(img)
                if cfg.center_crop:
                    w, h = pil.size
                    min_dim = min(w, h)
                    pil = pil.crop(((w - min_dim) // 2, (h - min_dim) // 2,
                                    (w + min_dim) // 2, (h + min_dim) // 2))

                # ── VLA forward ─────────────────────────────────────────────
                inputs = processor(pil, f"In: What action should the robot take to {task_name}?\nOut:", return_tensors="pt").to(device_vla, dtype=torch.bfloat16)
                with torch.no_grad():
                    out = vla(
                        **inputs,
                        output_hidden_states=cfg.output_hidden_states,
                    )
                # Decode action tokens → 7-DoF action
                action = vla.predict_action(
                    **inputs,
                    unnorm_key=task_suite_name,
                    do_sample=False,
                ).cpu().numpy().flatten()

                # ── Failure detection ────────────────────────────────────────
                hs = out.hidden_states[-1].squeeze(0).cpu()   # (N_tokens, D)
                raw_p, running_mean, running_sum = detector_step(
                    detector, hs, running_sum, step, device_fd
                )
                score_curve.append(running_mean)
                raw_curve.append(raw_p)

                if cfg.save_videos:
                    frames.append(img[..., :3].copy())

                # ── Intervention ─────────────────────────────────────────────
                if not detected and running_mean >= cfg.fd_threshold:
                    detected       = True
                    detection_step = step
                    task_fd_hits  += 1
                    total_interventions += 1
                    print(f"    [FD] t={step:3d}  score={running_mean:.3f}  "
                          f"→ FAILURE DETECTED  trial={trial}")

                    if not cfg.no_recovery:
                        execute_recovery(env, cfg.lift_steps, cfg.lift_delta_z)
                    break

                # ── Environment step ────────────────────────────────────────
                obs, reward, done, info = env.step(action)
                if done:
                    gt_success = bool(info.get("success", False))
                    break

            task_successes += int(gt_success)

            results[task_id].append({
                "trial":          trial,
                "gt_success":     gt_success,
                "detected":       detected,
                "detection_step": detection_step,
                "final_score":    score_curve[-1] if score_curve else None,
                "score_curve":    score_curve,
                "raw_curve":      raw_curve,
            })

        env.close()

        sr  = task_successes / cfg.num_trials_per_task
        fdr = task_fd_hits   / cfg.num_trials_per_task
        print(f"    SR={sr:.0%}  FD-rate={fdr:.0%}  "
              f"(interventions={task_fd_hits})")

    # ── Save results ────────────────────────────────────────────────────────
    summary = {
        "task_suite":         task_suite_name,
        "fd_threshold":       cfg.fd_threshold,
        "lift_steps":         cfg.lift_steps,
        "lift_delta_z":       cfg.lift_delta_z,
        "no_recovery":        cfg.no_recovery,
        "total_interventions": total_interventions,
        "tasks": {}
    }

    for task_id, trials in results.items():
        sr  = np.mean([t["gt_success"] for t in trials])
        fdr = np.mean([t["detected"]   for t in trials])
        # True positive rate: detected AND actually failed
        tp  = [t for t in trials if t["detected"] and not t["gt_success"]]
        fp  = [t for t in trials if t["detected"] and     t["gt_success"]]
        det_steps = [t["detection_step"] for t in trials if t["detection_step"] is not None]
        summary["tasks"][task_id] = {
            "success_rate":   float(sr),
            "fd_rate":        float(fdr),
            "true_positives": len(tp),
            "false_positives": len(fp),
            "mean_detection_step": float(np.mean(det_steps)) if det_steps else None,
        }

    summary_path = log_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Per-episode detail (without full curves to keep file small)
    detail = {tid: [{k: v for k, v in t.items() if k not in ("score_curve", "raw_curve")}
                    for t in trials]
              for tid, trials in results.items()}
    with open(log_dir / "per_episode.json", "w") as f:
        json.dump(detail, f, indent=2)

    # ── Print summary ───────────────────────────────────────────────────────
    all_trials   = [t for ts in results.values() for t in ts]
    overall_sr   = np.mean([t["gt_success"] for t in all_trials])
    overall_fdr  = np.mean([t["detected"]   for t in all_trials])
    tp_all       = [t for t in all_trials if t["detected"] and not t["gt_success"]]
    fp_all       = [t for t in all_trials if t["detected"] and     t["gt_success"]]
    det_steps_all = [t["detection_step"] for t in all_trials if t["detection_step"] is not None]

    print("\n" + "=" * 60)
    print("LIVE FAILURE DETECTOR — SUMMARY")
    print("=" * 60)
    print(f"  Overall success rate:       {overall_sr:.1%}")
    print(f"  Total interventions:        {total_interventions}")
    print(f"  Intervention rate:          {overall_fdr:.1%}")
    if det_steps_all:
        print(f"  Mean detection timestep:    {np.mean(det_steps_all):.1f}  "
              f"(std {np.std(det_steps_all):.1f})")
    print(f"  True-positive interventions:  {len(tp_all)}")
    print(f"  False-positive interventions: {len(fp_all)}")
    if tp_all or fp_all:
        prec = len(tp_all) / (len(tp_all) + len(fp_all) + 1e-9)
        print(f"  Intervention precision:     {prec:.1%}")
    print(f"\n  Logs saved to: {log_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
