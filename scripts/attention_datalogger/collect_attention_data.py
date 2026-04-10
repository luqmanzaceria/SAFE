"""
Day 1 Datalogger — collect_attention_data.py
============================================
Run OpenVLA on a LIBERO task while intercepting attention weights from every
transformer layer.  For each episode we save a .npz file containing:

  attention_maps  (T, H_patch, W_patch)   – attention peak heatmap per step
  robot_states    (T, D_state)             – proprio / EEF state
  images          (T, H_img, W_img, 3)    – raw RGB observations (uint8)
  actions         (T, 7)                  – predicted EEF delta actions
  success         ()                      – 1 = success, 0 = failure
  task_description  str                   – natural-language task string
  episode_idx       int

Usage — real OpenVLA + LIBERO
------------------------------
  python collect_attention_data.py \\
      --model_path openvla/openvla-7b \\
      --task_suite libero_10 \\
      --task_id 0 \\
      --n_success 10 \\
      --n_failure 10 \\
      --output_dir ./attention_rollouts

Usage — synthetic demo (no GPU / model required)
------------------------------------------------
  python collect_attention_data.py --mock --output_dir ./attention_rollouts

The mock mode generates 20 rollouts with realistic synthetic trajectories so
that the downstream analysis scripts can be developed and tested locally.
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Attention-hook machinery
# ---------------------------------------------------------------------------

class AttentionExtractor:
    """
    Registers forward hooks on every `nn.MultiheadAttention` (or any module
    whose name ends in 'attn') inside a model.  After each forward pass the
    captured weights are accessible via `.attention_maps`.

    For LLaMA-style models (used in OpenVLA) the transformer layers expose
    self-attention through `LlamaAttention.forward(..., output_attentions=True)`.
    We patch the model to always return attention weights and collect them here.

    Shape of captured weights per layer:
        (batch, n_heads, seq_len, seq_len)
    """

    def __init__(self, model: nn.Module, layers: Optional[list[int]] = None):
        self.model = model
        self.layers = layers  # None → use all layers
        self._hooks: list = []
        self._raw: list[torch.Tensor] = []
        self._n_image_tokens: int = 0  # set by caller before each forward
        self._image_token_start: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self):
        """Attach hooks to all attention sub-modules."""
        idx = 0
        for name, module in self.model.named_modules():
            if self._is_attention_module(module):
                if self.layers is None or idx in self.layers:
                    h = module.register_forward_hook(self._hook_fn)
                    self._hooks.append(h)
                idx += 1

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def clear(self):
        self._raw.clear()

    def get_spatial_map(
        self,
        patch_h: int,
        patch_w: int,
        query_token: int = -1,
        use_last_n_layers: int = 4,
    ) -> np.ndarray:
        """
        Build a (patch_h, patch_w) spatial attention map by:
          1. Taking the last `use_last_n_layers` captured layer maps.
          2. Averaging across layers and heads.
          3. Extracting the attention from `query_token` → image tokens.
          4. Reshaping to (patch_h, patch_w).

        Returns a float32 numpy array normalised to [0, 1].
        """
        if not self._raw:
            return np.ones((patch_h, patch_w), dtype=np.float32) / (patch_h * patch_w)

        layers = self._raw[-use_last_n_layers:]  # list of (B, H, S, S)

        # Average across layers and heads → (S, S)
        mean_attn = torch.stack(layers, dim=0).mean(dim=(0, 1, 2))  # (S, S)

        # Row for the query token (default: last token = action prediction)
        row = mean_attn[query_token]  # (S,)

        # Extract image-token slice
        i_start = self._image_token_start
        i_end = i_start + self._n_image_tokens
        if i_end > row.shape[0]:
            i_end = row.shape[0]
        img_attn = row[i_start:i_end].float().cpu().numpy()

        # Pad or trim to exactly patch_h * patch_w
        n_patches = patch_h * patch_w
        if len(img_attn) < n_patches:
            img_attn = np.pad(img_attn, (0, n_patches - len(img_attn)))
        else:
            img_attn = img_attn[:n_patches]

        # Reshape and normalise
        attn_map = img_attn.reshape(patch_h, patch_w)
        denom = attn_map.sum()
        if denom > 0:
            attn_map /= denom
        return attn_map.astype(np.float32)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _is_attention_module(module: nn.Module) -> bool:
        cls = type(module).__name__.lower()
        return "attention" in cls or isinstance(module, nn.MultiheadAttention)

    def _hook_fn(self, module, inputs, output):
        # transformers returns (hidden, attn_weights, ...) or just hidden
        if isinstance(output, tuple) and len(output) >= 2:
            attn = output[1]
            if attn is not None and isinstance(attn, torch.Tensor):
                # attn shape: (B, H, S, S) or (B, S, S) — normalise to 4-D
                if attn.ndim == 3:
                    attn = attn.unsqueeze(1)
                self._raw.append(attn.detach().float())


# ---------------------------------------------------------------------------
# Tiny wrapper that enables attention output in HuggingFace LLaMA / OpenVLA
# ---------------------------------------------------------------------------

def patch_model_for_attention(model: nn.Module):
    """
    Walk the model and flip `output_attentions=True` on any config / generation
    config we can find.  This is the cleanest way to get attention weights from
    HuggingFace transformer models without rewriting forward().
    """
    if hasattr(model, "config"):
        model.config.output_attentions = True
    if hasattr(model, "generation_config"):
        model.generation_config.output_attentions = True
    # Also patch any nested language model
    for attr in ["language_model", "model", "llm"]:
        child = getattr(model, attr, None)
        if child is not None and hasattr(child, "config"):
            child.config.output_attentions = True


# ---------------------------------------------------------------------------
# Attention-map helper (used by collect_real)
# ---------------------------------------------------------------------------

def _attentions_to_spatial_map(
    attentions,           # tuple[Tensor(1, n_heads, S, S)] — one per layer
    image_token_start: int,
    n_image_tokens: int,
    patch_h: int,
    patch_w: int,
    use_last_n_layers: int = 4,
    query_token: int = -1,
) -> np.ndarray:
    """
    Convert raw transformer attention tensors to a (patch_h, patch_w) spatial
    attention map:

      1. Take the last `use_last_n_layers` layers.
      2. Average across layers and attention heads.
      3. Extract the row for `query_token` (default: last token = action output).
      4. Slice out the image-token columns [image_token_start : +n_image_tokens].
      5. Reshape to (patch_h, patch_w) and normalise to [0, 1].

    Falls back to a uniform map when no attention tensors are available.
    """
    n_patches = patch_h * patch_w

    if attentions is None or len(attentions) == 0:
        return np.full((patch_h, patch_w), 1.0 / n_patches, dtype=np.float32)

    layers = attentions[-use_last_n_layers:]   # list of (1, H, S, S)

    # Stack → (n_layers, n_heads, S, S), then average over layers and heads
    stacked = torch.stack([a.float().squeeze(0) for a in layers], dim=0)  # (L, H, S, S)
    mean_attn = stacked.mean(dim=(0, 1))  # (S, S)

    # Row for the query token
    row = mean_attn[query_token]  # (S,)

    # Slice image tokens
    i_end = image_token_start + n_image_tokens
    img_attn = row[image_token_start : min(i_end, row.shape[0])].cpu().numpy()

    # Pad / trim to exactly n_patches
    if len(img_attn) < n_patches:
        img_attn = np.pad(img_attn, (0, n_patches - len(img_attn)))
    else:
        img_attn = img_attn[:n_patches]

    attn_map = img_attn.reshape(patch_h, patch_w).astype(np.float32)
    denom = attn_map.sum()
    if denom > 0:
        attn_map /= denom
    return attn_map


# ---------------------------------------------------------------------------
# Real OpenVLA + LIBERO collection
# ---------------------------------------------------------------------------

def collect_real(args):
    """
    Runs OpenVLA on LIBERO, intercepting attention maps at every step.

    Requires:
      - transformers >= 4.40.1  (pip install 'transformers==4.40.1')
      - libero package from  /autolab_project/LIBERO  (already installed)
      - A GPU with ~20 GB VRAM for the 7B model

    Attention extraction strategy
    ------------------------------
    We do NOT set output_attentions=True on the model config.  Doing so causes
    transformers 4.40.x to pass that flag through generate() into every decode
    step, where the causal-mask is built with an off-by-one shape (285 vs 284),
    crashing on the first call.

    Instead, per timestep we run TWO forward passes:
      1. model(**inputs, output_attentions=True)  — prefill only, no generation,
         no kv-cache iteration → causal mask is fine, attention weights captured.
      2. model.predict_action(**inputs)            — normal generation, no flags,
         returns the 7-D action vector.

    The two forward passes add ~50 % latency but are completely correct.
    """
    from PIL import Image
    from transformers import AutoModelForVision2Seq, AutoProcessor

    # LIBERO loads init_states with pickle; newer PyTorch requires weights_only=False
    _orig_torch_load = torch.load
    torch.load = lambda *a, **kw: _orig_torch_load(*a, **{**kw, "weights_only": False})

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[collect] Loading OpenVLA from '{args.model_path}' on {device} …")
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        attn_implementation="eager",  # required for explicit attention weights
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Ensure all model parameters are in the same dtype as the model's main dtype
    # to avoid "expected m1 and m2 to have the same dtype" errors during forward pass.
    model.to(torch.bfloat16 if device == "cuda" else torch.float32)
    # Do NOT call patch_model_for_attention() — setting output_attentions in the
    # config causes the causal-mask bug during generate().

    # Validate / select unnorm_key
    if hasattr(model, "norm_stats"):
        available = list(model.norm_stats.keys())
        if args.list_unnorm_keys:
            print("[unnorm_keys]", available)
            return
        if args.unnorm_key not in available:
            print(f"[warn] unnorm_key '{args.unnorm_key}' not found. Available: {available}")
            print("[warn] Falling back to 'bridge_orig'.")
            args.unnorm_key = "bridge_orig"

    # OpenVLA-7b: SigLIP 224×224 input → 16×16 = 256 image-patch tokens
    # Vision tokens begin at position 1 (position 0 = BOS).
    PATCH_H, PATCH_W = 16, 16
    IMAGE_TOKEN_START = 1
    N_IMAGE_TOKENS    = PATCH_H * PATCH_W

    # Build LIBERO environment
    env, suite, task, init_states = _build_libero_env(args.task_suite, args.task_id)
    task_description = task.language

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_init    = len(init_states)
    successes = failures = 0
    episode_idx = 0
    max_steps   = args.max_steps

    while successes < args.n_success or failures < args.n_failure:
        env.set_init_state(init_states[episode_idx % n_init])
        obs  = env.reset()
        attn_maps, robot_states, images, actions = [], [], [], []
        done = False
        t    = 0

        while not done and t < max_steps:
            image_rgb = obs["agentview_image"]  # (H, W, 3) uint8
            state_vec = obs.get("robot0_proprio-state", np.zeros(32, dtype=np.float32))

            pil_img = Image.fromarray(image_rgb)
            prompt  = f"In: What action should the robot take to {task_description}?\nOut:"
            
            # Diagnostic: print once per episode
            if t == 0:
                print(f"[debug] Task: {task_description}")
                print(f"[debug] Prompt: {prompt}")

            try:
                inputs = processor(prompt, pil_img, return_tensors="pt")
            except TypeError:
                inputs = processor(images=pil_img, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items() if hasattr(v, "to")}

            with torch.no_grad():
                # Ensure input tensors match the model's dtype (float32 on CPU, bfloat16 on GPU)
                model_dtype = torch.bfloat16 if device == "cuda" else torch.float32
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(model_dtype)

                # ── Pass 1: prefill-only forward → get attention weights ──────
                # output_attentions=True is safe here because we are NOT calling
                # generate(); there is no kv-cache decode loop, so the causal
                # mask is built once and is correctly shaped.
                fwd = model(**inputs, output_attentions=True, return_dict=True)
                attn_map = _attentions_to_spatial_map(
                    fwd.attentions,           # tuple of (1, H, S, S) per layer
                    IMAGE_TOKEN_START,
                    N_IMAGE_TOKENS,
                    PATCH_H, PATCH_W,
                    use_last_n_layers=4,
                )

                # ── Pass 2: generate action (no output_attentions flag) ───────
                # Do NOT pass attention_mask here. predict_action() appends one
                # token (29871) to input_ids before calling generate(), but the
                # processor's attention_mask is 1 token shorter.  If we pass the
                # stale mask, the Prismatic multimodal forward ends up with
                # sequence_length = N+1 but target_length = N in the causal mask,
                # causing the "size of tensor a … must match tensor b" RuntimeError.
                # Omitting attention_mask lets generate() build a fresh all-ones
                # mask from the actual (post-append) input_ids length.
                action_vec = model.predict_action(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs["pixel_values"],
                    unnorm_key=args.unnorm_key,
                    do_sample=False,
                )

            if isinstance(action_vec, torch.Tensor):
                action_vec = action_vec.cpu().numpy()
            action_vec = np.asarray(action_vec, dtype=np.float32).ravel()[:7]

            raw_mag = np.linalg.norm(action_vec[:3])

            # AMPLIFY ACTIONS for simulation (base OpenVLA is often too "quiet" for LIBERO)
            action_vec[:6] *= args.action_scale  # Scale up the 6-DOF arm movements
            
            # Diagnostic: print action magnitude every 50 steps
            if t % 50 == 0:
                mag = np.linalg.norm(action_vec[:3])
                print(f"  t={t:03d} | raw_mag={raw_mag:.6f} | scaled_mag={mag:.4f} | gripper={action_vec[6]:.2f}")

            t += 1

        success = int(info.get("success", False))

        # Skip if we already have enough of this outcome
        if success and successes >= args.n_success:
            episode_idx += 1
            continue
        if not success and failures >= args.n_failure:
            episode_idx += 1
            continue

        _save_episode(
            output_dir, episode_idx, task_description,
            np.stack(attn_maps),
            np.stack(robot_states),
            np.stack(images),
            np.stack(actions),
            success,
        )
        if success:
            successes += 1
            print(f"  ✓  success #{successes}  (t={t})")
        else:
            failures += 1
            print(f"  ✗  failure #{failures}  (t={t})")
        episode_idx += 1

    env.close()
    print(f"[collect] Done. Saved {episode_idx} episodes to {output_dir}")


def _build_libero_env(task_suite: str, task_id: int):
    """
    Create a LIBERO OffScreenRenderEnv for the given task suite and ID.

    Returns (env, suite, task, init_states).
    Note: caller must have already patched torch.load with weights_only=False
    before calling this (LIBERO init_states are pickle files).
    """
    import os
    import sys

    # Guarantee the LIBERO *project root* (not the inner libero/ source tree)
    # is on sys.path so `import libero.libero` resolves correctly.
    # The broken editable-install .pth sometimes adds LIBERO/libero/ instead of
    # LIBERO/, which makes `import libero` skip the namespace layer and breaks
    # `import libero.libero`.
    _LIBERO_ROOT = "/home/ubuntu/vlp/LIBERO"
    _LIBERO_INNER = os.path.join(_LIBERO_ROOT, "libero")
    # Remove the wrong inner path before adding the correct root
    sys.path = [p for p in sys.path if p != _LIBERO_INNER]
    if _LIBERO_ROOT not in sys.path:
        sys.path.insert(0, _LIBERO_ROOT)

    from libero.libero import benchmark, get_libero_path  # type: ignore
    from libero.libero.envs import OffScreenRenderEnv     # type: ignore

    suite_cls = benchmark.get_benchmark_dict()[task_suite]
    suite     = suite_cls()
    task      = suite.get_task(task_id)
    init_states = suite.get_task_init_states(task_id)

    bddl_root = get_libero_path("bddl_files")
    bddl_path = os.path.join(bddl_root, task.problem_folder, task.bddl_file)

    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        camera_names="agentview",
        camera_heights=224,
        camera_widths=224,
    )
    env.seed(0)
    return env, suite, task, init_states


def _parse_action(decoded: str) -> np.ndarray:
    """Fallback: parse OpenVLA's text-decoded action into a 7-D float array."""
    import ast
    try:
        start = decoded.find("[")
        end   = decoded.find("]") + 1
        action = np.array(ast.literal_eval(decoded[start:end]), dtype=np.float32)
        return action[:7] if len(action) >= 7 else np.zeros(7, dtype=np.float32)
    except Exception:
        return np.zeros(7, dtype=np.float32)


# ---------------------------------------------------------------------------
# Mock / synthetic data generator
# ---------------------------------------------------------------------------

def collect_mock(args):
    """
    Generate synthetic rollouts with realistic attention-peak behaviour:

    SUCCESS rollouts:
      Attention peak drifts smoothly toward a fixed 'target object' at
      image coordinates (target_x, target_y) and stays there.

    FAILURE rollouts ('wrong object'):
      Attention peak starts near target_x, target_y but at some mid-episode
      step it jumps to a distractor object on the other side of the image and
      stays there.
    """
    rng = np.random.default_rng(42)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    IMG_H, IMG_W = 224, 224
    PATCH_H, PATCH_W = 16, 16
    T_MAX = 60
    D_STATE = 8
    TASK_DESC = "place the red bowl on the stove (mock)"

    # Two objects in the scene
    target_patch   = np.array([8,  6])   # (row, col) in patch grid — correct bowl
    distractor_patch = np.array([8, 10])  # wrong bowl

    def _make_attn_map(peak_patch: np.ndarray, noise: float = 0.3) -> np.ndarray:
        """Gaussian bump centred at peak_patch + uniform noise."""
        grid_r, grid_c = np.mgrid[0:PATCH_H, 0:PATCH_W]
        sigma = 1.5
        bump = np.exp(-((grid_r - peak_patch[0])**2 + (grid_c - peak_patch[1])**2) / (2 * sigma**2))
        noisy = bump + rng.uniform(0, noise, bump.shape)
        noisy /= noisy.sum()
        return noisy.astype(np.float32)

    def _make_robot_state(t: int, success: bool) -> np.ndarray:
        """Simple sinusoidal EEF trajectory."""
        phase = 1.0 if success else 1.2
        x = 0.3 * np.sin(phase * t / T_MAX * np.pi)
        y = 0.1 * np.cos(phase * t / T_MAX * np.pi)
        z = 0.5 - 0.1 * (t / T_MAX)
        return np.array([x, y, z, 0, 0, 0, 0.04, 0], dtype=np.float32)

    episode_idx = 0
    for is_success in [True] * args.n_success + [False] * args.n_failure:
        T = rng.integers(T_MAX - 10, T_MAX + 1)
        jump_step = rng.integers(T // 3, T // 2) if not is_success else T + 1

        attn_maps, robot_states, images, actions = [], [], [], []

        # Smoothly interpolate peak from a random start toward the target/distractor
        start_patch = target_patch + rng.integers(-2, 3, size=2)
        for t in range(T):
            if t < jump_step:
                alpha = min(t / max(jump_step // 2, 1), 1.0)
                peak = (1 - alpha) * start_patch + alpha * target_patch
            else:
                # Jump to wrong object
                beta = min((t - jump_step) / 5.0, 1.0)
                peak = (1 - beta) * target_patch + beta * distractor_patch

            peak = np.clip(np.round(peak).astype(int),
                           [0, 0], [PATCH_H - 1, PATCH_W - 1])
            attn_maps.append(_make_attn_map(peak, noise=0.1))

            robot_states.append(_make_robot_state(t, is_success))

            # Synthetic RGB image: random background + coloured dots for objects
            img = rng.integers(60, 100, (IMG_H, IMG_W, 3), dtype=np.uint8)
            t_py = int(target_patch[0] * IMG_H / PATCH_H)
            t_px = int(target_patch[1] * IMG_W / PATCH_W)
            img[t_py-8:t_py+8, t_px-8:t_px+8] = [220, 60, 60]   # red bowl
            d_py = int(distractor_patch[0] * IMG_H / PATCH_H)
            d_px = int(distractor_patch[1] * IMG_W / PATCH_W)
            img[d_py-8:d_py+8, d_px-8:d_px+8] = [60, 120, 220]  # blue bowl
            images.append(img)

            actions.append(rng.normal(0, 0.05, 7).astype(np.float32))

        _save_episode(
            output_dir, episode_idx, TASK_DESC,
            np.stack(attn_maps),
            np.stack(robot_states),
            np.stack(images),
            np.stack(actions),
            int(is_success),
        )
        label = "success" if is_success else "failure"
        print(f"  mock ep {episode_idx:02d}  T={T}  {label}")
        episode_idx += 1

    print(f"\n[mock] Saved {episode_idx} synthetic episodes to {output_dir}")


# ---------------------------------------------------------------------------
# Shared save helper
# ---------------------------------------------------------------------------

def _save_episode(
    output_dir: Path,
    episode_idx: int,
    task_description: str,
    attention_maps: np.ndarray,  # (T, patch_h, patch_w)
    robot_states: np.ndarray,    # (T, D_state)
    images: np.ndarray,          # (T, H, W, 3) uint8
    actions: np.ndarray,         # (T, 7)
    success: int,
):
    fname = output_dir / f"ep{episode_idx:03d}_succ{success}.npz"
    np.savez_compressed(
        fname,
        attention_maps=attention_maps,
        robot_states=robot_states,
        images=images,
        actions=actions,
        success=np.array(success, dtype=np.int32),
        task_description=np.array(task_description),
        episode_idx=np.array(episode_idx, dtype=np.int32),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mock", action="store_true",
                   help="Generate synthetic data (no model / GPU required)")
    p.add_argument("--model_path", default="openvla/openvla-7b",
                   help="HuggingFace model ID or local path for OpenVLA")
    p.add_argument("--task_suite", default="libero_10",
                   help="LIBERO task suite name")
    p.add_argument("--task_id", type=int, default=0,
                   help="Task index within the suite")
    p.add_argument("--n_success", type=int, default=10,
                   help="Number of successful episodes to collect")
    p.add_argument("--n_failure", type=int, default=10,
                   help="Number of failure episodes to collect")
    p.add_argument("--max_steps", type=int, default=300,
                   help="Maximum steps per episode before declaring failure")
    p.add_argument("--unnorm_key", default="bridge_orig",
                   help="Action un-normalisation key baked into the OpenVLA checkpoint. "
                        "The base openvla/openvla-7b uses 'bridge_orig'. "
                        "A LIBERO-finetuned checkpoint may expose 'libero_10' etc. "
                        "Run with --list_unnorm_keys to see available keys.")
    p.add_argument("--list_unnorm_keys", action="store_true",
                   help="Print available unnorm_keys from the loaded model and exit")
    p.add_argument("--output_dir", default="./attention_rollouts",
                   help="Directory in which to save .npz rollout files")
    p.add_argument("--action_scale", type=float, default=5.0,
                   help="Multiplier for the arm delta actions (default: 5.0)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()
    if args.mock:
        collect_mock(args)
    else:
        collect_real(args)
    print(f"[done] elapsed {time.time() - t0:.1f}s")
