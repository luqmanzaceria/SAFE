#!/usr/bin/env python3
"""
live_attn_detector.py  —  Live Failure Detection with Temporal Attention Model
===============================================================================

Deploys the trained Temporal Attention + Hinge failure detector in a live robot
loop.  At every timestep the OpenVLA hidden state is passed through the attention
model; when the *causal attention-weighted score* crosses a threshold the robot
is immediately stopped and a recovery motion is executed.

Key differences from live_failure_detector.py (base MLP)
---------------------------------------------------------
  OLD (base MLP):  running_mean = Σ p_t / t            (uniform weights)
  NEW (attn):      score_t      = Σ w_i·p_i / Σ w_i    (learned weights)

  The attention model learns WHICH timesteps are most diagnostic of failure
  (e.g. the grasp attempt, the approach phase).  This enables:
    • Earlier detection — high-weight moments trigger the alarm sooner
    • Fewer false alarms — irrelevant timesteps are automatically down-weighted
    • Better generalisation to unseen tasks

Robot backends
--------------
  --robot libero       Simulated LIBERO environment (default, for testing)
  --robot nero         Real Nero arm via ROS service interface
  --robot widowx       WidowX arm (SAFE's real-robot setup)
  --robot ros_generic  Any ROS arm using /robot_action + /robot_obs topics

Wet-lab automation extension
-----------------------------
  This detector generalises directly to laboratory automation robots
  (liquid handlers, pipetting arms, sample sorters).  Replace the OpenVLA
  backbone with a VLM fine-tuned on lab imagery; the attention failure
  detector requires no changes.  Common lab failure modes it can catch:
    • Pipette tip not attached / wrong tip type
    • Liquid aspiration failure (bubbles, wrong depth)
    • Plate misalignment or collision
    • Reagent contamination (wrong well, cross-contamination)
    • Sample dropoff / spillage
  Set --task_prompt to the lab protocol description, e.g.:
    "Transfer 50μL from reagent A to well B3 of a 96-well plate"

Training the detector
---------------------
  1.  Run compare_with_safe.py (or best_detector.py) and add --save_attn_model:

        python scripts/compare_with_safe.py \\
            --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
            --output_dir ./compare_results --save_attn_model

  2.  This saves ./compare_results/attn_detector.pth

Live deployment — LIBERO simulation
-------------------------------------
    cd ~/vlp/openvla
    conda activate safe-openvla

    python ~/vlp/SAFE/scripts/live_attn_detector.py \\
        --robot libero \\
        --pretrained_checkpoint openvla/openvla-7b-finetuned-libero-spatial \\
        --task_suite_name libero_spatial \\
        --detector_path ~/vlp/SAFE/compare_results/attn_detector.pth \\
        --fd_threshold 0.55

Live deployment — Nero arm (real robot)
-----------------------------------------
    python ~/vlp/SAFE/scripts/live_attn_detector.py \\
        --robot nero \\
        --pretrained_checkpoint openvla/openvla-7b \\
        --task_prompt "Pick up the red vial and place it in the rack" \\
        --detector_path ~/vlp/SAFE/compare_results/attn_detector.pth \\
        --fd_threshold 0.60 \\
        --nero_host 192.168.1.100 \\
        --nero_port 8765 \\
        --num_trials 20
"""

# ── stdlib ────────────────────────────────────────────────────────────────────
import os
import sys
import abc
import json
import time
import argparse
import datetime
import pathlib
import collections
from dataclasses import dataclass, field
from typing import Optional

# ── third-party ───────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn

# ── path setup ────────────────────────────────────────────────────────────────
_THIS_FILE   = os.path.realpath(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_THIS_FILE)
_REPO_ROOT   = os.path.dirname(_SCRIPTS_DIR)
for _p in (_SCRIPTS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ═══════════════════════════════════════════════════════════════════════════════
#  Attention Failure Detector — model definition + online inference state
# ═══════════════════════════════════════════════════════════════════════════════

class _AttnModel(nn.Module):
    """
    Minimal self-contained copy of CombinedFailureDetector for deployment.
    Identical weights; no dependency on combined_detector.py at runtime.
    """
    def __init__(self, hidden_state_dim: int, task_embed_dim: int = 0,
                 hidden_dim: int = 256, n_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.task_embed_dim = task_embed_dim
        self.input_dim = hidden_state_dim + task_embed_dim

        enc, in_d = [], self.input_dim
        for _ in range(n_layers - 1):
            enc += [nn.Linear(in_d, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_d = hidden_dim
        self.encoder    = nn.Sequential(*enc)
        self.score_head = nn.Sequential(nn.Linear(in_d, 1), nn.Sigmoid())
        self.weight_head = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def step(self, h: torch.Tensor) -> tuple[float, float]:
        """
        Single-step causal inference.
        h : (D,) hidden state vector (already mean-pooled over tokens)
        Returns (raw_p, weight_w) — caller maintains running sums.
        """
        x = h.unsqueeze(0).float()      # (1, D)
        enc_out = self.encoder(x)
        p = self.score_head(enc_out).item()
        w = self.weight_head(x).item()
        return p, w


@dataclass
class AttnDetectorState:
    """
    Maintains the causal running state for online inference.
    Reset at the start of each episode.
    """
    cum_wp:   float = 0.0   # Σ w_i · p_i
    cum_w:    float = 0.0   # Σ w_i
    step:     int   = 0
    p_curve:  list  = field(default_factory=list)   # raw per-step p_t
    w_curve:  list  = field(default_factory=list)   # per-step weights w_t
    score_curve: list = field(default_factory=list) # causal aggregated score

    def update(self, p: float, w: float) -> float:
        """Feed one step; return current attention-weighted score."""
        self.cum_wp += w * p
        self.cum_w  += w
        score = self.cum_wp / (self.cum_w + 1e-8)
        self.p_curve.append(p)
        self.w_curve.append(w)
        self.score_curve.append(score)
        self.step += 1
        return score

    def reset(self):
        self.cum_wp = self.cum_w = 0.0
        self.step = 0
        self.p_curve.clear()
        self.w_curve.clear()
        self.score_curve.clear()


def load_attn_detector(path: str, device: str = "cpu") -> _AttnModel:
    """Load a saved _AttnModel / CombinedFailureDetector checkpoint."""
    ckpt = torch.load(path, map_location=device, weights_only=False)

    # Support both our checkpoint format and a raw state_dict
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state  = ckpt["model_state_dict"]
        h_dim  = ckpt.get("hidden_state_dim",
                 ckpt.get("input_dim", None))
        t_dim  = ckpt.get("task_embed_dim", 0)
        hid    = ckpt.get("hidden_dim", 256)
        layers = ckpt.get("n_layers", 2)
        drop   = ckpt.get("dropout", 0.1)
    else:
        raise ValueError(f"Unrecognised checkpoint format in {path}")

    if h_dim is None:
        # Infer from first encoder weight
        first_w = next(v for k, v in state.items() if "encoder" in k and "weight" in k)
        h_dim = first_w.shape[1]

    model = _AttnModel(h_dim, t_dim, hid, layers, drop).to(device)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f"[FD] Loaded attention detector  "
          f"({h_dim}+{t_dim}→{hid}×{layers}L  dropout={drop})")
    return model


# ═══════════════════════════════════════════════════════════════════════════════
#  Abstract Robot Interface
# ═══════════════════════════════════════════════════════════════════════════════

class RobotInterface(abc.ABC):
    """
    Abstract interface that decouples the failure detection loop from the
    specific robot backend.  Implement all abstract methods for a new robot.
    """

    @abc.abstractmethod
    def get_observation(self) -> dict:
        """
        Return current observation dict.  Must include at minimum:
          'image': np.ndarray  HxWx3  uint8 RGB
        May also include 'joint_pos', 'ee_pose', 'gripper_width', etc.
        """

    @abc.abstractmethod
    def execute_action(self, action: np.ndarray) -> tuple[dict, float, bool, dict]:
        """
        Send a 7-DoF action [dx, dy, dz, droll, dpitch, dyaw, dgripper].
        Returns (obs, reward, done, info)  — mirrors OpenAI Gym step().
        """

    @abc.abstractmethod
    def reset(self, task_id: int = 0, trial: int = 0) -> dict:
        """Reset environment / home robot.  Returns initial observation."""

    @abc.abstractmethod
    def execute_recovery(self) -> None:
        """
        Execute a safe recovery motion after a detected failure.
        Typically: lift end-effector up, open gripper, return to home.
        """

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up connections."""

    @property
    @abc.abstractmethod
    def max_steps(self) -> int:
        """Maximum steps per episode."""

    @property
    @abc.abstractmethod
    def task_name(self) -> str:
        """Human-readable task description."""


# ═══════════════════════════════════════════════════════════════════════════════
#  LIBERO Simulation Backend
# ═══════════════════════════════════════════════════════════════════════════════

class LiberoRobot(RobotInterface):
    """LIBERO off-screen simulation backend."""

    def __init__(self, task_suite_name: str, task_id: int, img_size: int = 256):
        try:
            from libero.libero import benchmark as libero_benchmark
            from libero.libero.env_wrapper import OffScreenRenderEnv
        except ImportError:
            raise SystemExit("libero not found — install LIBERO.")

        self._benchmark = libero_benchmark.get_benchmark_dict()[task_suite_name]()
        self._task_id   = task_id
        task            = self._benchmark.get_task(task_id)
        self._task_cfg  = task
        self._task_desc = task.name
        self._max_steps = self._benchmark.get_task_max_steps(task_id)
        self._init_states = self._benchmark.get_task_init_states(task_id)

        env_args = dict(
            bddl_file_name=task.bddl_file,
            camera_heights=img_size,
            camera_widths=img_size,
            camera_names="agentview",
            use_camera_obs=True,
            reward_shaping=False,
        )
        from libero.libero.env_wrapper import OffScreenRenderEnv
        self._env = OffScreenRenderEnv(**env_args)
        self._trial = 0

    def get_observation(self) -> dict:
        return self._last_obs

    def execute_action(self, action: np.ndarray):
        obs, reward, done, info = self._env.step(action)
        self._last_obs = obs
        return obs, reward, done, info

    def reset(self, task_id: int = 0, trial: int = 0) -> dict:
        self._trial   = trial
        init_state    = self._init_states[trial % len(self._init_states)]
        obs           = self._env.reset()
        self._env.set_init_state(init_state)
        self._last_obs = obs
        return obs

    def execute_recovery(self) -> None:
        lift = np.array([0., 0., 0.15, 0., 0., 0., 0.], dtype=np.float64)
        for _ in range(10):
            obs, _, done, _ = self._env.step(lift)
            self._last_obs = obs
            if done:
                break

    def close(self):
        self._env.close()

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def task_name(self) -> str:
        return self._task_desc


# ═══════════════════════════════════════════════════════════════════════════════
#  Nero Arm Backend (real robot)
# ═══════════════════════════════════════════════════════════════════════════════

class NeroArmRobot(RobotInterface):
    """
    Real Nero arm interface via a simple TCP/JSON socket protocol.

    The Nero arm controller is expected to expose a server at nero_host:nero_port
    that accepts JSON messages:
        {"cmd": "get_obs"}                    → {"image": <base64 RGB>, "joint_pos": [...]}
        {"cmd": "step", "action": [7 floats]} → {"done": bool, "success": bool}
        {"cmd": "reset"}                      → {"image": <base64 RGB>}
        {"cmd": "home"}                       → {}

    If you use ROS instead, replace the socket calls with rospy service calls.
    The abstract interface is identical; only this class needs to change.
    """

    def __init__(self, task_prompt: str, host: str = "localhost",
                 port: int = 8765, max_episode_steps: int = 400,
                 img_size: int = 256):
        import socket, base64
        self._host       = host
        self._port       = port
        self._task_desc  = task_prompt
        self._max_steps  = max_episode_steps
        self._img_size   = img_size
        self._last_obs   = None

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self._sock.connect((host, port))
            print(f"[Nero] Connected to {host}:{port}")
        except ConnectionRefusedError:
            raise SystemExit(
                f"[Nero] Cannot connect to {host}:{port}.\n"
                f"  → Start the Nero arm controller server first.\n"
                f"  → Or use --robot libero for simulation testing."
            )

    def _send_recv(self, msg: dict) -> dict:
        import json, base64
        data = (json.dumps(msg) + "\n").encode()
        self._sock.sendall(data)
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = self._sock.recv(65536)
            if not chunk:
                break
            buf += chunk
        return json.loads(buf.decode().strip())

    def _parse_obs(self, resp: dict) -> dict:
        import base64
        import cv2
        img_bytes = base64.b64decode(resp["image"])
        img_arr   = np.frombuffer(img_bytes, dtype=np.uint8)
        img_bgr   = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb   = cv2.resize(img_rgb, (self._img_size, self._img_size))
        obs = {"image": img_rgb}
        if "joint_pos" in resp:
            obs["joint_pos"] = np.array(resp["joint_pos"])
        if "ee_pose" in resp:
            obs["ee_pose"] = np.array(resp["ee_pose"])
        return obs

    def get_observation(self) -> dict:
        resp = self._send_recv({"cmd": "get_obs"})
        self._last_obs = self._parse_obs(resp)
        return self._last_obs

    def execute_action(self, action: np.ndarray):
        resp = self._send_recv({"cmd": "step",
                                "action": action.tolist()})
        done    = bool(resp.get("done", False))
        success = bool(resp.get("success", False))
        obs     = self.get_observation()
        return obs, float(success), done, {"success": success}

    def reset(self, task_id: int = 0, trial: int = 0) -> dict:
        resp = self._send_recv({"cmd": "reset",
                                "task_id": task_id, "trial": trial})
        return self._parse_obs(resp)

    def execute_recovery(self) -> None:
        """Lift end-effector, open gripper, return to safe home position."""
        # Step 1: lift straight up
        lift = np.array([0., 0., 0.15, 0., 0., 0., 0.])
        for _ in range(8):
            self._send_recv({"cmd": "step", "action": lift.tolist()})
        # Step 2: open gripper
        open_grip = np.array([0., 0., 0., 0., 0., 0., 1.])
        self._send_recv({"cmd": "step", "action": open_grip.tolist()})
        # Step 3: go home
        self._send_recv({"cmd": "home"})
        print("[Nero] Recovery complete — arm at home position")

    def close(self):
        self._send_recv({"cmd": "shutdown"})
        self._sock.close()
        print("[Nero] Connection closed")

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def task_name(self) -> str:
        return self._task_desc


# ═══════════════════════════════════════════════════════════════════════════════
#  ROS Generic Backend (any arm with ROS topics)
# ═══════════════════════════════════════════════════════════════════════════════

class ROSGenericRobot(RobotInterface):
    """
    Generic ROS backend.  Expects:
      Published topics (robot → this node):
        /robot/obs/image         sensor_msgs/Image
        /robot/obs/joint_states  sensor_msgs/JointState
      Subscribed topics (this node → robot):
        /robot/cmd/action        std_msgs/Float64MultiArray  [7-DoF]
        /robot/cmd/reset         std_msgs/Empty
        /robot/cmd/home          std_msgs/Empty
    """

    def __init__(self, task_prompt: str, max_episode_steps: int = 400):
        try:
            import rospy
            from sensor_msgs.msg import Image as ROSImage, JointState
            from std_msgs.msg import Float64MultiArray, Empty
            from cv_bridge import CvBridge
        except ImportError:
            raise SystemExit("rospy not found — source your ROS workspace.")

        import rospy
        rospy.init_node("live_attn_detector", anonymous=True)

        self._bridge     = CvBridge()
        self._task_desc  = task_prompt
        self._max_steps  = max_episode_steps
        self._last_image = None
        self._last_joint = None
        self._done       = False

        self._action_pub = rospy.Publisher("/robot/cmd/action",
                                           Float64MultiArray, queue_size=1)
        self._reset_pub  = rospy.Publisher("/robot/cmd/reset",
                                           Empty, queue_size=1)
        self._home_pub   = rospy.Publisher("/robot/cmd/home",
                                           Empty, queue_size=1)

        rospy.Subscriber("/robot/obs/image", ROSImage, self._image_cb)
        rospy.Subscriber("/robot/obs/joint_states", JointState, self._joint_cb)

        # Wait for first image
        import time
        timeout = 10.0
        t0 = time.time()
        while self._last_image is None and time.time() - t0 < timeout:
            time.sleep(0.1)
        if self._last_image is None:
            raise SystemExit("Timeout waiting for /robot/obs/image")

        print("[ROS] Connected to robot topics")

    def _image_cb(self, msg):
        import rospy
        from cv_bridge import CvBridge
        self._last_image = self._bridge.imgmsg_to_cv2(msg, "rgb8")

    def _joint_cb(self, msg):
        self._last_joint = np.array(msg.position)

    def get_observation(self) -> dict:
        obs = {"image": self._last_image.copy()}
        if self._last_joint is not None:
            obs["joint_pos"] = self._last_joint.copy()
        return obs

    def execute_action(self, action: np.ndarray):
        from std_msgs.msg import Float64MultiArray
        msg      = Float64MultiArray()
        msg.data = action.tolist()
        self._action_pub.publish(msg)
        import time; time.sleep(0.1)   # allow one control cycle
        obs      = self.get_observation()
        return obs, 0.0, False, {}

    def reset(self, task_id: int = 0, trial: int = 0) -> dict:
        from std_msgs.msg import Empty
        self._reset_pub.publish(Empty())
        import time; time.sleep(2.0)
        return self.get_observation()

    def execute_recovery(self) -> None:
        from std_msgs.msg import Float64MultiArray, Empty
        import time
        lift     = Float64MultiArray(data=[0., 0., 0.15, 0., 0., 0., 0.])
        open_g   = Float64MultiArray(data=[0., 0., 0., 0., 0., 0., 1.])
        for _ in range(8):
            self._action_pub.publish(lift)
            time.sleep(0.1)
        self._action_pub.publish(open_g)
        time.sleep(0.5)
        self._home_pub.publish(Empty())
        time.sleep(2.0)
        print("[ROS] Recovery complete")

    def close(self):
        import rospy
        rospy.signal_shutdown("live_attn_detector done")

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def task_name(self) -> str:
        return self._task_desc


# ═══════════════════════════════════════════════════════════════════════════════
#  VLA Inference Helper
# ═══════════════════════════════════════════════════════════════════════════════

def load_vla(checkpoint: str, device: str = "cuda"):
    """Load OpenVLA model and processor."""
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
        from PIL import Image as PILImage
    except ImportError:
        raise SystemExit("transformers not found — activate the openvla conda env.")

    print(f"[VLA] Loading {checkpoint} …")
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    vla.eval()
    print("[VLA] Model loaded.")
    return vla, processor


def vla_step(vla, processor, image_np: np.ndarray,
             task_name: str, device: str,
             center_crop: bool = True, unnorm_key: str = "libero_spatial"):
    """
    Run one VLA forward pass.
    Returns (action: np.ndarray [7], hidden_state: torch.Tensor [D]).
    """
    from PIL import Image as PILImage
    pil = PILImage.fromarray(image_np[..., :3].astype(np.uint8), "RGB")
    if center_crop:
        w, h     = pil.size
        min_dim  = min(w, h)
        pil      = pil.crop(((w - min_dim) // 2, (h - min_dim) // 2,
                              (w + min_dim) // 2, (h + min_dim) // 2))

    prompt  = f"In: What action should the robot take to {task_name}?\nOut:"
    inputs  = processor(pil, prompt, return_tensors="pt").to(
                  device, dtype=torch.bfloat16)

    with torch.no_grad():
        out = vla(**inputs, output_hidden_states=True)

    action = vla.predict_action(
        **inputs, unnorm_key=unnorm_key, do_sample=False
    ).cpu().numpy().flatten()

    # Last transformer layer, mean-pooled over tokens → (D,)
    hs = out.hidden_states[-1].squeeze(0).float().mean(dim=0).cpu()
    return action, hs


# ═══════════════════════════════════════════════════════════════════════════════
#  Main live detection loop
# ═══════════════════════════════════════════════════════════════════════════════

def run_live_loop(robot: RobotInterface,
                  vla, processor,
                  detector: _AttnModel,
                  args) -> dict:
    """
    Full live detection loop.
    Returns a results dict for later analysis.
    """
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir   = pathlib.Path(args.output_dir) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)

    results           = []
    total_ints        = 0
    total_true_pos    = 0
    total_false_pos   = 0

    n_trials = getattr(args, "num_trials", 20)
    if hasattr(args, "num_trials_per_task"):
        n_trials = args.num_trials_per_task

    n_tasks  = getattr(args, "n_tasks", 1)

    print(f"\n[EVAL] {n_tasks} task(s) × {n_trials} trials  "
          f"| threshold={args.fd_threshold}  "
          f"| no_recovery={'yes' if args.no_recovery else 'no'}\n")

    for task_id in range(n_tasks):
        task_successes = 0
        task_ints      = 0

        for trial in range(n_trials):
            obs     = robot.reset(task_id=task_id, trial=trial)
            fd_state = AttnDetectorState()
            detected       = False
            detection_step = None
            gt_success     = False
            t_start        = time.time()

            print(f"  Task {task_id:02d}  Trial {trial:02d}  "
                  f"— {robot.task_name[:55]}")

            for step in range(robot.max_steps):
                image = obs.get("image", obs.get("agentview_image",
                                 list(obs.values())[0]))

                # VLA forward
                action, hidden = vla_step(
                    vla, processor, image, robot.task_name,
                    device=args.device,
                    center_crop=args.center_crop,
                    unnorm_key=getattr(args, "unnorm_key",
                                       getattr(args, "task_suite_name", "libero_spatial")),
                )

                # Failure detector step
                with torch.no_grad():
                    p, w = detector.step(hidden.to(args.device))
                score = fd_state.update(p, w)

                # Threshold check
                if not detected and score >= args.fd_threshold:
                    detected       = True
                    detection_step = step
                    task_ints     += 1
                    total_ints    += 1
                    elapsed = time.time() - t_start
                    print(f"    [FD] t={step:3d}  score={score:.3f}  "
                          f"p={p:.3f}  w={w:.3f}  "
                          f"→ *** FAILURE DETECTED ***  "
                          f"(elapsed {elapsed:.1f}s)")

                    if not args.no_recovery:
                        robot.execute_recovery()
                    break

                # Environment step
                obs, reward, done, info = robot.execute_action(action)
                if done:
                    gt_success = bool(info.get("success", False))
                    break

            task_successes += int(gt_success)

            is_tp = detected and not gt_success
            is_fp = detected and gt_success
            total_true_pos  += int(is_tp)
            total_false_pos += int(is_fp)

            norm_det = detection_step / robot.max_steps if detection_step else None
            results.append({
                "task_id":        task_id,
                "trial":          trial,
                "gt_success":     int(gt_success),
                "detected":       int(detected),
                "true_positive":  int(is_tp),
                "false_positive": int(is_fp),
                "detection_step": detection_step,
                "norm_det_time":  norm_det,
                "final_score":    fd_state.score_curve[-1] if fd_state.score_curve else None,
                "score_curve":    fd_state.score_curve[:],
                "p_curve":        fd_state.p_curve[:],
                "w_curve":        fd_state.w_curve[:],
            })

        sr  = task_successes / n_trials
        fdr = task_ints       / n_trials
        print(f"    SR={sr:.0%}  FD-rate={fdr:.0%}  "
              f"(interventions={task_ints})\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    all_sr   = np.mean([r["gt_success"] for r in results])
    all_fdr  = np.mean([r["detected"]   for r in results])
    det_norm = [r["norm_det_time"] for r in results
                if r["norm_det_time"] is not None]
    precision = (total_true_pos /
                 (total_true_pos + total_false_pos + 1e-9))

    print("\n" + "=" * 60)
    print("LIVE ATTENTION FAILURE DETECTOR — SUMMARY")
    print("=" * 60)
    print(f"  Overall success rate:         {all_sr:.1%}")
    print(f"  Total interventions:          {total_ints}")
    print(f"  Intervention rate:            {all_fdr:.1%}")
    print(f"  True-positive interventions:  {total_true_pos}")
    print(f"  False-positive interventions: {total_false_pos}")
    print(f"  Intervention precision:       {precision:.1%}")
    if det_norm:
        print(f"  Mean normalised detect time:  {np.mean(det_norm):.3f} "
              f"(std {np.std(det_norm):.3f})")
    print(f"\n  Logs → {log_dir}/")
    print("=" * 60)

    # ── Save logs ─────────────────────────────────────────────────────────────
    summary = dict(
        model="temporal_attention_hinge",
        fd_threshold=args.fd_threshold,
        no_recovery=args.no_recovery,
        total_interventions=total_ints,
        true_positives=total_true_pos,
        false_positives=total_false_pos,
        precision=float(precision),
        overall_success_rate=float(all_sr),
        mean_norm_det_time=float(np.mean(det_norm)) if det_norm else None,
    )
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Save full episode data (strip curves to keep file small)
    light = [{k: v for k, v in r.items()
              if k not in ("score_curve", "p_curve", "w_curve")}
             for r in results]
    with open(log_dir / "per_episode.json", "w") as f:
        json.dump(light, f, indent=2)

    # Save score curves separately (useful for offline analysis)
    curves_path = log_dir / "score_curves.npz"
    np.savez(curves_path,
             score_curves=np.array([r["score_curve"] for r in results], dtype=object),
             p_curves=np.array([r["p_curve"]     for r in results], dtype=object),
             w_curves=np.array([r["w_curve"]     for r in results], dtype=object),
             gt_success=np.array([r["gt_success"] for r in results]),
             detected=np.array([r["detected"]    for r in results]))
    print(f"  Score curves → {curves_path}")

    return {"summary": summary, "episodes": results}


# ═══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def build_robot(args) -> RobotInterface:
    if args.robot == "libero":
        return LiberoRobot(
            task_suite_name=args.task_suite_name,
            task_id=getattr(args, "task_id", 0),
            img_size=256,
        )
    elif args.robot == "nero":
        return NeroArmRobot(
            task_prompt=args.task_prompt,
            host=args.nero_host,
            port=args.nero_port,
            max_episode_steps=getattr(args, "max_steps", 400),
        )
    elif args.robot in ("ros", "ros_generic", "widowx"):
        return ROSGenericRobot(
            task_prompt=args.task_prompt,
            max_episode_steps=getattr(args, "max_steps", 400),
        )
    else:
        raise ValueError(f"Unknown --robot backend: {args.robot!r}. "
                         f"Choose from: libero, nero, ros_generic, widowx")


def main():
    parser = argparse.ArgumentParser(
        description="Live failure detection with Temporal Attention model"
    )

    # Model
    parser.add_argument("--detector_path", required=True,
                        help="Path to attn_detector.pth (from compare_with_safe.py)")
    parser.add_argument("--pretrained_checkpoint", required=True,
                        help="OpenVLA checkpoint (HuggingFace or local path)")
    parser.add_argument("--fd_threshold",  type=float, default=0.55,
                        help="Attention score threshold to trigger intervention")
    parser.add_argument("--no_recovery",   action="store_true",
                        help="Log detections without executing recovery motion")

    # Robot backend
    parser.add_argument("--robot", default="libero",
                        choices=["libero", "nero", "ros_generic", "widowx"],
                        help="Robot backend to use")

    # LIBERO-specific
    parser.add_argument("--task_suite_name", default="libero_spatial")
    parser.add_argument("--task_id",         type=int, default=0)
    parser.add_argument("--num_trials_per_task", type=int, default=20)
    parser.add_argument("--n_tasks",         type=int, default=None,
                        help="Number of tasks to run (default: all in suite)")
    parser.add_argument("--center_crop",     type=bool, default=True)
    parser.add_argument("--unnorm_key",      default=None,
                        help="OpenVLA unnorm key (default: task_suite_name)")

    # Nero / real robot specific
    parser.add_argument("--task_prompt",     default="",
                        help="Task description for real robot deployment")
    parser.add_argument("--nero_host",       default="localhost")
    parser.add_argument("--nero_port",       type=int, default=8765)
    parser.add_argument("--max_steps",       type=int, default=400)

    # Misc
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available()
                                                     else "cpu")
    parser.add_argument("--output_dir", default="./live_attn_logs")

    args = parser.parse_args()

    if args.unnorm_key is None:
        args.unnorm_key = args.task_suite_name

    # ── Load detector ─────────────────────────────────────────────────────────
    detector = load_attn_detector(args.detector_path, device=args.device)
    detector.eval()

    # ── Load VLA ──────────────────────────────────────────────────────────────
    vla, processor = load_vla(args.pretrained_checkpoint, device=args.device)

    # ── Build robot ───────────────────────────────────────────────────────────
    robot = build_robot(args)

    if args.n_tasks is None:
        if args.robot == "libero":
            from libero.libero import benchmark as libero_benchmark
            bench = libero_benchmark.get_benchmark_dict()[args.task_suite_name]()
            args.n_tasks = bench.n_tasks
        else:
            args.n_tasks = 1

    # ── Run ───────────────────────────────────────────────────────────────────
    try:
        run_live_loop(robot, vla, processor, detector, args)
    finally:
        robot.close()


if __name__ == "__main__":
    main()
