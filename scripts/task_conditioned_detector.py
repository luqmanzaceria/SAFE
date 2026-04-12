#!/usr/bin/env python3
"""
Task-Conditioned Failure Detector
==================================

The base failure detector sees only the VLA hidden states and has no
knowledge of *which task* is being executed.  Different tasks have very
different failure signatures (e.g. "open drawer" fails differently from
"pick up cup"), so feeding the task description as extra context should
help the model learn task-specific decision boundaries.

Architecture
------------
The task description is encoded with TF-IDF + truncated SVD (LSA) into a
compact vector (no internet or GPU needed).  This task embedding is
concatenated with the mean-pooled hidden state before the MLP:

    input  =  [ hidden_state (D)  ||  task_embed (E) ]   dim = D + E
    → same MLP as base detector

This forces the model to learn: "for *this* task, what does a failure look
like in the hidden state?"

Training strategy
-----------------
Same BCE loss with pos_weight as the base detector, but with the enriched
input.  We also train a baseline (no task conditioning) under identical
conditions to show the gain.

Outputs (in output_dir/)
------------------------
  task_embed_pca.png      - 2-D PCA of the task embeddings (one dot per task)
  comparison_auc.png      - bar chart: base AUC vs task-conditioned AUC per task
  confusion_matrix.png    - confusion matrix (task-conditioned, test set)
  summary.txt             - AUC comparison table

Usage
-----
    python scripts/task_conditioned_detector.py \\
        --data_path ~/vlp/openvla/rollouts/single-foward/libero_spatial/ \\
        --output_dir ./task_cond_results
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import roc_auc_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from failure_detector import (
    load_rollouts,
    RolloutDataset,
    _pad_collate,
    FailureDetector,
    _compute_loss,
    predict,
    train_model,
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ──────────────────────────── Task encoder ─────────────────────────────────

class TaskEncoder:
    """
    Encodes task description strings to fixed-length vectors using TF-IDF
    over word n-grams followed by truncated SVD (Latent Semantic Analysis).

    This is intentionally lightweight — it runs on CPU with no external
    model downloads required.  For better quality you could swap in a
    sentence-transformer.
    """
    def __init__(self, n_components: int = 32):
        self.n_components = n_components
        self.tfidf = TfidfVectorizer(
            ngram_range=(1, 2),
            analyzer="word",
            strip_accents="unicode",
            lowercase=True,
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=0)
        self.fitted = False

    def fit(self, descriptions: list[str]):
        tfidf_mat = self.tfidf.fit_transform(descriptions)
        self.svd.fit(tfidf_mat)
        self.fitted = True
        var = self.svd.explained_variance_ratio_.sum()
        print(f"  [TaskEncoder] {self.n_components}-d LSA embedding "
              f"explains {var:.1%} of TF-IDF variance")

    def transform(self, descriptions: list[str]) -> np.ndarray:
        assert self.fitted, "Call fit() first"
        tfidf_mat = self.tfidf.transform(descriptions)
        return self.svd.transform(tfidf_mat).astype(np.float32)

    def encode_one(self, description: str) -> np.ndarray:
        return self.transform([description])[0]


# ──────────────────────────── Dataset ──────────────────────────────────────

class TaskCondDataset(Dataset):
    """
    Like RolloutDataset but each sample also carries a task embedding vector.
    """
    def __init__(self, rollouts: list, task_embeds: np.ndarray):
        self.rollouts     = rollouts
        self.task_embeds  = torch.from_numpy(task_embeds)   # (N, E)

    def __len__(self):
        return len(self.rollouts)

    def __getitem__(self, idx):
        r  = self.rollouts[idx]
        te = self.task_embeds[idx]                          # (E,)
        return {
            "features":   r.hidden_states,                 # (T, D)
            "task_embed": te,
            "label":      torch.tensor(float(r.episode_success)),
        }


def _pad_collate_task(batch):
    """Pad variable-length rollouts; keep task embedding per-sample."""
    max_len = max(b["features"].shape[0] for b in batch)
    D  = batch[0]["features"].shape[-1]
    E  = batch[0]["task_embed"].shape[-1]
    B  = len(batch)

    features   = torch.zeros(B, max_len, D)
    task_embs  = torch.zeros(B, E)
    masks      = torch.zeros(B, max_len)
    labels     = torch.zeros(B)

    for i, b in enumerate(batch):
        T = b["features"].shape[0]
        features[i, :T] = b["features"]
        masks[i, :T]    = 1.0
        task_embs[i]    = b["task_embed"]
        labels[i]       = b["label"]

    return {
        "features":      features,
        "task_embeds":   task_embs,
        "valid_masks":   masks,
        "success_labels": labels,
    }


# ──────────────────────────── Model ────────────────────────────────────────

class TaskCondFailureDetector(nn.Module):
    """
    Failure detector conditioned on a task description embedding.

    At each timestep t the input to the MLP is:
        [ hidden_state_t  ||  task_embed ]    shape: (D + E,)

    The task embedding is the same for every step in the episode, but
    the MLP learns to use it to shift its decision boundary per task.
    """
    def __init__(self, hidden_state_dim: int, task_embed_dim: int,
                 hidden_dim: int = 256, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        input_dim = hidden_state_dim + task_embed_dim
        layers, in_d = [], input_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_d, hidden_dim), nn.ReLU(),
                       nn.Dropout(dropout)]
            in_d = hidden_dim
        layers += [nn.Linear(in_d, 1), nn.Sigmoid()]
        self.mlp = nn.Sequential(*layers)

    def _expand_task(self, features: torch.Tensor,
                     task_embeds: torch.Tensor) -> torch.Tensor:
        """
        Expand task_embeds from (B, E) to (B, T, E) and concat with
        features (B, T, D) → (B, T, D+E).
        """
        te = task_embeds.unsqueeze(1).expand(-1, features.shape[1], -1)
        return torch.cat([features, te], dim=-1)

    def forward_raw(self, features: torch.Tensor,
                    task_embeds: torch.Tensor) -> torch.Tensor:
        """Per-step failure probability (B, T)."""
        x = self._expand_task(features, task_embeds)
        return self.mlp(x).squeeze(-1)

    def forward(self, features: torch.Tensor,
                task_embeds: torch.Tensor,
                valid_masks: torch.Tensor) -> torch.Tensor:
        """Running-mean failure score (B, T) — for inference."""
        raw  = self.forward_raw(features, task_embeds)
        cum  = torch.cumsum(raw, dim=-1)
        t    = torch.arange(1, raw.shape[1] + 1,
                             device=raw.device).float()
        return (cum / t.unsqueeze(0)) * valid_masks


def _train_task_cond(model, rollouts, task_embeds, n_epochs=300, lr=1e-3,
                     lambda_reg=1e-2, batch_size=32, device="cpu"):
    dataset = TaskCondDataset(rollouts, task_embeds)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         collate_fn=_pad_collate_task)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)
    model.train()
    pbar = tqdm(range(n_epochs), desc="Training (task-cond)", unit="ep")
    for _ in pbar:
        epoch_loss = 0.0
        for batch in loader:
            feat = batch["features"].to(device)
            te   = batch["task_embeds"].to(device)
            mask = batch["valid_masks"].to(device)
            lbl  = batch["success_labels"].to(device)

            opt.zero_grad()
            raw  = model.forward_raw(feat, te)
            loss = _compute_loss(raw, mask, lbl, lambda_reg, model)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()

        pbar.set_description(f"Loss {epoch_loss / len(loader):.4f}")
        sched.step()


@torch.no_grad()
def _predict_task_cond(model, rollouts, task_embeds, device="cpu"):
    model.eval()
    out = []
    for i, r in enumerate(rollouts):
        feat = r.hidden_states.unsqueeze(0).to(device)
        te   = torch.from_numpy(task_embeds[i:i+1]).to(device)
        mask = torch.ones(1, feat.shape[1], device=device)
        s    = model(feat, te, mask).squeeze(0).cpu().numpy()
        out.append(s)
    return out


# ──────────────────────────── Visualisations ───────────────────────────────

def plot_task_embed_pca(rollouts, encoder, out_dir):
    descs     = [r.task_description for r in rollouts]
    unique_d  = list(dict.fromkeys(descs))           # preserve order
    embeds    = encoder.transform(unique_d)           # (n_tasks, E)

    if embeds.shape[0] < 2:
        print("  (skip task_embed_pca — fewer than 2 unique tasks)")
        return

    n_comp = min(2, embeds.shape[0] - 1, embeds.shape[1])
    pca    = PCA(n_components=n_comp)
    proj   = pca.fit_transform(embeds)               # (n_tasks, 2)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[:, 0], proj[:, 1], s=60, alpha=0.8)
    for i, d in enumerate(unique_d):
        short = d[:40] + "…" if len(d) > 40 else d
        ax.annotate(short, proj[i], fontsize=6, alpha=0.8)
    ax.set_title("Task Embedding PCA  (each point = one task)")
    ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(out_dir, "task_embed_pca.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")


def plot_auc_comparison(base_aucs, cond_aucs, task_labels, out_dir):
    n = len(task_labels)
    x = np.arange(n)
    w = 0.35

    fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 4))
    ax.bar(x - w/2, base_aucs, w, label="Base (no task cond.)",
           color="steelblue", alpha=0.8)
    ax.bar(x + w/2, cond_aucs, w, label="Task-conditioned",
           color="darkorange", alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(task_labels, rotation=30, ha="right",
                                          fontsize=8)
    ax.set_ylim(0, 1.1); ax.axhline(0.5, color="gray", lw=1, linestyle="--")
    ax.set_ylabel("ROC-AUC"); ax.set_title("Base vs Task-Conditioned AUC per Task")
    ax.legend(); ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    path = os.path.join(out_dir, "comparison_auc.png")
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"  -> {path}")


# ──────────────────────────── CLI ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Task-conditioned failure detector",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_path",       required=True)
    parser.add_argument("--output_dir",      default="./task_cond_results")
    parser.add_argument("--train_ratio",     type=float, default=0.7)
    parser.add_argument("--task_embed_dim",  type=int,   default=32,
                        help="LSA embedding dimension for task descriptions")
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

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load ─────────────────────────────────────────────────────────────
    print("[1/5] Loading rollouts ...")
    all_r = load_rollouts(args.data_path)
    input_dim = all_r[0].hidden_states.shape[-1]

    # ── 2. Build task embeddings ────────────────────────────────────────────
    print("\n[2/5] Building task embeddings ...")
    encoder = TaskEncoder(n_components=args.task_embed_dim)
    all_descs = [r.task_description for r in all_r]
    encoder.fit(list(set(all_descs)))   # fit on unique descriptions
    all_embeds = encoder.transform(all_descs)   # (N, E)

    # ── 3. Split (stratified) ───────────────────────────────────────────────
    rng      = np.random.RandomState(args.seed)
    succ_idx = [i for i, r in enumerate(all_r) if     r.episode_success]
    fail_idx = [i for i, r in enumerate(all_r) if not r.episode_success]
    rng.shuffle(succ_idx); rng.shuffle(fail_idx)

    n_str = max(1, int(len(succ_idx) * args.train_ratio))
    n_ftr = max(1, int(len(fail_idx) * args.train_ratio))
    tr_idx = succ_idx[:n_str] + fail_idx[:n_ftr]
    te_idx = succ_idx[n_str:] + fail_idx[n_ftr:]

    train_r    = [all_r[i] for i in tr_idx]
    test_r     = [all_r[i] for i in te_idx]
    train_emb  = all_embeds[tr_idx]
    test_emb   = all_embeds[te_idx]

    # ── 4. Train both models ────────────────────────────────────────────────
    print("\n[3/5] Training base detector (no task conditioning) ...")
    base_model = FailureDetector(input_dim, hidden_dim=args.hidden_dim,
                                 n_layers=args.n_layers).to(args.device)
    train_model(base_model, train_r, n_epochs=args.n_epochs, lr=args.lr,
                lambda_reg=args.lambda_reg, batch_size=args.batch_size,
                device=args.device)

    print("\n[4/5] Training task-conditioned detector ...")
    cond_model = TaskCondFailureDetector(
        hidden_state_dim=input_dim,
        task_embed_dim=args.task_embed_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(args.device)
    _train_task_cond(cond_model, train_r, train_emb,
                     n_epochs=args.n_epochs, lr=args.lr,
                     lambda_reg=args.lambda_reg, batch_size=args.batch_size,
                     device=args.device)

    # ── 5. Evaluate & visualise ─────────────────────────────────────────────
    print("\n[5/5] Evaluating & visualising ...")
    base_scores = predict(base_model, test_r, device=args.device)
    cond_scores = _predict_task_cond(cond_model, test_r, test_emb,
                                     device=args.device)

    # Overall AUC
    y_true  = np.array([1 - r.episode_success for r in test_r])
    y_base  = np.array([s[-1] for s in base_scores])
    y_cond  = np.array([s[-1] for s in cond_scores])

    if len(np.unique(y_true)) > 1:
        auc_base = roc_auc_score(y_true, y_base)
        auc_cond = roc_auc_score(y_true, y_cond)
    else:
        auc_base = auc_cond = float("nan")

    print(f"\n  Base AUC:            {auc_base:.4f}")
    print(f"  Task-conditioned AUC:{auc_cond:.4f}  "
          f"({'↑' if auc_cond > auc_base else '↓'}"
          f" {abs(auc_cond - auc_base):.4f})")

    # Per-task AUC
    task_ids = sorted(set(r.task_id for r in test_r))
    base_task_aucs, cond_task_aucs, task_labels = [], [], []
    for tid in task_ids:
        idx = [i for i, r in enumerate(test_r) if r.task_id == tid]
        yt  = np.array([1 - test_r[i].episode_success for i in idx])
        if len(np.unique(yt)) < 2:
            continue
        yb  = np.array([y_base[i] for i in idx])
        yc  = np.array([y_cond[i] for i in idx])
        base_task_aucs.append(roc_auc_score(yt, yb))
        cond_task_aucs.append(roc_auc_score(yt, yc))
        task_labels.append(f"T{tid}")

    # Plots
    plot_task_embed_pca(all_r, encoder, args.output_dir)
    if task_labels:
        plot_auc_comparison(base_task_aucs, cond_task_aucs,
                            task_labels, args.output_dir)

    # Text summary
    summary_path = os.path.join(args.output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Base AUC:             {auc_base:.4f}\n")
        f.write(f"Task-conditioned AUC: {auc_cond:.4f}\n\n")
        f.write("Per-task:\n")
        for label, ba, ca in zip(task_labels, base_task_aucs, cond_task_aucs):
            f.write(f"  {label}:  base={ba:.4f}  cond={ca:.4f}  "
                    f"delta={ca-ba:+.4f}\n")
    print(f"  -> {summary_path}")
    print(f"\n  Outputs saved to: {os.path.abspath(args.output_dir)}/")


if __name__ == "__main__":
    main()
