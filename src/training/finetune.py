"""
Fine-tuning orchestrator for open-weight RNA structure models.

Supported models:
  - RhoFold+ (ml4bio/RhoFold)  — freeze encoder, train structure module
  - RibonanzaNet2 (via RNAPro) — freeze backbone, train projection + structure head
  - Boltz-1/2               — not yet fine-tunable via public API; use inference only

Fine-tuning strategy:
  Phase 1 (fast): freeze encoder, train only structure decoder  (~1-2h / A100)
  Phase 2 (full): unfreeze all, low LR                          (~4-8h / A100)

Usage:
  python -m src.training.finetune \
      --model rhofold \
      --rhofold_dir /path/to/RhoFold \
      --pretrained_ckpt /path/to/rhofold_pretrained.pt \
      --train_csv /path/to/train_sequences.csv \
      --labels_csv /path/to/train_labels.csv \
      --output_dir checkpoints/rhofold_finetuned \
      --phase 1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["rhofold", "ribonanzanet2"], required=True)
    # RhoFold args
    p.add_argument("--rhofold_dir", default="")
    p.add_argument("--pretrained_ckpt", default="")
    # RibonanzaNet2 args
    p.add_argument("--rnapro_dir", default="")
    p.add_argument("--ribonanzanet2_ckpt", default="")
    # Data
    p.add_argument("--train_csv", required=True)
    p.add_argument("--labels_csv", required=True)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--cluster_split", action="store_true",
                   help="Use sequence-similarity-aware split (requires cd-hit-est)")
    # Training
    p.add_argument("--phase", type=int, choices=[1, 2], default=1,
                   help="1=freeze encoder, 2=full fine-tune")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--n_epochs", type=int, default=10)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    _set_seed(args.seed)

    from src.data.loader import load_targets, load_labels, cluster_split

    print(f"Loading data from {args.train_csv} ...")
    targets = load_targets(args.train_csv)
    targets = load_labels(targets, args.labels_csv)
    # Drop targets with no ground-truth coords
    targets = [t for t in targets if t.coords is not None]
    print(f"  {len(targets)} targets with labels")

    if args.cluster_split:
        train_targets, val_targets = cluster_split(
            targets, val_fraction=args.val_fraction, seed=args.seed
        )
    else:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(targets))
        n_val = max(1, int(len(targets) * args.val_fraction))
        val_idx = set(idx[:n_val].tolist())
        train_targets = [t for i, t in enumerate(targets) if i not in val_idx]
        val_targets = [t for i, t in enumerate(targets) if i in val_idx]

    print(f"  train={len(train_targets)}  val={len(val_targets)}")

    if args.model == "rhofold":
        _finetune_rhofold(args, train_targets, val_targets)
    elif args.model == "ribonanzanet2":
        _finetune_ribonanzanet2(args, train_targets, val_targets)


def _finetune_rhofold(args, train_targets, val_targets):
    from src.inference.rhofold_runner import RhoFoldFineTuner

    freeze_encoder = args.phase == 1
    tuner = RhoFoldFineTuner(
        rhofold_dir=args.rhofold_dir,
        pretrained_checkpoint=args.pretrained_ckpt,
        output_dir=args.output_dir,
        freeze_encoder=freeze_encoder,
        lr=args.lr,
        n_epochs=args.n_epochs,
        device=args.device,
    )
    best_ckpt = tuner.finetune(train_targets, val_targets)
    print(f"Best checkpoint saved to: {best_ckpt}")


def _finetune_ribonanzanet2(args, train_targets, val_targets):
    """
    Fine-tune RibonanzaNet2 as a 3D structure encoder within the RNAPro framework.

    Approach:
      - Load RibonanzaNet2 pretrained weights
      - Add a lightweight MLP projection head: hidden_dim → (seq_len, 3)
      - Train on C1' coordinate regression with MSE + auxiliary distogram loss
      - Phase 1: freeze RibonanzaNet2, train head only
      - Phase 2: unfreeze all with 10x lower LR on backbone
    """
    import sys, torch, torch.nn as nn

    # Verify RNAPro is available
    if not args.rnapro_dir or not os.path.exists(args.rnapro_dir):
        raise ValueError(
            f"--rnapro_dir must point to a cloned RNAPro repository. "
            f"Got: {args.rnapro_dir}"
        )
    sys.path.insert(0, args.rnapro_dir)

    try:
        from ribonanzanet2.network import RibonanzaNet2  # type: ignore
    except ImportError:
        raise RuntimeError(
            "RibonanzaNet2 not importable from rnapro_dir. "
            "Check https://github.com/NVIDIA-Digital-Bio/RNAPro setup."
        )

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load pretrained backbone
    backbone = RibonanzaNet2()
    ckpt = torch.load(args.ribonanzanet2_ckpt, map_location="cpu")
    backbone.load_state_dict(ckpt if isinstance(ckpt, dict) and "model" not in ckpt else ckpt.get("model", ckpt))
    backbone = backbone.to(device)

    # Structure head: per-residue MLP → (3,) C1' offset
    hidden_dim = 256
    struct_head = nn.Sequential(
        nn.Linear(backbone.d_model, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 3),
    ).to(device)

    freeze_encoder = args.phase == 1
    if freeze_encoder:
        for p in backbone.parameters():
            p.requires_grad = False
        trainable = list(struct_head.parameters())
    else:
        # Phase 2: different LRs
        trainable = [
            {"params": backbone.parameters(), "lr": args.lr * 0.1},
            {"params": struct_head.parameters(), "lr": args.lr},
        ]

    optimizer = torch.optim.Adam(
        trainable if isinstance(trainable[0], torch.nn.Parameter) else trainable,
        lr=args.lr,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    best_ckpt = str(output_dir / "best_ribonanzanet2.pt")

    from src.inference.rhofold_runner import _tokenise_rna
    import torch.nn.functional as F

    for epoch in range(args.n_epochs):
        backbone.train() if not freeze_encoder else backbone.eval()
        struct_head.train()
        train_loss = _run_rbn2_epoch(
            backbone, struct_head, train_targets, optimizer, device, train=True
        )
        backbone.eval(); struct_head.eval()
        with torch.no_grad():
            val_loss = _run_rbn2_epoch(
                backbone, struct_head, val_targets, None, device, train=False
            )
        scheduler.step()
        print(f"[RibonanzaNet2] Epoch {epoch+1}/{args.n_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "backbone": backbone.state_dict(),
                "struct_head": struct_head.state_dict(),
                "epoch": epoch,
            }, best_ckpt)

    print(f"Best checkpoint saved to: {best_ckpt}")


def _run_rbn2_epoch(backbone, struct_head, targets, optimizer, device, train: bool) -> float:
    import torch, torch.nn.functional as F
    from src.inference.rhofold_runner import _tokenise_rna
    total_loss = 0.0
    n = 0
    for target in targets:
        if target.coords is None:
            continue
        token_ids = torch.tensor(
            _tokenise_rna(target.sequence), dtype=torch.long
        ).unsqueeze(0).to(device)
        gt = torch.tensor(target.coords[0], dtype=torch.float32).unsqueeze(0).to(device)

        embeddings = backbone(token_ids)          # (1, seq_len, d_model)
        pred = struct_head(embeddings)            # (1, seq_len, 3)

        # Normalise by centering the GT (models predict relative coords)
        gt_centered = gt - gt.mean(dim=1, keepdim=True)
        pred_centered = pred - pred.mean(dim=1, keepdim=True)
        loss = F.mse_loss(pred_centered, gt_centered)

        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(backbone.parameters()) + list(struct_head.parameters()), 1.0
            )
            optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def _set_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


if __name__ == "__main__":
    main()
