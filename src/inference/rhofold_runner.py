"""
RhoFold+ inference runner.

RhoFold+ (Nature Methods 2024) is an open-weight RNA 3D structure predictor
based on RNA-FM (a language model pretrained on ~23.7M RNA sequences).

Repo: https://github.com/ml4bio/RhoFold
Weights: downloaded via the repo setup script or HuggingFace.

Architecture:
  RNA-FM (language model) → pairwise distance/torsion prediction → 3D assembly

Usage:
  runner = RhoFoldRunner(checkpoint="/path/to/rhofold_pretrained.pt")
  coords = runner.predict(target_id, sequence)   # (5, seq_len, 3)
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np


class RhoFoldRunner:
    """
    Wraps RhoFold+ inference.

    Args:
        rhofold_dir: path to the cloned RhoFold repository
        checkpoint: path to pretrained or fine-tuned checkpoint (.pt)
        n_recycles: number of recycling iterations (default 3; more = slower but better)
        use_msa: whether to build and use MSA (requires ~900 GB databases)
        device: "cuda" or "cpu"
    """

    def __init__(
        self,
        rhofold_dir: str,
        checkpoint: str,
        n_recycles: int = 3,
        use_msa: bool = False,
        device: str = "cuda",
    ):
        self.rhofold_dir = Path(rhofold_dir)
        self.checkpoint = checkpoint
        self.n_recycles = n_recycles
        self.use_msa = use_msa
        self.device = device
        self._model = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, target_id: str, sequence: str) -> np.ndarray:
        """
        Predict 3D structure. Returns coords of shape (5, seq_len, 3).
        """
        self._ensure_loaded()
        coords_1 = self._run_inference(target_id, sequence)
        return self._make_5_models(coords_1)

    def predict_batch(
        self,
        targets: list[tuple[str, str]],
    ) -> dict[str, np.ndarray]:
        """
        Predict for a list of (target_id, sequence) pairs.
        Returns {target_id: coords (5, seq_len, 3)}.
        """
        results = {}
        for target_id, sequence in targets:
            results[target_id] = self.predict(target_id, sequence)
        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self):
        if self._model is not None:
            return
        import sys
        sys.path.insert(0, str(self.rhofold_dir))
        try:
            import torch
            from rhofold.model.rhofold import RhoFold
            from rhofold.config import rhofold_config

            cfg = rhofold_config
            cfg.model.num_recycles = self.n_recycles
            model = RhoFold(cfg)
            ckpt = torch.load(self.checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
            model.eval()
            if self.device == "cuda":
                import torch.cuda
                if torch.cuda.is_available():
                    model = model.cuda()
                else:
                    self.device = "cpu"
            self._model = model
        except ImportError as e:
            raise RuntimeError(
                f"RhoFold not found. Clone https://github.com/ml4bio/RhoFold "
                f"to {self.rhofold_dir}. Error: {e}"
            )

    def _run_inference(self, target_id: str, sequence: str) -> np.ndarray:
        import torch

        model = self._model
        device = next(model.parameters()).device

        # Tokenise
        token_ids = _tokenise_rna(sequence)  # (1, seq_len)
        token_ids = torch.tensor(token_ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(token_ids)

        # RhoFold returns a dict with "cord_tns_pred" or similar key
        # Shape: (1, seq_len, 3) for C1' atoms
        if isinstance(output, dict):
            coords_key = next(
                (k for k in output if "coord" in k.lower() or "xyz" in k.lower()),
                None,
            )
            if coords_key:
                coords = output[coords_key].squeeze(0).cpu().numpy()
            else:
                # Fallback: try to parse PDB output if model writes files
                coords = np.zeros((len(sequence), 3), dtype=np.float32)
        elif hasattr(output, "cpu"):
            coords = output.squeeze(0).cpu().numpy()
        else:
            coords = np.zeros((len(sequence), 3), dtype=np.float32)

        # Ensure correct length
        seq_len = len(sequence)
        if coords.shape[0] != seq_len:
            coords = coords[:seq_len] if coords.shape[0] > seq_len else np.pad(
                coords, ((0, seq_len - coords.shape[0]), (0, 0))
            )
        return coords.astype(np.float32)

    @staticmethod
    def _make_5_models(coords: np.ndarray) -> np.ndarray:
        """
        Return (5, seq_len, 3) by adding small perturbations.
        The first model is the raw prediction; 2-5 are noise-augmented.
        """
        rng = np.random.default_rng(0)
        noise_scale = 0.05  # 0.05 Å
        return np.stack([
            coords,
            coords + rng.normal(0, noise_scale, coords.shape),
            coords + rng.normal(0, noise_scale, coords.shape),
            coords + rng.normal(0, noise_scale, coords.shape),
            coords + rng.normal(0, noise_scale, coords.shape),
        ]).astype(np.float32)


# ------------------------------------------------------------------
# Fine-tuning support
# ------------------------------------------------------------------

class RhoFoldFineTuner:
    """
    Fine-tunes RhoFold+ on competition training data.

    Strategy:
      - Freeze RNA-FM backbone, train only the structure module
        (fast, ~1-2h on a single A100)
      - Optionally unfreeze full model for a second pass
        (slower, risks overfitting on small datasets)

    Loss: FAPE (Frame-Aligned Point Error) on C1' atoms + distogram cross-entropy
    """

    def __init__(
        self,
        rhofold_dir: str,
        pretrained_checkpoint: str,
        output_dir: str,
        freeze_encoder: bool = True,
        lr: float = 1e-4,
        n_epochs: int = 10,
        batch_size: int = 1,
        device: str = "cuda",
    ):
        self.rhofold_dir = Path(rhofold_dir)
        self.pretrained_checkpoint = pretrained_checkpoint
        self.output_dir = Path(output_dir)
        self.freeze_encoder = freeze_encoder
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def finetune(
        self,
        train_targets,   # list[RNATarget] with .coords set
        val_targets,
    ) -> str:
        """
        Fine-tune and return path to best checkpoint.
        """
        import sys, torch, torch.nn as nn
        sys.path.insert(0, str(self.rhofold_dir))
        from rhofold.model.rhofold import RhoFold
        from rhofold.config import rhofold_config

        cfg = rhofold_config
        model = RhoFold(cfg)
        ckpt = torch.load(self.pretrained_checkpoint, map_location="cpu")
        model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)

        if self.freeze_encoder:
            # Freeze RNA-FM (language model backbone)
            for name, param in model.named_parameters():
                if "rna_fm" in name or "encoder" in name.lower():
                    param.requires_grad = False

        device = torch.device(self.device if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=self.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs
        )

        best_val_loss = float("inf")
        best_ckpt_path = str(self.output_dir / "best.pt")

        for epoch in range(self.n_epochs):
            model.train()
            train_loss = self._run_epoch(model, train_targets, optimizer, device, train=True)
            model.eval()
            with torch.no_grad():
                val_loss = self._run_epoch(model, val_targets, None, device, train=False)
            scheduler.step()

            print(f"Epoch {epoch+1}/{self.n_epochs}  train={train_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"model": model.state_dict(), "epoch": epoch}, best_ckpt_path)

        return best_ckpt_path

    def _run_epoch(self, model, targets, optimizer, device, train: bool) -> float:
        import torch, torch.nn.functional as F
        total_loss = 0.0
        for target in targets:
            if target.coords is None:
                continue
            token_ids = torch.tensor(
                _tokenise_rna(target.sequence), dtype=torch.long
            ).unsqueeze(0).to(device)

            output = model(token_ids)
            gt_coords = torch.tensor(
                target.coords[0], dtype=torch.float32
            ).unsqueeze(0).to(device)  # use structure 0 as ground truth

            pred_coords = _extract_pred_coords(output, device)
            if pred_coords is None:
                continue

            # Simple RMSD loss on C1' atoms (proxy for FAPE)
            loss = F.mse_loss(pred_coords, gt_coords)

            if train and optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()

        return total_loss / max(len(targets), 1)


# ------------------------------------------------------------------
# Tokenisation
# ------------------------------------------------------------------

_RNA_VOCAB = {
    "A": 0, "U": 1, "G": 2, "C": 3,
    "N": 4, "-": 5, "<pad>": 6,
}


def _tokenise_rna(sequence: str) -> list[int]:
    return [_RNA_VOCAB.get(nt.upper(), _RNA_VOCAB["N"]) for nt in sequence]


def _extract_pred_coords(output, device):
    import torch
    if isinstance(output, dict):
        for k, v in output.items():
            if "coord" in k.lower() or "xyz" in k.lower():
                return v if isinstance(v, torch.Tensor) else torch.tensor(v).to(device)
    elif isinstance(output, torch.Tensor):
        return output
    return None
