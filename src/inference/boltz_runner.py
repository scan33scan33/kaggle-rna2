"""
Boltz-1 inference runner.

Boltz-1 (MIT, 2024) is a fully open-source model achieving AlphaFold3-level
accuracy on protein/RNA/DNA/small-molecule structure prediction.

Repo:    https://github.com/jwohlwend/boltz
License: MIT
Install: pip install boltz

Boltz-2 (2025) improves RNA and DNA-protein complex accuracy further
and is also open-source — swap the model_type flag to use it.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


class BoltzRunner:
    """
    Wraps Boltz-1/Boltz-2 inference via its CLI.

    Args:
        model_type: "boltz1" or "boltz2"
        cache_dir: where Boltz downloads/caches model weights
        n_structures: number of structures to generate per target (max 5)
        recycling_steps: number of recycles (default 3)
        diffusion_samples: number of diffusion samples (default 1)
        device: "cuda" or "cpu"
    """

    def __init__(
        self,
        model_type: str = "boltz1",
        cache_dir: str = "~/.boltz",
        n_structures: int = 5,
        recycling_steps: int = 3,
        diffusion_samples: int = 1,
        device: str = "cuda",
    ):
        self.model_type = model_type
        self.cache_dir = os.path.expanduser(cache_dir)
        self.n_structures = n_structures
        self.recycling_steps = recycling_steps
        self.diffusion_samples = diffusion_samples
        self.device = device

    def predict(self, target_id: str, sequence: str) -> np.ndarray:
        """
        Predict 3D structure for a single RNA target.
        Returns coords of shape (n_structures, seq_len, 3).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            input_fasta = self._write_input(tmpdir, target_id, sequence)
            out_dir = os.path.join(tmpdir, "output")
            self._run_boltz(input_fasta, out_dir)
            return self._load_predictions(out_dir, target_id, sequence)

    def predict_batch(
        self,
        targets: list[tuple[str, str]],
    ) -> dict[str, np.ndarray]:
        """
        Predict for a list of (target_id, sequence) pairs.
        Boltz supports batched FASTA input for efficiency.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write all sequences to a single FASTA
            fasta_path = os.path.join(tmpdir, "batch.fasta")
            with open(fasta_path, "w") as f:
                for target_id, sequence in targets:
                    f.write(f">{target_id}|rna\n{sequence}\n")
            out_dir = os.path.join(tmpdir, "output")
            self._run_boltz(fasta_path, out_dir)
            results = {}
            for target_id, sequence in targets:
                results[target_id] = self._load_predictions(out_dir, target_id, sequence)
            return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_input(self, tmpdir: str, target_id: str, sequence: str) -> str:
        """
        Write Boltz-compatible FASTA input.
        Boltz uses "|rna" suffix to specify molecule type.
        """
        fasta_path = os.path.join(tmpdir, f"{target_id}.fasta")
        with open(fasta_path, "w") as f:
            f.write(f">{target_id}|rna\n{sequence}\n")
        return fasta_path

    def _run_boltz(self, input_path: str, out_dir: str):
        cmd = [
            "boltz", "predict", input_path,
            "--out_dir", out_dir,
            "--cache", self.cache_dir,
            "--recycling_steps", str(self.recycling_steps),
            "--diffusion_samples", str(self.diffusion_samples),
            "--num_workers", "4",
            "--accelerator", self.device,
        ]
        if self.model_type == "boltz2":
            cmd += ["--model", "boltz2"]

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            raise RuntimeError(
                "boltz CLI not found. Install with: pip install boltz"
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Boltz prediction failed: {e}")

    def _load_predictions(
        self,
        out_dir: str,
        target_id: str,
        sequence: str,
    ) -> np.ndarray:
        """
        Load predicted C1' coordinates from Boltz output directory.
        Returns (n_structures, seq_len, 3).
        """
        seq_len = len(sequence)
        all_coords: list[np.ndarray] = []

        # Boltz outputs one CIF/PDB per sample; filenames vary by version
        cif_dir = Path(out_dir)
        candidates = sorted(
            list(cif_dir.rglob(f"{target_id}*.cif"))
            + list(cif_dir.rglob(f"{target_id}*.pdb"))
        )

        for cif_path in candidates[: self.n_structures]:
            coords = _extract_c1prime_from_structure(str(cif_path), seq_len)
            all_coords.append(coords)

        # Pad to n_structures if fewer predictions were produced
        while len(all_coords) < self.n_structures:
            if all_coords:
                rng = np.random.default_rng(len(all_coords))
                all_coords.append(all_coords[0] + rng.normal(0, 0.1, all_coords[0].shape))
            else:
                all_coords.append(np.zeros((seq_len, 3), dtype=np.float32))

        return np.stack(all_coords[: self.n_structures]).astype(np.float32)


# ------------------------------------------------------------------
# NVIDIA RNAPro runner (wraps RibonanzaNet2 + Protenix)
# ------------------------------------------------------------------

class RNAProRunner:
    """
    Wraps NVIDIA RNAPro inference.

    RNAPro = RibonanzaNet2 (encoder) + Protenix (AF3-like structure module)
    Repo:    https://github.com/NVIDIA-Digital-Bio/RNAPro
    License: Apache-2.0
    Weights: HuggingFace / NGC

    Args:
        rnapro_dir: path to cloned RNAPro repo
        checkpoint_dir: directory containing model checkpoints
        use_ribonanzanet2: whether to use RibonanzaNet2 encoder (recommended)
        device: "cuda" or "cpu"
    """

    def __init__(
        self,
        rnapro_dir: str,
        checkpoint_dir: str,
        use_ribonanzanet2: bool = True,
        device: str = "cuda",
    ):
        self.rnapro_dir = Path(rnapro_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.use_ribonanzanet2 = use_ribonanzanet2
        self.device = device

    def predict(self, target_id: str, sequence: str) -> np.ndarray:
        """Returns (5, seq_len, 3)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            input_json = self._write_input_json(tmpdir, target_id, sequence)
            out_dir = os.path.join(tmpdir, "output")
            os.makedirs(out_dir)
            self._run_rnapro(input_json, out_dir)
            return self._load_predictions(out_dir, target_id, len(sequence))

    def _write_input_json(self, tmpdir: str, target_id: str, sequence: str) -> str:
        data = {
            "name": target_id,
            "sequences": [{"rna": {"id": "A", "sequence": sequence}}],
        }
        path = os.path.join(tmpdir, "input.json")
        with open(path, "w") as f:
            json.dump(data, f)
        return path

    def _run_rnapro(self, input_json: str, out_dir: str):
        cmd = [
            "python", str(self.rnapro_dir / "inference.py"),
            "--input", input_json,
            "--output_dir", out_dir,
            "--checkpoint_dir", str(self.checkpoint_dir),
            f"--model.use_RibonanzaNet2={'true' if self.use_ribonanzanet2 else 'false'}",
        ]
        try:
            subprocess.run(cmd, check=True, cwd=str(self.rnapro_dir))
        except (FileNotFoundError, subprocess.CalledProcessError) as e:
            raise RuntimeError(f"RNAPro inference failed: {e}")

    def _load_predictions(
        self, out_dir: str, target_id: str, seq_len: int
    ) -> np.ndarray:
        all_coords: list[np.ndarray] = []
        for cif_path in sorted(Path(out_dir).rglob("*.cif"))[:5]:
            all_coords.append(_extract_c1prime_from_structure(str(cif_path), seq_len))
        while len(all_coords) < 5:
            base = all_coords[0] if all_coords else np.zeros((seq_len, 3), np.float32)
            rng = np.random.default_rng(len(all_coords))
            all_coords.append(base + rng.normal(0, 0.1, base.shape))
        return np.stack(all_coords[:5]).astype(np.float32)


# ------------------------------------------------------------------
# Shared utility
# ------------------------------------------------------------------

def _extract_c1prime_from_structure(path: str, seq_len: int) -> np.ndarray:
    """Extract C1' coordinates from a CIF or PDB file."""
    coords: list[list[float]] = []
    try:
        ext = Path(path).suffix.lower()
        if ext == ".cif":
            from Bio.PDB import MMCIFParser
            parser = MMCIFParser(QUIET=True)
            structure = parser.get_structure("s", path)
        else:
            from Bio.PDB import PDBParser
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("s", path)
        for model in structure:
            for chain in model:
                for residue in chain:
                    atom = residue.child_dict.get("C1'")
                    if atom is not None:
                        coords.append(atom.get_vector().get_array().tolist())
            break
    except Exception:
        pass

    arr = np.array(coords, dtype=np.float32) if coords else np.zeros((seq_len, 3), np.float32)
    if arr.shape[0] >= seq_len:
        return arr[:seq_len]
    pad = np.tile(arr[-1:], (seq_len - arr.shape[0], 1)) if len(arr) else np.zeros((seq_len - arr.shape[0], 3), np.float32)
    return np.concatenate([arr, pad], axis=0)
