"""
Hybrid RNA 3D Structure Prediction Pipeline
============================================

Decision logic per target:
  1. Template search (Infernal + BLAST vs PDB)
     → if best TM-score >= tm_threshold: use template model
  2. Else: run fine-tuned deep learning models
     → primary:  RhoFold+ (fine-tuned)
     → secondary: RibonanzaNet2 head (fine-tuned)
     → fallback:  Boltz-1 (zero-shot)
  3. Ensemble: score all available predictions with the local
     TM-score estimator and return the top 5.

Usage:
  from src.pipeline import HybridPipeline, PipelineConfig

  cfg = PipelineConfig.from_yaml("configs/baseline/config.yaml")
  pipeline = HybridPipeline(cfg)
  predictions = pipeline.run(targets)          # {target_id: (5, L, 3)}
  pipeline.save_submission(predictions, "submissions/sub.csv")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


@dataclass
class PipelineConfig:
    # Template search
    use_template_search: bool = True
    pdb_dir: str = "data/pdb"
    blast_db: str = "data/blast/pdb_rna"
    rfam_cm: Optional[str] = "data/rfam/Rfam.cm"
    tm_threshold: float = 0.45

    # RhoFold+
    use_rhofold: bool = True
    rhofold_dir: str = ""
    rhofold_checkpoint: str = ""

    # RibonanzaNet2
    use_ribonanzanet2: bool = True
    rnapro_dir: str = ""
    ribonanzanet2_checkpoint: str = ""
    ribonanzanet2_head_checkpoint: str = ""

    # Boltz-1/2
    use_boltz: bool = True
    boltz_model: str = "boltz1"
    boltz_cache_dir: str = "~/.boltz"

    # Ensemble
    ensemble_method: str = "best_of_5"   # "best_of_5" | "average" | "rank_weighted"
    n_structures: int = 5

    # Runtime
    device: str = "cuda"
    n_cpus: int = 4

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**{k: v for k, v in data.items() if hasattr(cls, k)})


class HybridPipeline:
    """
    Orchestrates template search + fine-tuned model inference + ensembling.
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self._template_searcher = None
        self._rhofold = None
        self._boltz = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, targets) -> dict[str, np.ndarray]:
        """
        Run the full pipeline on a list of RNATarget objects.
        Returns {target_id: coords (5, seq_len, 3)}.
        """
        predictions: dict[str, np.ndarray] = {}

        for target in targets:
            print(f"[pipeline] {target.target_id}  len={len(target.sequence)}")
            coords = self._predict_single(target)
            predictions[target.target_id] = coords

        return predictions

    def save_submission(
        self,
        predictions: dict[str, np.ndarray],
        output_path: str,
    ):
        from src.data.loader import make_submission
        df = make_submission(predictions, output_path)
        # Sanity checks
        assert not df.isnull().any().any(), "NaN values in submission!"
        print(f"Submission saved: {output_path}  ({len(df)} rows)")
        return df

    # ------------------------------------------------------------------
    # Per-target prediction
    # ------------------------------------------------------------------

    def _predict_single(self, target) -> np.ndarray:
        cfg = self.cfg
        seq_len = len(target.sequence)
        candidates: list[np.ndarray] = []

        # --- Step 1: Template search ---
        if cfg.use_template_search:
            tmpl_result = self._run_template_search(target)
            if tmpl_result.used_template and tmpl_result.best_coords is not None:
                best_hit = max(tmpl_result.hits, key=lambda h: h.tm_score)
                print(f"  → template hit  {best_hit.pdb_id}  TM={best_hit.tm_score:.3f}")
                candidates.append(tmpl_result.best_coords)

        # --- Step 2: Deep learning models ---
        # Run when: no template was found, OR template search is disabled entirely.
        if not candidates:
            if cfg.use_rhofold:
                try:
                    dl_coords = self._run_rhofold(target)
                    candidates.append(dl_coords)
                    print(f"  → RhoFold+ OK")
                except Exception as e:
                    print(f"  ! RhoFold+ failed: {e}")

            if cfg.use_boltz:
                try:
                    boltz_coords = self._run_boltz(target)
                    candidates.append(boltz_coords)
                    print(f"  → Boltz OK")
                except Exception as e:
                    print(f"  ! Boltz failed: {e}")

        # --- Step 3: Ensemble ---
        if not candidates:
            print(f"  ! All methods failed — returning zeros")
            return np.zeros((cfg.n_structures, seq_len, 3), dtype=np.float32)

        return self._ensemble(candidates, seq_len)

    # ------------------------------------------------------------------
    # Method-specific runners (lazy-loaded)
    # ------------------------------------------------------------------

    def _run_template_search(self, target):
        if self._template_searcher is None:
            from src.inference.template_search import TemplateSearcher
            self._template_searcher = TemplateSearcher(
                pdb_dir=self.cfg.pdb_dir,
                blast_db=self.cfg.blast_db,
                rfam_cm=self.cfg.rfam_cm,
                tm_threshold=self.cfg.tm_threshold,
                n_cpus=self.cfg.n_cpus,
            )
        return self._template_searcher.search(target.target_id, target.sequence)

    def _run_rhofold(self, target) -> np.ndarray:
        if self._rhofold is None:
            from src.inference.rhofold_runner import RhoFoldRunner
            self._rhofold = RhoFoldRunner(
                rhofold_dir=self.cfg.rhofold_dir,
                checkpoint=self.cfg.rhofold_checkpoint,
                device=self.cfg.device,
            )
        return self._rhofold.predict(target.target_id, target.sequence)

    def _run_boltz(self, target) -> np.ndarray:
        if self._boltz is None:
            from src.inference.boltz_runner import BoltzRunner
            self._boltz = BoltzRunner(
                model_type=self.cfg.boltz_model,
                cache_dir=self.cfg.boltz_cache_dir,
                n_structures=self.cfg.n_structures,
                device=self.cfg.device,
            )
        return self._boltz.predict(target.target_id, target.sequence)

    # ------------------------------------------------------------------
    # Ensembling
    # ------------------------------------------------------------------

    def _ensemble(
        self,
        candidates: list[np.ndarray],
        seq_len: int,
    ) -> np.ndarray:
        """
        Combine predictions from multiple sources.

        Each candidate is (n_structures, seq_len, 3).
        Returns (5, seq_len, 3).
        """
        cfg = self.cfg
        method = cfg.ensemble_method

        # Flatten all structures across candidates
        all_structs: list[np.ndarray] = []
        for c in candidates:
            for i in range(c.shape[0]):
                all_structs.append(c[i])  # (seq_len, 3)

        if method == "best_of_5" or len(all_structs) <= cfg.n_structures:
            # Just return the first n_structures (template takes priority)
            while len(all_structs) < cfg.n_structures:
                rng = np.random.default_rng(len(all_structs))
                all_structs.append(
                    all_structs[0] + rng.normal(0, 0.05, all_structs[0].shape)
                )
            return np.stack(all_structs[:cfg.n_structures]).astype(np.float32)

        elif method == "rank_weighted":
            # Use pairwise GDT/RMSD to pick the most "consensus" structures
            selected = _pick_consensus_structures(all_structs, cfg.n_structures)
            return np.stack(selected).astype(np.float32)

        elif method == "average":
            # Average coords from same-source structures; pad to 5
            avg = np.mean([s for s in all_structs[:cfg.n_structures]], axis=0)
            out = [avg] * cfg.n_structures
            return np.stack(out).astype(np.float32)

        return np.stack(all_structs[:cfg.n_structures]).astype(np.float32)


def _pick_consensus_structures(
    structs: list[np.ndarray],
    n: int,
) -> list[np.ndarray]:
    """
    Pick n structures that are most consistent with the ensemble median
    (minimum average RMSD to all others — a proxy for consensus quality).
    """
    k = len(structs)
    # Compute pairwise RMSD matrix
    rmsd = np.zeros((k, k))
    for i in range(k):
        for j in range(i + 1, k):
            diff = structs[i] - structs[j]
            r = np.sqrt(np.mean(np.sum(diff ** 2, axis=-1)))
            rmsd[i, j] = rmsd[j, i] = r
    # Score = mean RMSD to all others (lower = more consensus-like)
    scores = rmsd.mean(axis=1)
    ranked = np.argsort(scores)
    return [structs[i] for i in ranked[:n]]
