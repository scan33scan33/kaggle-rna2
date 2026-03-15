"""
Template-Based Modelling (TBM) pipeline.

Strategy (winner of Part 1):
  1. Search the PDB RNA database with Infernal (covariance model search)
     and/or BLAST-N for sequence-similar structures.
  2. For each target, select the best-scoring template (TM-align > 0.45
     is the threshold for a "correct global fold").
  3. Build a model by threading/aligning the query onto the template.

Dependencies:
  - Infernal (cmsearch) — https://eddylab.org/infernal/
  - BLAST+ (blastn) — for sequence search
  - US-align / TM-align — for structural alignment and scoring
  - rMSA — for MSA construction (used by AF3 baseline)

All tools run locally (no internet); databases are pre-downloaded.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class TemplateHit:
    pdb_id: str          # e.g. "7ABC_A"
    e_value: float
    tm_score: float      # TM-align score vs query (if computed)
    coords_path: str     # path to template PDB/mmCIF


@dataclass
class TemplateResult:
    target_id: str
    hits: list[TemplateHit] = field(default_factory=list)
    best_coords: Optional[np.ndarray] = None   # (5, seq_len, 3) if found
    used_template: bool = False


class TemplateSearcher:
    """
    Searches for structural templates for a list of RNA sequences.

    Args:
        pdb_dir: directory containing PDB/mmCIF files indexed for BLAST/Infernal
        blast_db: path to BLAST nucleotide database of PDB RNA chains
        rfam_cm: path to Rfam covariance model database (for Infernal)
        tm_threshold: minimum TM-score to accept a template
        n_cpus: number of CPU cores to use
    """

    def __init__(
        self,
        pdb_dir: str,
        blast_db: str,
        rfam_cm: Optional[str] = None,
        tm_threshold: float = 0.45,
        n_cpus: int = 4,
    ):
        self.pdb_dir = Path(pdb_dir)
        self.blast_db = blast_db
        self.rfam_cm = rfam_cm
        self.tm_threshold = tm_threshold
        self.n_cpus = n_cpus

    def search(self, target_id: str, sequence: str) -> TemplateResult:
        result = TemplateResult(target_id=target_id)

        hits = self._blast_search(target_id, sequence)
        if self.rfam_cm:
            hits += self._infernal_search(target_id, sequence)

        # Sort by e-value, deduplicate by pdb_id
        seen: set[str] = set()
        unique_hits: list[TemplateHit] = []
        for h in sorted(hits, key=lambda x: x.e_value):
            if h.pdb_id not in seen:
                seen.add(h.pdb_id)
                unique_hits.append(h)

        result.hits = unique_hits[:10]  # keep top 10

        # Score templates with TM-align and pick best above threshold
        best_hit = self._score_templates(sequence, result.hits)
        if best_hit is not None and best_hit.tm_score >= self.tm_threshold:
            result.used_template = True
            result.best_coords = self._build_model(sequence, best_hit)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _blast_search(self, target_id: str, sequence: str) -> list[TemplateHit]:
        hits: list[TemplateHit] = []
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                query_fa = os.path.join(tmpdir, "query.fa")
                blast_out = os.path.join(tmpdir, "blast.tsv")
                with open(query_fa, "w") as f:
                    f.write(f">{target_id}\n{sequence}\n")
                subprocess.run(
                    [
                        "blastn", "-query", query_fa,
                        "-db", self.blast_db,
                        "-out", blast_out,
                        "-outfmt", "6 sseqid evalue bitscore",
                        "-num_threads", str(self.n_cpus),
                        "-max_target_seqs", "20",
                        "-evalue", "1e-3",
                    ],
                    check=True, capture_output=True,
                )
                with open(blast_out) as f:
                    for line in f:
                        parts = line.strip().split("\t")
                        if len(parts) < 2:
                            continue
                        pdb_id = parts[0]
                        e_value = float(parts[1])
                        pdb_path = str(self.pdb_dir / f"{pdb_id.lower()}.cif")
                        hits.append(TemplateHit(
                            pdb_id=pdb_id,
                            e_value=e_value,
                            tm_score=0.0,
                            coords_path=pdb_path,
                        ))
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass  # blastn not installed or db missing — skip
        return hits

    def _infernal_search(self, target_id: str, sequence: str) -> list[TemplateHit]:
        hits: list[TemplateHit] = []
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                query_fa = os.path.join(tmpdir, "query.fa")
                cm_out = os.path.join(tmpdir, "cmsearch.txt")
                with open(query_fa, "w") as f:
                    f.write(f">{target_id}\n{sequence}\n")
                subprocess.run(
                    [
                        "cmsearch", "--cpu", str(self.n_cpus),
                        "--tblout", cm_out,
                        self.rfam_cm, query_fa,
                    ],
                    check=True, capture_output=True,
                )
                with open(cm_out) as f:
                    for line in f:
                        if line.startswith("#"):
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        pdb_id = parts[0]
                        e_value = float(parts[4])
                        pdb_path = str(self.pdb_dir / f"{pdb_id.lower()}.cif")
                        hits.append(TemplateHit(
                            pdb_id=pdb_id,
                            e_value=e_value,
                            tm_score=0.0,
                            coords_path=pdb_path,
                        ))
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        return hits

    def _score_templates(
        self,
        query_seq: str,
        hits: list[TemplateHit],
    ) -> Optional[TemplateHit]:
        """Run US-align to get TM-scores for each hit."""
        best: Optional[TemplateHit] = None
        for hit in hits:
            if not os.path.exists(hit.coords_path):
                continue
            tm = _run_usalign(query_seq, hit.coords_path)
            hit.tm_score = tm
            if best is None or tm > best.tm_score:
                best = hit
        return best

    def _build_model(
        self,
        query_seq: str,
        hit: TemplateHit,
    ) -> np.ndarray:
        """
        Thread query onto template and return 5 slightly perturbed
        coordinate sets (to satisfy the 5-model submission requirement).
        """
        coords_1 = _extract_c1prime_coords(hit.coords_path, query_seq)
        n = coords_1.shape[0]
        rng = np.random.default_rng(42)
        # Generate 5 models with small Gaussian noise (0.1 Å std) on copies 2-5
        coords_5 = np.stack([
            coords_1,
            coords_1 + rng.normal(0, 0.1, (n, 3)),
            coords_1 + rng.normal(0, 0.1, (n, 3)),
            coords_1 + rng.normal(0, 0.1, (n, 3)),
            coords_1 + rng.normal(0, 0.1, (n, 3)),
        ])  # (5, seq_len, 3)
        return coords_5


# ------------------------------------------------------------------
# Utility functions (require external tools)
# ------------------------------------------------------------------

def _run_usalign(query_seq: str, template_cif: str) -> float:
    """Run US-align and return TM-score. Returns 0.0 on failure."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            query_fa = os.path.join(tmpdir, "q.fa")
            with open(query_fa, "w") as f:
                f.write(f">query\n{query_seq}\n")
            result = subprocess.run(
                ["USalign", template_cif, query_fa, "-mm", "4", "-ter", "0"],
                capture_output=True, text=True, check=True,
            )
            for line in result.stdout.splitlines():
                if line.startswith("TM-score="):
                    return float(line.split("=")[1].split()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    return 0.0


def _extract_c1prime_coords(cif_path: str, query_seq: str) -> np.ndarray:
    """
    Extract C1' atom coordinates from a CIF file.
    Returns array of shape (seq_len, 3).
    If seq_len of template != query, pads/truncates to match.
    """
    try:
        from Bio.PDB import MMCIFParser
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("tmpl", cif_path)
        coords = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    atom = residue.child_dict.get("C1'")
                    if atom is not None:
                        coords.append(atom.get_vector().get_array())
            break  # first model only
        arr = np.array(coords, dtype=np.float32)
    except Exception:
        # Fallback: return zeros
        arr = np.zeros((len(query_seq), 3), dtype=np.float32)

    target_len = len(query_seq)
    if len(arr) >= target_len:
        return arr[:target_len]
    # Pad with last known coordinate
    pad = np.repeat(arr[-1:], target_len - len(arr), axis=0)
    return np.concatenate([arr, pad], axis=0)
