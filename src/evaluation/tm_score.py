"""
Local TM-score evaluation using US-align.

US-align must be installed: https://zhanggroup.org/US-align/
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def tm_score_from_coords(
    pred_coords: np.ndarray,
    true_coords: np.ndarray,
) -> float:
    """
    Compute TM-score between predicted and ground-truth C1' coordinates.
    Writes temporary PDB files and calls US-align.

    Args:
        pred_coords: (seq_len, 3) predicted C1' coordinates
        true_coords: (seq_len, 3) ground-truth C1' coordinates

    Returns:
        TM-score in [0, 1], or 0.0 if US-align is unavailable.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        pred_pdb = os.path.join(tmpdir, "pred.pdb")
        true_pdb = os.path.join(tmpdir, "true.pdb")
        _write_c1prime_pdb(pred_coords, pred_pdb)
        _write_c1prime_pdb(true_coords, true_pdb)
        return _run_usalign_tm(pred_pdb, true_pdb)


def evaluate_predictions(
    predictions: dict[str, np.ndarray],
    targets,
) -> dict[str, float]:
    """
    Evaluate all predictions against ground truth.

    Args:
        predictions: {target_id: (n_structures, seq_len, 3)}
        targets: list of RNATarget with .coords set

    Returns:
        {target_id: best_of_5_tm_score}
    """
    scores: dict[str, float] = {}
    gt_map = {t.target_id: t.coords for t in targets if t.coords is not None}

    for target_id, pred_coords in predictions.items():
        gt_coords = gt_map.get(target_id)
        if gt_coords is None:
            continue
        best_tm = 0.0
        for s in range(pred_coords.shape[0]):
            for g in range(gt_coords.shape[0]):
                tm = tm_score_from_coords(pred_coords[s], gt_coords[g])
                best_tm = max(best_tm, tm)
        scores[target_id] = best_tm

    if scores:
        mean_tm = np.mean(list(scores.values()))
        print(f"Mean TM-score (best-of-5): {mean_tm:.4f}  (n={len(scores)})")
    return scores


def _write_c1prime_pdb(coords: np.ndarray, path: str):
    """Write C1' atoms as a minimal PDB file."""
    with open(path, "w") as f:
        for i, (x, y, z) in enumerate(coords):
            f.write(
                f"ATOM  {i+1:5d}  C1' RNA A{i+1:4d}    "
                f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
            )
        f.write("END\n")


def _run_usalign_tm(pred_pdb: str, true_pdb: str) -> float:
    try:
        result = subprocess.run(
            ["USalign", pred_pdb, true_pdb, "-mm", "4", "-ter", "0"],
            capture_output=True, text=True, check=True,
        )
        for line in result.stdout.splitlines():
            if line.startswith("TM-score="):
                return float(line.split("=")[1].split()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    return 0.0
