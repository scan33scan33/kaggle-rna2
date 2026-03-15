"""
Data loading and preprocessing for Stanford RNA 3D Folding Part 2.

Competition data fields:
  - target_id: str
  - sequence: str (A/U/G/C + modified residues)
  - temporal_cutoff: str (date)
  - description: str (optional)

Labels (train only):
  - target_id, residue_index, x_1..x_5, y_1..y_5, z_1..z_5
    (C1' atom coordinates for 5 structures per residue)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RNATarget:
    target_id: str
    sequence: str
    temporal_cutoff: Optional[str] = None
    description: Optional[str] = None
    # Shape: (n_structures, seq_len, 3) — C1' xyz per structure
    coords: Optional[np.ndarray] = None


def load_targets(csv_path: str | Path) -> list[RNATarget]:
    """Load targets from the competition CSV."""
    df = pd.read_csv(csv_path)
    targets = []
    for _, row in df.iterrows():
        targets.append(RNATarget(
            target_id=row["target_id"],
            sequence=row["sequence"],
            temporal_cutoff=row.get("temporal_cutoff"),
            description=row.get("description"),
        ))
    return targets


def load_labels(
    targets: list[RNATarget],
    labels_csv: str | Path,
    n_structures: int = 5,
) -> list[RNATarget]:
    """
    Attach ground-truth C1' coordinates to targets.

    Expected label CSV columns:
      target_id, residue_index, x_1..x_N, y_1..y_N, z_1..z_N
    """
    df = pd.read_csv(labels_csv)
    labels_by_id: dict[str, pd.DataFrame] = {
        tid: grp.sort_values("residue_index")
        for tid, grp in df.groupby("target_id")
    }
    for target in targets:
        grp = labels_by_id.get(target.target_id)
        if grp is None:
            continue
        seq_len = len(target.sequence)
        coords = np.full((n_structures, seq_len, 3), np.nan)
        for s in range(1, n_structures + 1):
            x_col, y_col, z_col = f"x_{s}", f"y_{s}", f"z_{s}"
            if x_col not in grp.columns:
                continue
            idx = grp["residue_index"].values - 1  # 1-indexed → 0-indexed
            coords[s - 1, idx, 0] = grp[x_col].values
            coords[s - 1, idx, 1] = grp[y_col].values
            coords[s - 1, idx, 2] = grp[z_col].values
        target.coords = coords
    return targets


def make_submission(
    predictions: dict[str, np.ndarray],
    output_path: str | Path,
) -> pd.DataFrame:
    """
    Build submission CSV from predictions.

    Args:
        predictions: {target_id: np.ndarray of shape (n_structures, seq_len, 3)}
        output_path: where to save the CSV

    Returns:
        DataFrame with columns: target_id, residue_index,
            x_1..x_5, y_1..y_5, z_1..z_5
    """
    rows = []
    for target_id, coords in predictions.items():
        n_structures, seq_len, _ = coords.shape
        for res_idx in range(seq_len):
            row: dict = {"target_id": target_id, "residue_index": res_idx + 1}
            for s in range(n_structures):
                row[f"x_{s+1}"] = coords[s, res_idx, 0]
                row[f"y_{s+1}"] = coords[s, res_idx, 1]
                row[f"z_{s+1}"] = coords[s, res_idx, 2]
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return df


def cluster_split(
    targets: list[RNATarget],
    identity_threshold: float = 0.8,
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[RNATarget], list[RNATarget]]:
    """
    Sequence-similarity-aware train/val split using CD-HIT-EST style clustering.
    Falls back to random split if cd-hit-est is not available.

    Returns (train_targets, val_targets).
    """
    import subprocess, tempfile, os

    fasta_lines = []
    for t in targets:
        fasta_lines.append(f">{t.target_id}\n{t.sequence}")
    fasta = "\n".join(fasta_lines)

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            fasta_path = os.path.join(tmpdir, "seqs.fa")
            out_path = os.path.join(tmpdir, "clusters")
            with open(fasta_path, "w") as f:
                f.write(fasta)
            subprocess.run(
                ["cd-hit-est", "-i", fasta_path, "-o", out_path,
                 "-c", str(identity_threshold), "-T", "4", "-M", "4000"],
                check=True, capture_output=True,
            )
            clusters = _parse_cdhit_clstr(out_path + ".clstr")
    except (FileNotFoundError, subprocess.CalledProcessError):
        # cd-hit-est not available — fall back to random split
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(targets))
        n_val = max(1, int(len(targets) * val_fraction))
        val_idx = set(indices[:n_val].tolist())
        train = [t for i, t in enumerate(targets) if i not in val_idx]
        val = [t for i, t in enumerate(targets) if i in val_idx]
        return train, val

    # Pick one cluster member per cluster for val
    rng = np.random.default_rng(seed)
    id_to_target = {t.target_id: t for t in targets}
    n_val = max(1, int(len(clusters) * val_fraction))
    val_cluster_ids = rng.choice(list(clusters.keys()), size=n_val, replace=False)
    val_ids: set[str] = set()
    for cid in val_cluster_ids:
        val_ids.add(rng.choice(clusters[cid]))
    train = [t for t in targets if t.target_id not in val_ids]
    val = [t for t in targets if t.target_id in val_ids]
    return train, val


def _parse_cdhit_clstr(clstr_path: str) -> dict[str, list[str]]:
    clusters: dict[str, list[str]] = {}
    current: list[str] = []
    cluster_id = 0
    with open(clstr_path) as f:
        for line in f:
            if line.startswith(">Cluster"):
                if current:
                    clusters[str(cluster_id)] = current
                    cluster_id += 1
                current = []
            else:
                # e.g. "0  123aa, >target_id... *"
                parts = line.split(">")
                if len(parts) > 1:
                    tid = parts[1].split(".")[0].strip()
                    current.append(tid)
    if current:
        clusters[str(cluster_id)] = current
    return clusters
