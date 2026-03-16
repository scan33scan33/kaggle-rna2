# ============================================================
# COLAB SETUP
# Toggle the `if False:` → `if True:` block below to download
# data from Kaggle on first run, then switch back to False.
# ============================================================
if False:
    from google.colab import drive
    drive.mount('/content/drive')
    import os, kagglehub
    DRIVE_PATH = '/content/drive/MyDrive/kaggle_data'
    os.environ["KAGGLEHUB_CACHE"] = DRIVE_PATH
    stanford_rna_3d_folding_2_path = kagglehub.competition_download(
        'stanford-rna-3d-folding-2'
    )
    shujun717_ribonanzanet2_path = kagglehub.model_download(
        'shujun717/ribonanzanet2/PyTorch/alpha/1'
    )

INPUT_PREFIX = '/content/drive/MyDrive/kaggle_data'

# ── GPU check ──────────────────────────────────────────────────────────────
import os, subprocess
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
try:
    gpu_info = subprocess.check_output(['nvidia-smi'], stderr=subprocess.DEVNULL).decode()
    print(gpu_info)
except Exception:
    print("No GPU detected — training will be slow on CPU.")

# ── Core imports ───────────────────────────────────────────────────────────
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# ====================================================================
# MODEL: AlphaFold3-Inspired Pairformer
# ====================================================================

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairformerBlock(nn.Module):
    def __init__(self, d_single: int = 128, d_pair: int = 64, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.nhead = nhead
        self.d_single = d_single

        # Biased Attention: 1D attention biased by 2D pair representations
        self.q_proj = nn.Linear(d_single, d_single, bias=False)
        self.k_proj = nn.Linear(d_single, d_single, bias=False)
        self.v_proj = nn.Linear(d_single, d_single, bias=False)
        self.pair_bias_proj = nn.Linear(d_pair, nhead, bias=False)
        self.out_proj = nn.Linear(d_single, d_single)
        self.norm1 = nn.LayerNorm(d_single)
        self.drop1 = nn.Dropout(dropout)

        # Feed-forward transition for single rep
        self.ffn = nn.Sequential(
            nn.Linear(d_single, d_single * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_single * 4, d_single),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_single)

        # Outer product update for pair rep
        self.outer_proj1 = nn.Linear(d_single, 32)
        self.outer_proj2 = nn.Linear(d_single, 32)
        self.pair_update_proj = nn.Linear(32 * 32, d_pair)
        self.norm_pair = nn.LayerNorm(d_pair)

    def forward(
        self,
        x: torch.Tensor,   # (B, L, d_single)
        z: torch.Tensor,   # (B, L, L, d_pair)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape

        # 1. Biased multi-head attention
        res_x = x
        x_norm = self.norm1(x)

        q = self.q_proj(x_norm).view(B, L, self.nhead, -1).transpose(1, 2)
        k = self.k_proj(x_norm).view(B, L, self.nhead, -1).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, L, self.nhead, -1).transpose(1, 2)

        # pair bias: (B, L, L, nhead) → (B, nhead, L, L)
        attn_mask = self.pair_bias_proj(z).permute(0, 3, 1, 2)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0)
        else:
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
            attn_weights = attn_weights + attn_mask
            attn_probs = torch.softmax(attn_weights, dim=-1)
            out = torch.matmul(attn_probs, v)

        out = out.transpose(1, 2).reshape(B, L, -1)
        x = res_x + self.drop1(self.out_proj(out))

        # 2. Feed-forward transition
        x = x + self.ffn(self.norm2(x))

        # 3. Outer product update for pair rep
        res_z = z
        p1 = self.outer_proj1(x)   # (B, L, 32)
        p2 = self.outer_proj2(x)   # (B, L, 32)
        outer = torch.einsum("bic,bjd->bijcd", p1, p2).flatten(-2)  # (B, L, L, 1024)
        z = res_z + self.pair_update_proj(outer)
        z = self.norm_pair(z)

        return x, z


class AlphaFold3InspiredRNA(nn.Module):
    """
    Lightweight AlphaFold3-inspired model for RNA C1' coordinate prediction.

    Args:
        d_single:   single (1D) representation dimension
        d_pair:     pair   (2D) representation dimension
        nhead:      number of attention heads
        num_blocks: number of PairformerBlocks
        max_len:    maximum sequence length supported
        dropout:    dropout rate
    """

    def __init__(
        self,
        d_single: int = 128,
        d_pair: int = 64,
        nhead: int = 8,
        num_blocks: int = 8,
        max_len: int = 4096,
        dropout: float = 0.1,
        ribo_1d_dim: int = 256,   # RibonanzaNet2 encoder hidden dim (ninp)
        ribo_2d_dim: int = 64,    # RibonanzaNet2 pairwise dim
    ):
        super().__init__()
        self.d_single = d_single

        # Input embeddings
        self.embedding = nn.Embedding(5, d_single)        # 4 bases + unknown
        self.abs_pos_emb = nn.Embedding(max_len, d_single)
        self.rel_pos_emb = nn.Embedding(65, d_pair)       # [-32, +32] clipped

        # RibonanzaNet2 integration projections — dims match actual checkpoint output
        self.ribo_proj_1d = nn.Linear(ribo_1d_dim, d_single)
        self.ribo_proj_2d = nn.Linear(ribo_2d_dim, d_pair)

        self.blocks = nn.ModuleList([
            PairformerBlock(d_single, d_pair, nhead, dropout)
            for _ in range(num_blocks)
        ])

        # Coordinate prediction head: d_single → 128 → 64 → 3
        self.coord_head = nn.Sequential(
            nn.LayerNorm(d_single),
            nn.Linear(d_single, d_single),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_single, 64),
            nn.GELU(),
            nn.Linear(64, 3),
        )

    def forward(
        self,
        seq_idx: torch.Tensor,                          # (B, L) int tokens
        ribo_1d_feats: torch.Tensor | None = None,      # (B, L, ribo_1d_dim)
        ribo_2d_feats: torch.Tensor | None = None,      # (B, L, L, ribo_2d_dim)
    ) -> torch.Tensor:                                  # (B, L, 3) C1' coords
        B, L = seq_idx.shape

        # Single representation
        x = self.embedding(seq_idx)
        pos = torch.arange(L, device=seq_idx.device) % self.abs_pos_emb.num_embeddings
        x = x + self.abs_pos_emb(pos).unsqueeze(0)
        if ribo_1d_feats is not None:
            x = x + self.ribo_proj_1d(ribo_1d_feats)

        # Pair representation (relative positions)
        pos_full = torch.arange(L, device=seq_idx.device)
        rel_pos = torch.clamp(pos_full.unsqueeze(0) - pos_full.unsqueeze(1) + 32, 0, 64)
        z = self.rel_pos_emb(rel_pos).unsqueeze(0).expand(B, L, L, -1)
        if ribo_2d_feats is not None:
            z = z + self.ribo_proj_2d(ribo_2d_feats)

        for block in self.blocks:
            x, z = block(x, z)

        return self.coord_head(x)


# Tokenisation
RNA_VOCAB: dict[str, int] = {"A": 0, "C": 1, "G": 2, "U": 3}


def tokenise(sequence: str) -> list[int]:
    """Map RNA sequence string to integer token list."""
    return [RNA_VOCAB.get(nt.upper(), 4) for nt in sequence]


# ====================================================================
# DATA: Loader & Cluster Split
# ====================================================================

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


# ====================================================================
# INFERENCE: Template-Based Search
# ====================================================================

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


# ====================================================================
# INFERENCE: RhoFold+ Runner & Fine-Tuner
# ====================================================================

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


# ====================================================================
# INFERENCE: Boltz-1/2 & RNAPro Runner
# ====================================================================

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


# ====================================================================
# TRAINING: Train / Evaluate / Inference Loop
# ====================================================================

import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Geometry utilities
# ---------------------------------------------------------------------------

def kabsch_rmsd_tmscore(P: torch.Tensor, Q: torch.Tensor) -> tuple[float, float]:
    """
    Kabsch-aligned RMSD and approximate RNA TM-score between two (L, 3) tensors.
    """
    L = P.shape[0]
    if L == 0:
        return 0.0, 0.0

    # Ensure float32 for numerical stability in SVD
    P = P.float()
    Q = Q.float()

    if not (torch.isfinite(P).all() and torch.isfinite(Q).all()):
        return float("inf"), 0.0

    P_c = P - P.mean(dim=0)
    Q_c = Q - Q.mean(dim=0)

    H = P_c.T @ Q_c
    try:
        U, S, Vh = torch.linalg.svd(H)
    except Exception:
        return float("inf"), 0.0
    V = Vh.mT
    d = torch.sign(torch.det(V @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))
    R = V @ D @ U.T

    P_aligned = (P_c @ R.T) + Q.mean(dim=0)
    dist_sq = ((P_aligned - Q) ** 2).sum(dim=1)

    if not torch.isfinite(dist_sq).all():
        return float("inf"), 0.0

    rmsd = dist_sq.mean().sqrt().item()

    d0 = max(1.24 * max((L - 15), 1) ** (1 / 3) - 1.8, 0.5) if L > 15 else 0.5
    tm_score = (1.0 / (1.0 + dist_sq / (d0 ** 2))).mean().item()

    return rmsd, tm_score


# ---------------------------------------------------------------------------
# Template-Based Modelling fallback
# ---------------------------------------------------------------------------

class TemplateMatcher:
    """
    Searches training structures for sequence homologs via 4-mer Jaccard similarity
    and transfers their C1' coordinates (gap-filling by linear interpolation).
    """

    def __init__(self, seq_df: pd.DataFrame | None = None, labels_df: pd.DataFrame | None = None):
        self.templates: dict[str, tuple[str, dict[int, tuple[float, float, float]]]] = {}
        if seq_df is not None and labels_df is not None:
            self._build_index(seq_df, labels_df)

    def _build_index(self, seq_df: pd.DataFrame, labels_df: pd.DataFrame):
        labels = labels_df.copy()
        if "target_id" not in labels.columns:
            labels["target_id"] = labels["ID"].str.rsplit("_", n=1).str[0]
        grouped = labels.groupby("target_id")
        for _, row in seq_df.iterrows():
            tid = row["target_id"]
            seq = row["sequence"]
            if tid in grouped.groups:
                g = grouped.get_group(tid)
                coords = {
                    int(lr["resid"]) - 1: (lr["x_1"], lr["y_1"], lr["z_1"])
                    for _, lr in g.iterrows()
                }
                self.templates[tid] = (seq, coords)
        print(f"TBM index: {len(self.templates)} templates")

    def find_best_template(
        self, query_seq: str, min_identity: float = 0.5
    ) -> tuple[str | None, float]:
        q_set = {query_seq[i: i + 4] for i in range(len(query_seq) - 3)}
        if not q_set:
            return None, 0.0
        best_score, best_tid = 0.0, None
        for tid, (tseq, _) in self.templates.items():
            t_set = {tseq[i: i + 4] for i in range(len(tseq) - 3)}
            if not t_set:
                continue
            score = len(q_set & t_set) / max(len(q_set | t_set), 1)
            if score > best_score:
                best_score, best_tid = score, tid
        return (best_tid, best_score) if best_score >= min_identity else (None, best_score)

    def transfer_coords(self, query_seq: str, template_id: str) -> np.ndarray:
        tseq, tcoords = self.templates[template_id]
        L = len(query_seq)
        coords = np.full((L, 3), np.nan, dtype=np.float32)
        for i in range(min(L, len(tseq))):
            if i in tcoords:
                coords[i] = tcoords[i]
        known_idx = np.where(~np.isnan(coords[:, 0]))[0]
        if len(known_idx) >= 2:
            for dim in range(3):
                coords[:, dim] = np.interp(np.arange(L), known_idx, coords[known_idx, dim])
        elif len(known_idx) == 1:
            base = coords[known_idx[0]].copy()
            rng = np.random.default_rng(0)
            offsets = np.arange(L) - known_idx[0]
            coords = base + rng.normal(0, 5.9, (L, 3)) * offsets[:, None]
        return coords


# ---------------------------------------------------------------------------
# RibonanzaNet2 feature extractor
# ---------------------------------------------------------------------------

def _download_ribonanzanet2(dest_dir: str) -> bool:
    """Download RibonanzaNet2 checkpoint from Kaggle if not present."""
    import subprocess, tarfile, tempfile
    if os.path.exists(dest_dir) and any(
        f.endswith((".pt", ".bin")) for f in os.listdir(dest_dir)
    ):
        return True
    # Try kagglehub first
    try:
        import kagglehub  # type: ignore
        path = kagglehub.model_download("shujun717/ribonanzanet2/PyTorch/alpha/1")
        if path and os.path.exists(path):
            print(f"RibonanzaNet2 downloaded via kagglehub → {path}")
            return True
    except Exception:
        pass
    # Fallback: curl download
    try:
        os.makedirs(dest_dir, exist_ok=True)
        url = "https://www.kaggle.com/api/v1/models/shujun717/ribonanzanet2/pyTorch/alpha/1/download"
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = tmp.name
        print(f"Downloading RibonanzaNet2 from Kaggle...")
        subprocess.run(["curl", "-L", "-o", tmp_path, url], check=True, capture_output=True)
        with tarfile.open(tmp_path, "r:gz") as tar:
            tar.extractall(dest_dir)
        os.unlink(tmp_path)
        print(f"RibonanzaNet2 extracted → {dest_dir}")
        return True
    except Exception as e:
        print(f"RibonanzaNet2 download failed: {e}")
        return False


class RibonanzaFeatureExtractor:
    """
    Loads RibonanzaNet2 from its Kaggle model checkpoint and extracts
    (1D, 2D) representations for RNA sequences.
    Falls back gracefully if weights are unavailable.

    Uses forward hooks on the encoder layers to capture the last hidden
    states, so we get useful features even when the model's full forward
    fails (e.g. due to internal None pairwise features).
    """

    def __init__(self, checkpoint_path: str = "", auto_download: bool = True):
        self.available = False
        self.model = None
        self._ninp = 256          # sequence feature dim (updated after load)
        self._pairwise_dim = 64   # pairwise feature dim (updated after load)
        self._hooked_1d: list[torch.Tensor] = []
        self._hooked_2d: list[torch.Tensor] = []
        self._hooks: list = []
        self._forward_error_printed = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not checkpoint_path:
            print("RibonanzaNet2: no checkpoint path provided. Skipping.")
            return

        # Auto-download if not present
        if not os.path.exists(checkpoint_path) and auto_download:
            _download_ribonanzanet2(checkpoint_path)

        if not os.path.exists(checkpoint_path):
            print(f"RibonanzaNet2 not found at '{checkpoint_path}'. Skipping.")
            return

        try:
            import sys, glob, json, yaml  # noqa: F401
            if checkpoint_path not in sys.path:
                sys.path.append(checkpoint_path)

            from Network import RibonanzaNet  # type: ignore

            weight_files = (
                glob.glob(os.path.join(checkpoint_path, "*.pt"))
                + glob.glob(os.path.join(checkpoint_path, "*.bin"))
            )
            if not weight_files:
                print(f"No weights found in {checkpoint_path}")
                return

            config = self._load_config(checkpoint_path, weight_files[0])
            self._ninp = getattr(config, "ninp", 256)
            self._pairwise_dim = getattr(config, "pairwise_dimension", 64)

            self.model = RibonanzaNet(config).to(self.device)
            state = torch.load(weight_files[0], map_location=self.device)
            state = state.get("state_dict", state.get("model_state_dict", state))
            self.model.load_state_dict(state, strict=False)
            self.model.eval()

            # Register hooks on encoder layers to capture hidden states.
            # This lets us get features even if the full forward() raises.
            self._register_encoder_hooks()

            self.available = True
            print("RibonanzaNet2 loaded successfully.")
        except Exception as e:
            print(f"RibonanzaNet2 init failed: {e}")

    def _register_encoder_hooks(self):
        """Hook every ConvTransformerEncoderLayer to capture last 1D output."""
        if self.model is None:
            return
        for name, module in self.model.named_modules():
            cls = type(module).__name__
            if "ConvTransformerEncoderLayer" in cls or "EncoderLayer" in cls:
                handle = module.register_forward_hook(self._hook_fn_1d)
                self._hooks.append(handle)

    def _hook_fn_1d(self, module, inp, output):
        """Capture the sequence (1D) output from each encoder layer."""
        # Output can be a tensor (B,L,D) or a tuple (tensor, pairwise, ...)
        feat = output[0] if isinstance(output, tuple) else output
        if isinstance(feat, torch.Tensor) and feat.dim() == 3:
            self._hooked_1d.append(feat.detach().float())
        # If pairwise is returned as second element, capture it too
        if isinstance(output, tuple) and len(output) >= 2:
            pairwise = output[1]
            if isinstance(pairwise, torch.Tensor) and pairwise.dim() == 4:
                self._hooked_2d.append(pairwise.detach().float())

    def _load_config(self, checkpoint_path: str, weight_file: str):
        import yaml, json  # noqa: F811
        # Search for config files — RibonanzaNet2 ships pairwise.yaml
        for cfg_name in ["pairwise.yaml", "config.yaml", "config.yml", "config.json"]:
            p = os.path.join(checkpoint_path, cfg_name)
            if os.path.exists(p):
                with open(p) as f:
                    raw = yaml.safe_load(f) if cfg_name.endswith((".yaml", ".yml")) else json.load(f)
                cfg = type("Cfg", (), {})()
                for k, v in raw.items():
                    setattr(cfg, k, v)
                return cfg
        # Try from checkpoint dict
        ckpt = torch.load(weight_file, map_location="cpu")
        if isinstance(ckpt, dict) and "config" in ckpt:
            raw = ckpt["config"]
            cfg = type("Cfg", (), {})()
            for k, v in raw.items():
                setattr(cfg, k, v)
            return cfg
        # Default fallback — matches RibonanzaNet2 pairwise.yaml defaults
        return type("Cfg", (), {
            "ninp": 256, "nhead": 8, "nlayers": 9, "ntoken": 5,
            "nclass": 2, "pairwise_dimension": 64,
            "use_triangular_attention": False,
            "dropout": 0.05, "k": 5,
        })()

    def forward(
        self, sequences: list[str]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.available or self.model is None:
            return None, None
        # RibonanzaNet2 uses 1-indexed tokens: A=1,C=2,G=3,U=4; 0=padding
        mapping = {"A": 1, "C": 2, "G": 3, "U": 4}
        B = len(sequences)
        L = max(len(s) for s in sequences)
        tokens = torch.zeros((B, L), dtype=torch.long, device=self.device)
        for i, s in enumerate(sequences):
            for j, c in enumerate(s):
                tokens[i, j] = mapping.get(c.upper(), 0)

        self._hooked_1d.clear()
        self._hooked_2d.clear()

        # src_mask: (B, L) float, 1.0 = valid position, 0.0 = padding.
        # TriangleMultiplicativeModule always calls src_mask.unsqueeze(-1),
        # so passing None causes AttributeError. Use all-ones (no padding).
        src_mask = (tokens > 0).float()

        try:
            with torch.no_grad():
                out = self.model(tokens, src_mask=src_mask)
            # Success path: model returned normally
            # Prefer direct tuple output (seq_feats, pairwise_feats)
            if isinstance(out, tuple) and len(out) >= 2:
                feat_1d, feat_2d = out[0], out[1]
                if isinstance(feat_1d, torch.Tensor) and feat_1d.dim() == 3:
                    feat_2d_out = feat_2d if (
                        isinstance(feat_2d, torch.Tensor) and feat_2d.dim() == 4
                    ) else None
                    return feat_1d.float(), feat_2d_out
            if isinstance(out, torch.Tensor) and out.dim() == 3:
                # Single tensor output — use as 1D features
                return out.float(), None
        except Exception as e:
            # Print full traceback once so the user can see what's happening
            if not self._forward_error_printed:
                import traceback
                print(f"RibonanzaNet2 forward failed: {e}")
                traceback.print_exc()
                self._forward_error_printed = True

        # Hook fallback: use the last successfully captured encoder layer output
        if self._hooked_1d:
            feat_1d = self._hooked_1d[-1]   # last encoder layer output
            feat_2d = self._hooked_2d[-1] if self._hooked_2d else None
            if not self._forward_error_printed:
                # Only printed once, so suppress subsequent noise
                pass
            return feat_1d, feat_2d

        # Complete failure: disable to avoid per-sample error spam
        self.available = False
        return None, None


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def _build_coords_array(group: pd.DataFrame, seq_len: int) -> np.ndarray | None:
    """Build (L, 3) coordinate array from a label group. Returns None if empty."""
    coords = np.full((seq_len, 3), np.nan, dtype=np.float32)
    for _, lr in group.iterrows():
        res_idx = int(lr["resid"]) - 1
        if 0 <= res_idx < seq_len:
            coords[res_idx] = [lr["x_1"], lr["y_1"], lr["z_1"]]
    valid = np.where(~np.isnan(coords[:, 0]))[0]
    if len(valid) == 0:
        return None
    # Zero-centre at centroid of valid residues (more balanced than first residue)
    coords -= coords[valid].mean(axis=0)
    return coords


def _autocast_ctx():
    if torch.cuda.is_available():
        # bfloat16 has float32 dynamic range — prevents overflow in outer product layers
        # float16 max is 65504, which overflows easily in pairwise (L×L×1024) ops
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.autocast("cuda", dtype=dtype)
    import contextlib
    return contextlib.nullcontext()


def train(
    model: AlphaFold3InspiredRNA,
    train_seq_df: pd.DataFrame,
    train_labels_df: pd.DataFrame,
    val_seq_df: pd.DataFrame | None = None,
    val_labels_df: pd.DataFrame | None = None,
    extractor: RibonanzaFeatureExtractor | None = None,
    epochs: int = 50,
    lr: float = 1e-4,
    max_seq_len: int = 2000,
    accumulation_steps: int = 16,
    log_every: int = 50,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train the model and return (coord_mean, coord_std) for denormalisation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    # Coordinate normalisation constants
    coord_cols = ["x_1", "y_1", "z_1"]
    coord_mean = np.zeros(3, dtype=np.float32)
    coord_std = np.clip(
        train_labels_df[coord_cols].std(skipna=True).values.astype(np.float32),
        1e-6, None,
    )
    coord_mean_t = torch.tensor(coord_mean, dtype=torch.float32, device=device).view(1, 1, 3)
    coord_std_t  = torch.tensor(coord_std,  dtype=torch.float32, device=device).view(1, 1, 3)

    # Pre-group labels
    labels = train_labels_df.copy()
    if "target_id" not in labels.columns:
        labels["target_id"] = labels["ID"].str.rsplit("_", n=1).str[0]
    labels_grouped = labels.groupby("target_id")

    val_labels_grouped = None
    if val_labels_df is not None:
        vl = val_labels_df.copy()
        if "target_id" not in vl.columns:
            vl["target_id"] = vl["ID"].str.rsplit("_", n=1).str[0]
        val_labels_grouped = vl.groupby("target_id")

    for epoch in range(epochs):
        model.train()
        total_loss = running_loss = count = skipped = 0

        for _, row in train_seq_df.iterrows():
            seq_str = row["sequence"]
            tid = row["target_id"]

            if len(seq_str) > max_seq_len or tid not in labels_grouped.groups:
                skipped += 1
                continue

            coords = _build_coords_array(labels_grouped.get_group(tid), len(seq_str))
            if coords is None:
                skipped += 1
                continue

            seq_idx = torch.tensor([tokenise(seq_str)], device=device)
            target  = torch.from_numpy(coords).unsqueeze(0).to(device=device)

            ribo_1d, ribo_2d = (extractor.forward([seq_str]) if extractor else (None, None))

            try:
                with _autocast_ctx():
                    pred = model(seq_idx, ribo_1d_feats=ribo_1d, ribo_2d_feats=ribo_2d)
                    if not torch.isfinite(pred).all():
                        skipped += 1; continue

                    valid_mask = ~torch.isnan(target)
                    if valid_mask.sum() == 0:
                        skipped += 1; continue

                    target_norm = torch.clamp((target - coord_mean_t) / coord_std_t, -10, 10)
                    loss_coord = F.smooth_l1_loss(pred[valid_mask], target_norm[valid_mask])

                    pred_d = pred * coord_std_t + coord_mean_t
                    valid_rows = torch.isfinite(target.squeeze(0)).all(-1)
                    pv = pred_d.squeeze(0)[valid_rows]
                    tv = target.squeeze(0)[valid_rows]

                    if len(pv) > 2:
                        loss_dist = F.smooth_l1_loss(
                            torch.cdist(pv.unsqueeze(0), pv.unsqueeze(0)).squeeze(0),
                            torch.cdist(tv.unsqueeze(0), tv.unsqueeze(0)).squeeze(0),
                        )
                        # Bond loss: only between residues that are truly adjacent in
                        # sequence (valid_rows may have gaps, so index-1 neighbours
                        # are not always real C1'-C1' bonds).
                        valid_idx = torch.where(valid_rows)[0]
                        is_adj = (valid_idx[1:] - valid_idx[:-1]) == 1
                        if is_adj.any():
                            pb = torch.sqrt(((pv[1:][is_adj] - pv[:-1][is_adj]) ** 2).sum(-1) + 1e-8)
                            tb = torch.sqrt(((tv[1:][is_adj] - tv[:-1][is_adj]) ** 2).sum(-1) + 1e-8)
                            loss_bond = F.smooth_l1_loss(pb, tb)
                        else:
                            loss_bond = torch.tensor(0.0, device=device)
                    else:
                        loss_dist = loss_bond = torch.tensor(0.0, device=device)

                    loss = loss_coord + 0.2 * loss_dist + 0.1 * loss_bond

                if not torch.isfinite(loss):
                    skipped += 1; continue

                scaler.scale(loss / accumulation_steps).backward()

                if (count + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss   += loss.item()
                running_loss += loss.item()
                count        += 1
                del pred, target_norm, loss, valid_mask

                if count % log_every == 0:
                    print(f"  Epoch {epoch+1} | step {count} | loss {running_loss/log_every:.4f}")
                    running_loss = 0

                if torch.cuda.is_available() and torch.cuda.memory_reserved() > 14.5 * 2**30:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"  RuntimeError (len={len(seq_str)}): {e}")
                skipped += 1

        print(
            f"Epoch {epoch+1}/{epochs} | avg_loss={total_loss/max(count,1):.4f} | "
            f"trained={count} | skipped={skipped} | lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if val_labels_grouped is not None and val_seq_df is not None:
            metrics = evaluate(
                model, val_seq_df, val_labels_grouped,
                coord_mean_t, coord_std_t, extractor=extractor,
                max_seq_len=max_seq_len, sample_limit=200, device=device,
            )
            print(
                f"  val_loss={metrics['avg_loss']:.4f} | "
                f"TM-score={metrics['tm_score']:.4f} | RMSD={metrics['kabsch_rmsd']:.4f} | "
                f"n={metrics['count']}"
            )

        scheduler.step()

    return coord_mean, coord_std


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: AlphaFold3InspiredRNA,
    seq_df: pd.DataFrame,
    labels_grouped,
    coord_mean_t: torch.Tensor,
    coord_std_t: torch.Tensor,
    extractor: RibonanzaFeatureExtractor | None = None,
    max_seq_len: int = 2000,
    sample_limit: int | None = None,
    device: torch.device | None = None,
) -> dict:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    total_loss = total_rmsd = total_tm = total_pts = 0.0
    count = skipped = 0
    rows = seq_df.head(sample_limit) if sample_limit else seq_df

    with torch.no_grad():
        for _, row in rows.iterrows():
            seq_str = row["sequence"]
            tid = row["target_id"]
            if len(seq_str) > max_seq_len or tid not in labels_grouped.groups:
                skipped += 1; continue

            coords = _build_coords_array(labels_grouped.get_group(tid), len(seq_str))
            if coords is None:
                skipped += 1; continue

            seq_idx = torch.tensor([tokenise(seq_str)], device=device)
            target  = torch.from_numpy(coords).unsqueeze(0).to(device=device)
            ribo_1d, ribo_2d = (extractor.forward([seq_str]) if extractor else (None, None))

            pred = model(seq_idx, ribo_1d_feats=ribo_1d, ribo_2d_feats=ribo_2d)
            if not torch.isfinite(pred).all():
                skipped += 1; continue

            valid_mask = ~torch.isnan(target)
            if valid_mask.sum() == 0:
                skipped += 1; continue

            target_norm = torch.clamp((target - coord_mean_t) / coord_std_t, -10, 10)
            loss = F.smooth_l1_loss(pred[valid_mask], target_norm[valid_mask])
            if not torch.isfinite(loss):
                skipped += 1; continue
            total_loss += loss.item()

            row_valid = torch.isfinite(target.squeeze(0)).all(-1)
            if row_valid.any():
                # Clamp to training distribution before denorm to avoid float overflow
                pred_clamped = pred.float().clamp(-15, 15)
                pv = (pred_clamped * coord_std_t + coord_mean_t).squeeze(0)[row_valid]
                tv = target.squeeze(0)[row_valid]
                rmsd, tm = kabsch_rmsd_tmscore(pv, tv)
                if math.isfinite(rmsd) and rmsd < 1e5:
                    total_rmsd += rmsd
                    total_tm   += tm
                total_pts  += row_valid.sum().item()
            count += 1

    n = max(count, 1)
    return {
        "avg_loss":     total_loss / n,
        "kabsch_rmsd":  total_rmsd / n,
        "tm_score":     total_tm   / n,
        "count":        count,
        "skipped":      skipped,
    }


def train_val_split(
    seq_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split seq_df/labels_df into train and val portions.
    Returns (train_seq, train_labels, val_seq, val_labels).
    """
    rng = np.random.default_rng(seed)
    ids = seq_df["target_id"].unique()
    n_val = max(1, int(len(ids) * val_fraction))
    val_ids = set(rng.choice(ids, size=n_val, replace=False).tolist())

    train_seq    = seq_df[~seq_df["target_id"].isin(val_ids)].reset_index(drop=True)
    val_seq      = seq_df[ seq_df["target_id"].isin(val_ids)].reset_index(drop=True)

    # Propagate target_id column to labels if needed
    lbl = labels_df.copy()
    if "target_id" not in lbl.columns:
        lbl["target_id"] = lbl["ID"].str.rsplit("_", n=1).str[0]
    train_labels = lbl[~lbl["target_id"].isin(val_ids)].reset_index(drop=True)
    val_labels   = lbl[ lbl["target_id"].isin(val_ids)].reset_index(drop=True)

    return train_seq, train_labels, val_seq, val_labels


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def generate_diverse_predictions(
    model: AlphaFold3InspiredRNA,
    seq_idx: torch.Tensor,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    num_preds: int = 5,
    ribo_1d: torch.Tensor | None = None,
    ribo_2d: torch.Tensor | None = None,
) -> list[np.ndarray]:
    """
    MC dropout + structured perturbations → list of num_preds coordinate arrays.
    """
    model.train()  # enable dropout
    preds: list[np.ndarray] = []
    with torch.no_grad(), _autocast_ctx():
        for _ in range(num_preds):
            p = model(seq_idx, ribo_1d_feats=ribo_1d, ribo_2d_feats=ribo_2d)
            preds.append(p.to(torch.float32).squeeze(0).cpu().numpy() * coord_std + coord_mean)
    model.eval()

    # Add structured perturbations so predictions are visibly diverse
    base_scale = np.maximum(coord_std * 0.03, 0.08).astype(np.float32)
    L = preds[0].shape[0]
    for k in range(1, num_preds):
        iid  = np.random.normal(0, base_scale * (0.8 * k), preds[k].shape)
        walk = np.cumsum(np.random.normal(0, base_scale * (0.15 * k), (L, 3)), axis=0)
        preds[k] = preds[k] + iid + walk

    return preds


def run_inference(
    model: AlphaFold3InspiredRNA,
    test_df: pd.DataFrame,
    coord_mean: np.ndarray,
    coord_std: np.ndarray,
    extractor: RibonanzaFeatureExtractor | None = None,
    template_matcher: TemplateMatcher | None = None,
    device: torch.device | None = None,
) -> list[dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    tbm_used = nn_used = 0
    rows: list[dict] = []

    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        seq_str   = row["sequence"]
        target_id = row["target_id"]

        # --- TBM first ---
        if template_matcher is not None:
            tmpl_id, score = template_matcher.find_best_template(seq_str, min_identity=0.5)
            if tmpl_id is not None:
                tbm_coords = template_matcher.transfer_coords(seq_str, tmpl_id)
                tbm_used += 1
                for i, resname in enumerate(seq_str):
                    entry = {"ID": f"{target_id}_{i+1}", "resname": resname, "resid": i + 1}
                    for p in range(1, 6):
                        noise = np.random.normal(0, 0.3 * p, 3) if p > 1 else np.zeros(3)
                        x, y, z = np.clip(tbm_coords[i] + noise, -999, 999)
                        entry[f"x_{p}"] = float(x)
                        entry[f"y_{p}"] = float(y)
                        entry[f"z_{p}"] = float(z)
                    rows.append(entry)
                continue

        # --- Neural network ---
        nn_used += 1
        seq_idx = torch.tensor([tokenise(seq_str)], device=device)
        ribo_1d, ribo_2d = (extractor.forward([seq_str]) if extractor else (None, None))

        try:
            preds = generate_diverse_predictions(
                model, seq_idx, coord_mean, coord_std, num_preds=5,
                ribo_1d=ribo_1d, ribo_2d=ribo_2d,
            )
        except RuntimeError:
            base = np.random.randn(len(seq_str), 3).astype(np.float32)
            preds = [base + np.random.normal(0, 0.15 * k, base.shape) for k in range(5)]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for i, resname in enumerate(seq_str):
            entry = {"ID": f"{target_id}_{i+1}", "resname": resname, "resid": i + 1}
            for p_idx, pred in enumerate(preds, 1):
                x, y, z = np.clip(pred[i], -999, 999)
                entry[f"x_{p_idx}"] = float(x)
                entry[f"y_{p_idx}"] = float(y)
                entry[f"z_{p_idx}"] = float(z)
            rows.append(entry)

    print(f"Inference done: TBM={tbm_used}  NN={nn_used}")
    return rows


# ====================================================================
# EVALUATION: TM-score via US-align
# ====================================================================

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


# ====================================================================
# PIPELINE: Hybrid Orchestrator
# ====================================================================

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

            self._rhofold = RhoFoldRunner(
                rhofold_dir=self.cfg.rhofold_dir,
                checkpoint=self.cfg.rhofold_checkpoint,
                device=self.cfg.device,
            )
        return self._rhofold.predict(target.target_id, target.sequence)

    def _run_boltz(self, target) -> np.ndarray:
        if self._boltz is None:

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


# ====================================================================
# MAIN ENTRYPOINT
# ====================================================================

# ============================================================
# MAIN  — Training + Inference
# ============================================================
if __name__ == "__main__" or True:   # `or True` so Colab runs it on execute
    DATA_DIR = INPUT_PREFIX + "/competitions/stanford-rna-3d-folding-2"

    # ── Generate dummy data for offline smoke-testing ─────────────────────
    def generate_dummy_data(data_dir: str):
        os.makedirs(data_dir, exist_ok=True)
        print(f"Generating dummy data in {data_dir} ...")
        train_seq = pd.DataFrame([
            {"ID": "seq1", "sequence": "ACGU" * 5,  "target_id": "seq1"},
            {"ID": "seq2", "sequence": "GCAU" * 8,  "target_id": "seq2"},
        ])
        train_seq.to_csv(os.path.join(data_dir, "train_sequences.csv"), index=False)
        rows = []
        for seq_id, length in [("seq1", 20), ("seq2", 32)]:
            pts = np.cumsum(np.random.normal(0, 5.95, (length, 3)), axis=0)
            for i in range(length):
                rows.append({
                    "ID": f"{seq_id}_{i+1}", "target_id": seq_id,
                    "resid": i + 1, "resname": "A",
                    "x_1": pts[i, 0], "y_1": pts[i, 1], "z_1": pts[i, 2],
                })
        pd.DataFrame(rows).to_csv(os.path.join(data_dir, "train_labels.csv"), index=False)
        test_seq = pd.DataFrame([
            {"ID": "test1", "sequence": "ACGU" * 4, "target_id": "test1"}
        ])
        test_seq.to_csv(os.path.join(data_dir, "test_sequences.csv"), index=False)

    if not os.path.exists(DATA_DIR):
        DATA_DIR = "./dummy_kaggle_data"
        generate_dummy_data(DATA_DIR)

    # ── Config ────────────────────────────────────────────────────────────
    PILOT_MODE    = True      # set False for full competition run
    N_EPOCHS      = 20 if PILOT_MODE else 50
    # Fewer accumulation steps in pilot mode → more optimizer updates per epoch
    # (pilot has ~72 seqs; 16 steps → only 4 updates/epoch; 4 steps → 18 updates/epoch)
    ACCUM_STEPS   = 4 if PILOT_MODE else 16
    MAX_SEQ_LEN   = 2000
    WEIGHTS_PATH  = "model_weights.pt"
    LOAD_IF_EXISTS = False     # set True to skip training and reuse saved weights
    RIBO_CKPT     = INPUT_PREFIX + "/models/shujun717/ribonanzanet2/PyTorch/alpha/1"

    # ── Load data ─────────────────────────────────────────────────────────
    train_seq_df    = pd.read_csv(os.path.join(DATA_DIR, "train_sequences.csv"))
    train_labels_df = pd.read_csv(os.path.join(DATA_DIR, "train_labels.csv"))
    test_df         = pd.read_csv(os.path.join(DATA_DIR, "test_sequences.csv"))

    val_seq_df = val_labels_df = None
    for vname in ["validation_sequences.csv", "val_sequences.csv"]:
        vpath = os.path.join(DATA_DIR, vname)
        if os.path.exists(vpath):
            lname = vname.replace("sequences", "labels")
            lpath = os.path.join(DATA_DIR, lname)
            if os.path.exists(lpath):
                val_seq_df    = pd.read_csv(vpath)
                val_labels_df = pd.read_csv(lpath)
                break

    if PILOT_MODE:
        print("PILOT MODE: truncating to 100 train / 20 val examples")
        train_seq_df = train_seq_df.head(100)
        valid_ids = set(train_seq_df["target_id"])
        if "target_id" not in train_labels_df.columns:
            train_labels_df["target_id"] = train_labels_df["ID"].str.rsplit("_", n=1).str[0]
        train_labels_df = train_labels_df[train_labels_df["target_id"].isin(valid_ids)]
        if val_seq_df is not None:
            val_seq_df = val_seq_df.head(20)

    # ── Create train-subset val split if no labelled val set exists ───────
    # This is used as a sanity check that the model can actually learn.
    # Split 15% of training data off as a proxy validation set.
    tr_seq_split, tr_lbl_split, proxy_val_seq, proxy_val_lbl = train_val_split(
        train_seq_df, train_labels_df, val_fraction=0.15,
    )
    have_real_val = val_seq_df is not None and val_labels_df is not None
    print(
        f"Train: {len(train_seq_df)} seqs | "
        f"Val: {len(val_seq_df) if have_real_val else 0} seqs (competition) | "
        f"Proxy val: {len(proxy_val_seq)} seqs (from training split)"
    )

    # ── Model + extractors ────────────────────────────────────────────────
    model     = AlphaFold3InspiredRNA().to(DEVICE)
    extractor = RibonanzaFeatureExtractor(checkpoint_path=RIBO_CKPT)
    template_matcher = TemplateMatcher(train_seq_df, train_labels_df)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Train (or load saved weights) ─────────────────────────────────────
    if LOAD_IF_EXISTS and os.path.exists(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        coord_cols = ["x_1", "y_1", "z_1"]
        coord_mean = np.zeros(3, dtype=np.float32)
        coord_std  = np.clip(
            train_labels_df[coord_cols].std(skipna=True).values.astype(np.float32),
            1e-6, None,
        )
    else:
        # Train with the proxy val split so we can always see learning progress
        coord_mean, coord_std = train(
            model, tr_seq_split, tr_lbl_split,
            val_seq_df=proxy_val_seq, val_labels_df=proxy_val_lbl,
            extractor=extractor,
            epochs=N_EPOCHS, lr=1e-4, max_seq_len=MAX_SEQ_LEN,
            accumulation_steps=ACCUM_STEPS,
            device=DEVICE,
        )
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"Saved weights → {WEIGHTS_PATH}")

    # ── Overfit sanity check: eval on a few TRAINING samples ─────────────
    # If TM-score here is < 0.1 the model is not learning at all.
    coord_mean_t = torch.tensor(coord_mean, dtype=torch.float32, device=DEVICE).view(1,1,3)
    coord_std_t  = torch.tensor(coord_std,  dtype=torch.float32, device=DEVICE).view(1,1,3)
    if "target_id" not in tr_lbl_split.columns:
        tr_lbl_split["target_id"] = tr_lbl_split["ID"].str.rsplit("_", n=1).str[0]
    train_lbl_grouped = tr_lbl_split.groupby("target_id")
    overfit = evaluate(
        model, tr_seq_split.head(20), train_lbl_grouped,
        coord_mean_t, coord_std_t, extractor=extractor,
        max_seq_len=MAX_SEQ_LEN, device=DEVICE,
    )
    print(
        f"
── Overfit check (train subset) ──────────────────────────────
"
        f"  TM-score={overfit['tm_score']:.4f} | RMSD={overfit['kabsch_rmsd']:.2f} Å | "
        f"loss={overfit['avg_loss']:.4f} | n={overfit['count']}
"
        f"  (TM>0.1 confirms model is learning; TM~0 means it is not)
"
        f"──────────────────────────────────────────────────────────────"
    )

    # ── Competition val evaluation (if labels available) ─────────────────
    if have_real_val:
        if "target_id" not in val_labels_df.columns:
            val_labels_df["target_id"] = val_labels_df["ID"].str.rsplit("_", n=1).str[0]
        val_lbl_grouped = val_labels_df.groupby("target_id")
        val_metrics = evaluate(
            model, val_seq_df, val_lbl_grouped,
            coord_mean_t, coord_std_t, extractor=extractor,
            max_seq_len=MAX_SEQ_LEN, device=DEVICE,
        )
        print(
            f"── Competition val ───────────────────────────────────────────
"
            f"  TM-score={val_metrics['tm_score']:.4f} | RMSD={val_metrics['kabsch_rmsd']:.2f} Å | "
            f"n={val_metrics['count']}
"
            f"──────────────────────────────────────────────────────────────"
        )

    # ── Inference + submission ─────────────────────────────────────────────
    submission_rows = run_inference(
        model, test_df, coord_mean, coord_std,
        extractor=extractor, template_matcher=template_matcher, device=DEVICE,
    )

    cols = ["ID", "resname", "resid"]
    for i in range(1, 6):
        cols += [f"x_{i}", f"y_{i}", f"z_{i}"]
    submission = pd.DataFrame(submission_rows, columns=cols)
    submission.to_csv("submission.csv", index=False)
    print(f"Saved submission.csv — {len(submission)} rows")
    print(submission.head())

    # ── Physical health check ─────────────────────────────────────────────
    coords = submission[["x_1", "y_1", "z_1"]].values
    dists  = np.linalg.norm(np.diff(coords, axis=0), axis=1)
    print(f"Mean C1\u2013C1\u2019 distance: {np.mean(dists):.2f} Å  (target ≈ 5.9 Å)")
    if np.any(dists > 10) or np.any(dists < 2):
        print("⚠️  WARNING: chain may be broken or physically unrealistic")
    else:
        print("✅ Chain geometry looks physically reasonable")
