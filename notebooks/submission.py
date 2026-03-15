"""
Kaggle Submission Notebook — Stanford RNA 3D Folding Part 2
===========================================================
This file is the Kaggle notebook entry point.

Kaggle code competition constraints:
  - No internet access during inference
  - All model weights must be attached as Kaggle datasets
  - GPU: 1x T4 (16 GB) or P100 (16 GB)
  - RAM: ~13 GB
  - Time limit: 9h

Run order:
  1. Install / setup dependencies (offline wheels in Kaggle dataset)
  2. Load pipeline config
  3. Load test sequences
  4. Run hybrid pipeline (template → RhoFold+ → Boltz fallback)
  5. Build and validate submission CSV
"""

# ── 0. Paths ────────────────────────────────────────────────────────────────

import os, sys

# Kaggle dataset mount points (update to match your attached datasets)
COMPETITION_DATA = "/kaggle/input/stanford-rna-3d-folding-2"
RHOFOLD_REPO     = "/kaggle/input/rhofold-repo/RhoFold"
RHOFOLD_WEIGHTS  = "/kaggle/input/rhofold-weights/rhofold_pretrained.pt"
PDB_DIR          = "/kaggle/input/pdb-rna-chains"
BLAST_DB         = "/kaggle/input/pdb-blast-db/pdb_rna"
BOLTZ_CACHE      = "/kaggle/input/boltz-weights"
OUTPUT_DIR       = "/kaggle/working"

# Add src to path (assumes this repo is attached as a Kaggle dataset)
REPO_ROOT = "/kaggle/input/kaggle-rna2-repo"
sys.path.insert(0, REPO_ROOT)

# ── 1. Install offline wheels ────────────────────────────────────────────────

# Wheels should be pre-downloaded and attached as a Kaggle dataset.
# Example: kaggle datasets download -d <your-username>/rna-offline-wheels
WHEELS_DIR = "/kaggle/input/rna-offline-wheels"
if os.path.exists(WHEELS_DIR):
    os.system(f"pip install --no-index --find-links={WHEELS_DIR} biopython pyyaml einops boltz -q")

# Install RhoFold dependencies from its requirements.txt
if os.path.exists(RHOFOLD_REPO):
    os.system(f"pip install -r {RHOFOLD_REPO}/requirements.txt -q")
    sys.path.insert(0, RHOFOLD_REPO)

# ── 2. Imports ───────────────────────────────────────────────────────────────

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.loader import load_targets, make_submission
from src.pipeline import HybridPipeline, PipelineConfig

# ── 3. Config ────────────────────────────────────────────────────────────────

cfg = PipelineConfig(
    # Template search
    use_template_search=os.path.exists(PDB_DIR),
    pdb_dir=PDB_DIR,
    blast_db=BLAST_DB,
    rfam_cm=None,                       # set to Rfam.cm path if available
    tm_threshold=0.45,

    # RhoFold+
    use_rhofold=os.path.exists(RHOFOLD_WEIGHTS),
    rhofold_dir=RHOFOLD_REPO,
    rhofold_checkpoint=RHOFOLD_WEIGHTS,

    # RibonanzaNet2 — disabled until fine-tuned checkpoint is available
    use_ribonanzanet2=False,

    # Boltz-1 fallback
    use_boltz=os.path.exists(BOLTZ_CACHE),
    boltz_model="boltz1",
    boltz_cache_dir=BOLTZ_CACHE,

    ensemble_method="best_of_5",
    n_structures=5,
    device="cuda",
    n_cpus=2,
)

print("Pipeline config:")
print(f"  template_search : {cfg.use_template_search}")
print(f"  rhofold         : {cfg.use_rhofold}")
print(f"  boltz           : {cfg.use_boltz}")

# ── 4. Load test targets ─────────────────────────────────────────────────────

test_csv = os.path.join(COMPETITION_DATA, "test_sequences.csv")
targets = load_targets(test_csv)
print(f"\nLoaded {len(targets)} test targets")
print(f"  seq len range: {min(len(t.sequence) for t in targets)} – "
      f"{max(len(t.sequence) for t in targets)}")

# ── 5. Run pipeline ──────────────────────────────────────────────────────────

pipeline = HybridPipeline(cfg)
predictions = pipeline.run(targets)

# ── 6. Build and validate submission ─────────────────────────────────────────

submission_path = os.path.join(OUTPUT_DIR, "submission.csv")
df = pipeline.save_submission(predictions, submission_path)

# Validation checks
print("\nSubmission validation:")
expected_cols = {"target_id", "residue_index",
                 "x_1", "y_1", "z_1",
                 "x_2", "y_2", "z_2",
                 "x_3", "y_3", "z_3",
                 "x_4", "y_4", "z_4",
                 "x_5", "y_5", "z_5"}
missing_cols = expected_cols - set(df.columns)
assert not missing_cols, f"Missing columns: {missing_cols}"
assert not df.isnull().any().any(), "NaN values found in submission!"
assert (df["residue_index"] > 0).all(), "residue_index must be 1-indexed"

n_targets_in_sub = df["target_id"].nunique()
print(f"  rows           : {len(df)}")
print(f"  unique targets : {n_targets_in_sub} / {len(targets)}")
print(f"  NaN check      : PASS")
print(f"  col check      : PASS")
print(f"\nSubmission saved to: {submission_path}")
print(df.head())
