#!/usr/bin/env python3
"""
export_colab.py
===============
Flatten the kaggle-rna2 src/ modules into a single self-contained Python file
that can be pasted into a Google Colab notebook or run directly.

Usage:
    python export_colab.py                         # → colab_export.py
    python export_colab.py --out my_notebook.py
    python export_colab.py --mode notebook         # → colab_export.ipynb

What it does:
  1. Emits a Colab-specific setup header (Drive mount, kagglehub download,
     pip installs, GPU check) — guarded by `if False:` so it can be toggled.
  2. Inlines each src/ module in dependency order, stripping `from src.X import Y`
     style imports (since everything is in the same namespace).
  3. Appends the main training + inference entrypoint.
  4. Optionally wraps the result as a .ipynb with one code cell per section.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Module dependency order (earlier = fewer dependencies)
# ---------------------------------------------------------------------------

# Each entry: (module_path_relative_to_src, display_title)
MODULE_ORDER = [
    ("models/pairformer.py",          "MODEL: AlphaFold3-Inspired Pairformer"),
    ("data/loader.py",                "DATA: Loader & Cluster Split"),
    ("inference/template_search.py",  "INFERENCE: Template-Based Search"),
    ("inference/rhofold_runner.py",   "INFERENCE: RhoFold+ Runner & Fine-Tuner"),
    ("inference/boltz_runner.py",     "INFERENCE: Boltz-1/2 & RNAPro Runner"),
    ("training/train.py",             "TRAINING: Train / Evaluate / Inference Loop"),
    ("evaluation/tm_score.py",        "EVALUATION: TM-score via US-align"),
    ("pipeline.py",                   "PIPELINE: Hybrid Orchestrator"),
]

# Regex patterns for imports that will be stripped (inlined instead)
_INTERNAL_IMPORT_RE = re.compile(
    r"^\s*(from\s+src[\.\w]+\s+import\s+.*|import\s+src[\.\w]+.*)\s*$",
    re.MULTILINE,
)
# Also strip bare `from __future__ import annotations` after the first occurrence
_FUTURE_IMPORT_RE = re.compile(r"^\s*from\s+__future__\s+import\s+annotations\s*$", re.MULTILINE)

# ---------------------------------------------------------------------------
# Colab header
# ---------------------------------------------------------------------------

COLAB_SETUP_HEADER = '''\
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
'''

# ---------------------------------------------------------------------------
# Main entrypoint (appended at the end)
# ---------------------------------------------------------------------------

MAIN_ENTRYPOINT = '''\
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
    N_EPOCHS      = 5 if PILOT_MODE else 50
    MAX_SEQ_LEN   = 2000
    WEIGHTS_PATH  = "model_weights.pt"
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

    print(f"Train: {len(train_seq_df)} seqs | Val: {len(val_seq_df) if val_seq_df is not None else 0} seqs")

    # ── Model + extractors ────────────────────────────────────────────────
    model     = AlphaFold3InspiredRNA().to(DEVICE)
    extractor = RibonanzaFeatureExtractor(checkpoint_path=RIBO_CKPT)
    template_matcher = TemplateMatcher(train_seq_df, train_labels_df)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # ── Train (or load saved weights) ─────────────────────────────────────
    if os.path.exists(WEIGHTS_PATH):
        print(f"Loading weights from {WEIGHTS_PATH}")
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
        coord_cols = ["x_1", "y_1", "z_1"]
        coord_mean = np.zeros(3, dtype=np.float32)
        coord_std  = np.clip(
            train_labels_df[coord_cols].std(skipna=True).values.astype(np.float32),
            1e-6, None,
        )
    else:
        coord_mean, coord_std = train(
            model, train_seq_df, train_labels_df,
            val_seq_df=val_seq_df, val_labels_df=val_labels_df,
            extractor=extractor,
            epochs=N_EPOCHS, lr=1e-4, max_seq_len=MAX_SEQ_LEN,
            device=DEVICE,
        )
        torch.save(model.state_dict(), WEIGHTS_PATH)
        print(f"Saved weights → {WEIGHTS_PATH}")

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
    print(f"Mean C1\\u2013C1\\u2019 distance: {np.mean(dists):.2f} Å  (target ≈ 5.9 Å)")
    if np.any(dists > 10) or np.any(dists < 2):
        print("⚠️  WARNING: chain may be broken or physically unrealistic")
    else:
        print("✅ Chain geometry looks physically reasonable")
'''

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_internal_imports(source: str, first_file: bool) -> str:
    """Remove `from src.X import Y` lines; keep only one `from __future__` block."""
    source = _INTERNAL_IMPORT_RE.sub("", source)
    if not first_file:
        source = _FUTURE_IMPORT_RE.sub("", source)
    return source


def _strip_module_docstring_and_future(source: str) -> str:
    """
    Remove the top-level module docstring (already used as section header)
    and duplicate __future__ imports from subsequent files.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return source
    first_stmt = tree.body[0] if tree.body else None
    if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant):
        lines = source.splitlines(keepends=True)
        # Skip lines up to and including the closing triple-quote
        end_line = first_stmt.end_lineno
        source = "".join(lines[end_line:])
    return source.lstrip("\n")


def _section_banner(title: str) -> str:
    bar = "=" * 68
    return f"\n# {bar}\n# {title}\n# {bar}\n"


def build_flat_source(repo_root: Path) -> str:
    src_root = repo_root / "src"
    parts: list[str] = [COLAB_SETUP_HEADER]
    first = True

    for rel_path, title in MODULE_ORDER:
        fpath = src_root / rel_path
        if not fpath.exists():
            parts.append(f"\n# [SKIPPED — not found: src/{rel_path}]\n")
            continue

        raw = fpath.read_text(encoding="utf-8")
        raw = _strip_internal_imports(raw, first_file=first)
        raw = _strip_module_docstring_and_future(raw)
        parts.append(_section_banner(title))
        parts.append(raw)
        first = False

    parts.append(_section_banner("MAIN ENTRYPOINT"))
    parts.append(MAIN_ENTRYPOINT)
    return "\n".join(parts)


def build_notebook(flat_source: str) -> dict:
    """
    Wrap the flat source into a minimal .ipynb with one code cell per section.
    """
    sections = re.split(r"(# ={68}\n# .+\n# ={68}\n)", flat_source)
    cells = []

    def make_cell(src: str) -> dict:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src.splitlines(keepends=True),
        }

    # First chunk = setup header (before any banner)
    if sections:
        cells.append(make_cell(sections[0]))

    # Pair banners with their content
    i = 1
    while i < len(sections) - 1:
        banner  = sections[i]
        content = sections[i + 1]
        cells.append(make_cell(banner + content))
        i += 2

    if i < len(sections):
        cells.append(make_cell(sections[i]))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export repo to a flat Colab script or notebook.")
    p.add_argument(
        "--out", default="",
        help="Output file path. Defaults to colab_export.py or colab_export.ipynb.",
    )
    p.add_argument(
        "--mode", choices=["script", "notebook"], default="script",
        help="'script' → single .py file; 'notebook' → .ipynb with cells per section.",
    )
    p.add_argument(
        "--repo", default="",
        help="Path to repo root. Defaults to the directory containing this script.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(args.repo) if args.repo else Path(__file__).parent

    flat = build_flat_source(repo_root)

    if args.mode == "notebook":
        out_path = Path(args.out) if args.out else repo_root / "colab_export.ipynb"
        nb = build_notebook(flat)
        out_path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Notebook written → {out_path}  ({len(nb['cells'])} cells)")
    else:
        out_path = Path(args.out) if args.out else repo_root / "colab_export.py"
        out_path.write_text(flat, encoding="utf-8")
        lines = flat.count("\n")
        print(f"Script  written → {out_path}  ({lines:,} lines)")


if __name__ == "__main__":
    main()
