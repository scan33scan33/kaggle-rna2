"""
Training and inference loop for AlphaFold3InspiredRNA.

Losses:
  1. Coordinate loss   — smooth L1 on normalised C1' coords
  2. Distogram loss    — smooth L1 on pairwise inter-residue distances
  3. Bond length loss  — smooth L1 on consecutive C1'-C1' distances (~5.9 Å)

Inference:
  - MC dropout for diverse structure sampling
  - Template-Based Modelling (TBM) fallback via k-mer similarity
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.models.pairformer import AlphaFold3InspiredRNA, tokenise


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

    P_c = P - P.mean(dim=0)
    Q_c = Q - Q.mean(dim=0)

    H = P_c.T @ Q_c
    U, S, Vh = torch.linalg.svd(H)   # torch.svd is deprecated; linalg.svd returns Vh = V.T
    V = Vh.mT
    d = torch.sign(torch.det(V @ U.T))
    D = torch.diag(torch.tensor([1.0, 1.0, d], device=P.device, dtype=P.dtype))
    R = V @ D @ U.T

    P_aligned = (P_c @ R.T) + Q.mean(dim=0)
    dist_sq = ((P_aligned - Q) ** 2).sum(dim=1)

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

class RibonanzaFeatureExtractor:
    """
    Loads RibonanzaNet2 from its Kaggle model checkpoint and extracts
    (1D, 2D) representations for RNA sequences.
    Falls back gracefully if weights are unavailable.
    """

    def __init__(self, checkpoint_path: str = ""):
        self.available = False
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if not checkpoint_path or not os.path.exists(checkpoint_path):
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
            self.model = RibonanzaNet(config).to(self.device)
            state = torch.load(weight_files[0], map_location=self.device)
            state = state.get("state_dict", state.get("model_state_dict", state))
            self.model.load_state_dict(state, strict=False)
            self.model.eval()
            self.available = True
            print("RibonanzaNet2 loaded successfully.")
        except Exception as e:
            print(f"RibonanzaNet2 init failed: {e}")

    def _load_config(self, checkpoint_path: str, weight_file: str):
        import yaml, json  # noqa: F811
        for cfg_name in ["config.yaml", "config.yml", "config.json"]:
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
        # Default fallback
        return type("Cfg", (), {
            "ninp": 384, "nhid": 384, "nhead": 12, "nlayers": 12,
            "dropout": 0.1, "k": 5, "ntoken": 6, "dim": 384, "pair_dim": 128,
        })()

    def forward(
        self, sequences: list[str]
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.available or self.model is None:
            return None, None
        mapping = {"A": 1, "C": 2, "G": 3, "U": 4}
        B = len(sequences)
        L = max(len(s) for s in sequences)
        tokens = torch.zeros((B, L), dtype=torch.long, device=self.device)
        for i, s in enumerate(sequences):
            for j, c in enumerate(s):
                tokens[i, j] = mapping.get(c.upper(), 0)
        try:
            with torch.no_grad():
                out = self.model(tokens)
            if isinstance(out, tuple) and len(out) >= 2:
                return out[0], out[1]
            if isinstance(out, torch.Tensor) and out.dim() == 3:
                return out, torch.zeros(B, L, L, 32, device=self.device)
        except Exception as e:
            print(f"RibonanzaNet2 forward failed: {e}")
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
    # Zero-centre at first valid residue
    coords -= coords[valid[0]]
    return coords


def _autocast_ctx():
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.float16)
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
            target  = torch.tensor([coords], dtype=torch.float32, device=device)

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

                    loss = loss_coord + 0.02 * loss_dist + 0.05 * loss_bond

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
            target  = torch.tensor([coords], dtype=torch.float32, device=device)
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
                pv = (pred * coord_std_t + coord_mean_t).squeeze(0)[row_valid]
                tv = target.squeeze(0)[row_valid]
                rmsd, tm = kabsch_rmsd_tmscore(pv, tv)
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
