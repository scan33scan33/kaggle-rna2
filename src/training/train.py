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

import math
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


def _collate_batch(
    batch: list[tuple[str, np.ndarray]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Pad a list of (seq_str, coords_np) into tensors for a single forward pass."""
    seq_strs = [s[0] for s in batch]
    L_max = max(len(s) for s in seq_strs)
    B = len(batch)
    seq_tokens = torch.zeros((B, L_max), dtype=torch.long, device=device)
    targets = torch.full((B, L_max, 3), float("nan"), dtype=torch.float32, device=device)
    for i, (seq_str, coords) in enumerate(batch):
        toks = tokenise(seq_str)
        seq_tokens[i, : len(toks)] = torch.tensor(toks, dtype=torch.long, device=device)
        targets[i, : coords.shape[0]] = torch.from_numpy(coords).to(device)
    return seq_tokens, targets, seq_strs


def train(
    model: AlphaFold3InspiredRNA,
    train_seq_df: pd.DataFrame,
    train_labels_df: pd.DataFrame,
    val_seq_df: pd.DataFrame | None = None,
    val_labels_df: pd.DataFrame | None = None,
    extractor: RibonanzaFeatureExtractor | None = None,
    epochs: int = 50,
    lr: float = 1e-4,
    lr_min: float = 1e-6,
    weight_decay: float = 0.01,
    batch_size: int = 1,
    grad_clip: float = 1.0,
    dist_loss_weight: float = 0.2,
    bond_loss_weight: float = 0.1,
    max_seq_len: int = 2000,
    log_every: int = 10,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Train the model and return (coord_mean, coord_std) for denormalisation.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr_min)
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

    # Pre-collect all valid (seq_str, coords) pairs once — avoids re-filtering each epoch
    all_samples: list[tuple[str, np.ndarray]] = []
    skipped_init = 0
    for _, row in train_seq_df.iterrows():
        seq_str = row["sequence"]
        tid = row["target_id"]
        if len(seq_str) > max_seq_len or tid not in labels_grouped.groups:
            skipped_init += 1
            continue
        coords = _build_coords_array(labels_grouped.get_group(tid), len(seq_str))
        if coords is None:
            skipped_init += 1
            continue
        all_samples.append((seq_str, coords))
    print(f"Training on {len(all_samples)} sequences ({skipped_init} skipped), batch_size={batch_size}")

    import random

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = skipped_batches = 0

        # Shuffle and form batches each epoch
        shuffled = all_samples.copy()
        random.shuffle(shuffled)
        batches = [shuffled[i : i + batch_size] for i in range(0, len(shuffled), batch_size)]

        for batch in batches:
            seq_tokens, targets, seq_strs = _collate_batch(batch, device)
            ribo_1d, ribo_2d = (extractor.forward(seq_strs) if extractor else (None, None))

            try:
                with _autocast_ctx():
                    preds = model(seq_tokens, ribo_1d_feats=ribo_1d, ribo_2d_feats=ribo_2d)
                    # preds: (B, L_max, 3)

                    if not torch.isfinite(preds).all():
                        skipped_batches += 1
                        continue

                    valid_mask = ~torch.isnan(targets)
                    if valid_mask.sum() == 0:
                        skipped_batches += 1
                        continue

                    target_norm = torch.clamp((targets - coord_mean_t) / coord_std_t, -10, 10)
                    loss_coord = F.smooth_l1_loss(preds[valid_mask], target_norm[valid_mask])

                    # Per-sample distance and bond loss (variable valid lengths per sample)
                    pred_d = preds * coord_std_t + coord_mean_t
                    loss_dist = loss_bond = torch.tensor(0.0, device=device)
                    n_struct = 0
                    for b in range(len(batch)):
                        valid_rows = torch.isfinite(targets[b]).all(-1)
                        if valid_rows.sum() <= 2:
                            continue
                        pv = pred_d[b][valid_rows]
                        tv = targets[b][valid_rows]
                        loss_dist = loss_dist + F.smooth_l1_loss(
                            torch.cdist(pv.unsqueeze(0), pv.unsqueeze(0)).squeeze(0),
                            torch.cdist(tv.unsqueeze(0), tv.unsqueeze(0)).squeeze(0),
                        )
                        valid_idx = torch.where(valid_rows)[0]
                        is_adj = (valid_idx[1:] - valid_idx[:-1]) == 1
                        if is_adj.any():
                            pb = torch.sqrt(((pv[1:][is_adj] - pv[:-1][is_adj]) ** 2).sum(-1) + 1e-8)
                            tb = torch.sqrt(((tv[1:][is_adj] - tv[:-1][is_adj]) ** 2).sum(-1) + 1e-8)
                            loss_bond = loss_bond + F.smooth_l1_loss(pb, tb)
                        n_struct += 1
                    if n_struct > 0:
                        loss_dist = loss_dist / n_struct
                        loss_bond = loss_bond / n_struct

                    loss = loss_coord + dist_loss_weight * loss_dist + bond_loss_weight * loss_bond

                if not torch.isfinite(loss):
                    optimizer.zero_grad()
                    skipped_batches += 1
                    continue

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                total_loss += loss.item()
                batch_count += 1

                if batch_count % log_every == 0:
                    print(f"  Epoch {epoch+1} | batch {batch_count} | loss {total_loss/batch_count:.4f}")

                if torch.cuda.is_available() and torch.cuda.memory_reserved() > 14.5 * 2**30:
                    torch.cuda.empty_cache()

            except RuntimeError as e:
                print(f"  RuntimeError (batch L_max={seq_tokens.shape[1]}): {e}")
                optimizer.zero_grad()
                skipped_batches += 1

        print(
            f"Epoch {epoch+1}/{epochs} | avg_loss={total_loss/max(batch_count,1):.4f} | "
            f"batches={batch_count} | skipped={skipped_batches} | lr={scheduler.get_last_lr()[0]:.2e}"
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
