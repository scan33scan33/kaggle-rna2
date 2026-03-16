"""
AlphaFold3-Inspired Pairformer model for RNA 3D structure prediction.

Architecture:
  - Token embedding + absolute positional embedding for single (1D) representation
  - Relative positional embedding for pair (2D) representation
  - N x PairformerBlock:
      * Biased multi-head attention (pair rep biases single attention)
      * Feed-forward transition on single rep
      * Outer-product update of pair rep from single rep
  - Coordinate head: LayerNorm → Linear → GELU → Linear → 3 (C1' xyz)

Optional inputs:
  - ribo_1d_feats: RibonanzaNet2 1D sequence embeddings (B, L, d_ribo_1d)
  - ribo_2d_feats: RibonanzaNet2 2D pair embeddings    (B, L, L, d_ribo_2d)
"""

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
    ):
        super().__init__()
        self.d_single = d_single

        # Input embeddings
        self.embedding = nn.Embedding(5, d_single)        # 4 bases + unknown
        self.abs_pos_emb = nn.Embedding(max_len, d_single)
        self.rel_pos_emb = nn.Embedding(65, d_pair)       # [-32, +32] clipped

        # RibonanzaNet2 integration projections — LazyLinear infers input dim on
        # first forward so we don't need to hard-code the checkpoint's hidden size.
        self.ribo_proj_1d = nn.LazyLinear(d_single)
        self.ribo_proj_2d = nn.LazyLinear(d_pair)

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
