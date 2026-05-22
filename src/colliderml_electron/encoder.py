
import torch
from torch import nn


class CellEncoder(nn.Module):
    def __init__(
        self,
        fourier_embed: nn.Module,   # FourierPositionalEncoding (no learnable params)
        fourier_out_dim: int,       # width coming out of fourier_embed
        model_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.fourier_embed = fourier_embed

        # Adapter: Fourier output width -> transformer working width.
        self.in_proj = nn.Linear(fourier_out_dim, model_dim)

        # Mask-aware transformer encoder stack.
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,   # expect (B, L, D), matching collate_pad output
            norm_first=True,    # pre-norm: more stable to train deep stacks
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x_sampled, x_high_level, mask):
        # 1. Fourier embed: positions get multi-scale sin/cos; content passes through.
        emb = self.fourier_embed(x_sampled, x_high_level)   # (B, L, fourier_out_dim)

        # 2. Linear adapter to model_dim.
        h = self.in_proj(emb)                                # (B, L, D)

        # 3. N mask-aware encoder layers.
        #    src_key_padding_mask convention is True = ignore, matching `mask`.
        h = self.encoder(h, src_key_padding_mask=mask)       # (B, L, D)

        return h, mask