
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
        per_region_proj: bool = False,
        region_eta_boundary: float = 1.5,
    ):
        super().__init__()
        self.fourier_embed = fourier_embed

        # Adapter: Fourier output width -> transformer working width.
        # With per_region_proj=True, cells are routed by their own pseudorapidity
        # (from x_sampled geometry, invariant under the dataset's azimuthal
        # rotation) through in_proj (barrel) or in_proj_endcap.
        self.in_proj = nn.Linear(fourier_out_dim, model_dim)
        self.per_region_proj = per_region_proj
        self.region_eta_boundary = region_eta_boundary
        if per_region_proj:
            self.in_proj_endcap = nn.Linear(fourier_out_dim, model_dim)
            # Identical init: at step 0 the model is exactly equivalent to the
            # shared-projection baseline; any region divergence is learned.
            self.in_proj_endcap.load_state_dict(self.in_proj.state_dict())

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

        # 2. Linear adapter to model_dim (optionally region-routed per cell).
        if self.per_region_proj:
            x, y, z = x_sampled.unbind(-1)                   # (B, L) each, mm
            rho = torch.sqrt(x * x + y * y).clamp_min(1e-6)  # guard padding zeros
            cell_eta = torch.asinh(z / rho)                  # rotation-invariant
            is_endcap = (cell_eta.abs() > self.region_eta_boundary).unsqueeze(-1)
            h = torch.where(is_endcap, self.in_proj_endcap(emb), self.in_proj(emb))
        else:
            h = self.in_proj(emb)                            # (B, L, D)                              # (B, L, D)

        # 3. N mask-aware encoder layers.
        #    src_key_padding_mask convention is True = ignore, matching `mask`.
        h = self.encoder(h, src_key_padding_mask=mask)       # (B, L, D)

        return h, mask