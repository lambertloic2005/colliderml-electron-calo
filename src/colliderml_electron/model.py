import torch
from torch import nn
import torch.nn.functional as F

from colliderml_electron.embedding import FourierPositionalEncoding
from colliderml_electron.encoder import CellEncoder


class ConcatCaloRegressor(nn.Module):

    def __init__(
        self,
        max_cells: int = 256,
        model_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 6,
    ):
        super().__init__()

        self.max_cells = max_cells
        self.model_dim = model_dim

        # Your dataset gives:
        # x_sampled: (B, L, 3) -> x, y, z
        # x_high_level: (B, L, 7) -> log energy + 6 detector one-hot values
        self.fourier_embed = FourierPositionalEncoding(
            input_dim=3,
            high_level_dim=7,
            num_frequencies=[6, 6, 6],
            dim_max=[1100.0, 1100.0, 3000.0],
        )

        self.encoder = CellEncoder(
            fourier_embed=self.fourier_embed,
            fourier_out_dim=self.fourier_embed.output_dim,
            model_dim=model_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        concat_dim = max_cells * model_dim

        self.head = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def _force_fixed_length(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, L, D)

        The MLP head needs a fixed input size, so we force every shower to have
        exactly max_cells latent vectors.

        If L > max_cells: keep first max_cells cells.
        If L < max_cells: pad with zero latent vectors.
        """
        B, L, D = h.shape

        if L > self.max_cells:
            h = h[:, : self.max_cells, :]
        elif L < self.max_cells:
            pad_len = self.max_cells - L
            h = F.pad(h, pad=(0, 0, 0, pad_len), value=0.0)

        return h

    def forward(
        self,
        x_sampled: torch.Tensor,
        x_high_level: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:

        h, mask = self.encoder(x_sampled, x_high_level, mask)

        # Very important: remove padding vectors before concatenation.
        # The transformer ignores padding as keys/values, but padded output rows
        # can still contain nonzero vectors, so we zero them manually.
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)

        h = self._force_fixed_length(h)

        # Concatenate all fixed-length per-cell vectors into one shower vector.
        z = h.flatten(start_dim=1)

        pred = self.head(z)
        return pred