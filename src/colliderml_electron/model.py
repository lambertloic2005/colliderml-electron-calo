import torch
from torch import nn
import torch.nn.functional as F

from colliderml_electron.embedding import FourierPositionalEncoding
from colliderml_electron.encoder import CellEncoder


class ConcatCaloRegressor(nn.Module):
    """
    First runnable baseline:

        cells
        -> select top max_cells cells by energy
        -> FourierPositionalEncoding
        -> CellEncoder
        -> concatenate fixed number of per-cell latent vectors
        -> MLP regression head
    """

    def __init__(
    self,
    max_cells: int = 256,
    model_dim: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    output_dim: int = 6,
    high_level_dim: int = 7,
    ):
        super().__init__()

        self.max_cells = max_cells
        self.model_dim = model_dim

        self.fourier_embed = FourierPositionalEncoding(
            input_dim=3,
            high_level_dim=high_level_dim,
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

    def _select_top_cells(
        self,
        x_sampled: torch.Tensor,
        x_high_level: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Select at most max_cells cells before the transformer.

        x_sampled: (B, L, 3)
        x_high_level: (B, L, 7)
        mask: (B, L), True = padding, False = real cell

        We use x_high_level[..., 0] as the energy-like score.
        """

        B, L, _ = x_sampled.shape

        if L <= self.max_cells:
            return x_sampled, x_high_level, mask

        scores = x_high_level[..., 0]

        # Never choose padding cells unless an event has fewer than max_cells
        # real cells, in which case some padding is unavoidable.
        scores = scores.masked_fill(mask, float("-inf"))

        _, idx = torch.topk(
            scores,
            k=self.max_cells,
            dim=1,
            largest=True,
            sorted=True,
        )

        x_sampled = x_sampled.gather(
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, x_sampled.shape[-1]),
        )

        x_high_level = x_high_level.gather(
            dim=1,
            index=idx.unsqueeze(-1).expand(-1, -1, x_high_level.shape[-1]),
        )

        mask = mask.gather(dim=1, index=idx)

        return x_sampled, x_high_level, mask

    def _force_fixed_length(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: (B, L, D)

        The MLP head needs exactly max_cells vectors.
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
        """
        x_sampled: (B, L, 3)
        x_high_level: (B, L, 7)
        mask: (B, L), True = padding, False = real cell
        """

        # Important: reduce sequence length before transformer attention.
        x_sampled, x_high_level, mask = self._select_top_cells(
            x_sampled,
            x_high_level,
            mask,
        )

        h, mask = self.encoder(x_sampled, x_high_level, mask)

        # Remove padding vectors before concatenation.
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)

        h = self._force_fixed_length(h)

        z = h.flatten(start_dim=1)

        pred = self.head(z)

        return pred
    
    # ============================================================================
# Add to src/colliderml_electron/model.py, AFTER the ConcatCaloRegressor class.
# It inherits _select_top_cells from ConcatCaloRegressor and only changes how the
# (B, L, D) encoder output is aggregated into a fixed vector: a small Conv1d stack
# along the (energy-sorted) sequence, then a mask-aware mean+max pool, then a head.
# ============================================================================


class ConvCaloRegressor(ConcatCaloRegressor):
    """
    Same front end as ConcatCaloRegressor (top-cell selection + Fourier embed +
    transformer encoder), but the aggregation is:

        encoder output (B, L, D)
        -> Conv1d stack along the sequence  (channels = D)
        -> masked mean+max pool over real cells
        -> MLP head

    Note on ordering: _select_top_cells uses topk(sorted=True) on the energy
    score, so position 0 is the highest-energy cell. The conv therefore mixes
    energy-adjacent cells. That is the structure available here; sorting cells by
    a physical axis (depth/radius) before the conv is a separate future lever.
    """

    def __init__(
        self,
        max_cells: int = 256,
        model_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_dim: int = 6,
        high_level_dim: int = 7,
        conv_dim: int = 128,
        kernel_size: int = 5,
    ):
        # Build embed + encoder via the parent, then throw away its concat head.
        super().__init__(
            max_cells=max_cells,
            model_dim=model_dim,
            n_heads=n_heads,
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_dim=output_dim,
            high_level_dim=high_level_dim,
        )

        pad = kernel_size // 2  # 'same' length
        self.conv = nn.Sequential(
            nn.Conv1d(model_dim, conv_dim, kernel_size=kernel_size, padding=pad),
            nn.GELU(),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # mean + max pooled -> 2 * conv_dim
        self.head = nn.Sequential(
            nn.LayerNorm(2 * conv_dim),
            nn.Linear(2 * conv_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def _force_fixed_length_with_mask(self, h, mask):
        """Pad/truncate BOTH h (B,L,D) and mask (B,L) to exactly max_cells.
        Appended positions are marked True (= padding)."""
        B, L, D = h.shape
        if L > self.max_cells:
            return h[:, : self.max_cells, :], mask[:, : self.max_cells]
        if L < self.max_cells:
            pad_len = self.max_cells - L
            h = F.pad(h, pad=(0, 0, 0, pad_len), value=0.0)
            mask = F.pad(mask, pad=(0, pad_len), value=True)  # True = padding
        return h, mask

    @staticmethod
    def _masked_mean_max(h_conv, mask):
        """h_conv: (B, C, L)   mask: (B, L) True=padding.  Returns (B, 2C)."""
        valid = (~mask).unsqueeze(1).to(h_conv.dtype)        # (B,1,L)
        summed = (h_conv * valid).sum(dim=2)                 # (B,C)
        count = valid.sum(dim=2).clamp_min(1.0)              # (B,1)
        mean = summed / count
        neg_inf = torch.finfo(h_conv.dtype).min
        masked = h_conv.masked_fill(mask.unsqueeze(1), neg_inf)
        mx = masked.max(dim=2).values                        # (B,C)
        return torch.cat([mean, mx], dim=1)                  # (B,2C)

    def forward(self, x_sampled, x_high_level, mask):
        x_sampled, x_high_level, mask = self._select_top_cells(
            x_sampled, x_high_level, mask
        )

        h, mask = self.encoder(x_sampled, x_high_level, mask)   # (B, L, D)
        h, mask = self._force_fixed_length_with_mask(h, mask)   # (B, max_cells, D)

        h = h.masked_fill(mask.unsqueeze(-1), 0.0)              # zero padding pre-conv
        h = h.transpose(1, 2)                                  # (B, D, L) for Conv1d
        h = self.conv(h)                                       # (B, conv_dim, L)

        z = self._masked_mean_max(h, mask)                     # (B, 2*conv_dim)
        return self.head(z)