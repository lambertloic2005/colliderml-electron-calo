# src/colliderml_electron/regressor.py
from __future__ import annotations
import torch
from torch import nn
from .encoder import CellEncoder


class Regressor(nn.Module):
    def __init__(self, model_dim: int = 128, hidden: int = 128, n_outputs: int = 3):  # was 2
        super().__init__()
        self.pool_norm = nn.LayerNorm(model_dim)        # keep if you used the LayerNorm fix
        self.head = nn.Sequential(
            nn.Linear(model_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_outputs),               # now 3: [eta, phi_cos, phi_sin]
        )
    # combine() and forward() unchanged
    def combine(self, h, mask):
        h = h.masked_fill(mask.unsqueeze(-1), 0.0)
        return h.sum(dim=1)

    def forward(self, h, mask):
        pooled = self.pool_norm(self.combine(h, mask))   # normalise before the head
        return self.head(pooled)


class EtaPhiModel(nn.Module):
    def __init__(self, encoder: CellEncoder, regressor: Regressor):
        super().__init__()
        self.encoder = encoder
        self.regressor = regressor

    def forward(self, x_sampled, x_high_level, mask):
        h, mask = self.encoder(x_sampled, x_high_level, mask)  # (B, L, D)
        return self.regressor(h, mask)  


def eta_phi_geometric_loss(pred, eta_true, phi_true, eps=1e-6):
    """pred: (B, 3) = [eta, phi_cos, phi_sin]."""
    eta_pred = pred[:, 0]
    phi_cos, phi_sin = pred[:, 1], pred[:, 2]

    eta_loss = (eta_pred - eta_true).pow(2).mean()                 # not periodic -> MSE

    cos_t, sin_t = torch.cos(phi_true), torch.sin(phi_true)        # unit-circle coords
    phi_loss = ((phi_cos - cos_t).pow(2) + (phi_sin - sin_t).pow(2)).mean()

    loss = torch.log(eta_loss + eps) + torch.log(phi_loss + eps)
    geo = torch.sqrt((eta_loss + eps) * (phi_loss + eps)).item()
    return loss, {"eta_loss": eta_loss.item(), "phi_loss": phi_loss.item(), "geo_mean": geo}

def phi_only_loss(pred, phi_true):
    # pred row is [eta, phi_cos, phi_sin]; ignore eta entirely here
    cos_t, sin_t = torch.cos(phi_true), torch.sin(phi_true)
    phi_loss = ((pred[:, 1] - cos_t).pow(2) + (pred[:, 2] - sin_t).pow(2)).mean()
    return phi_loss, {"phi_loss": phi_loss.item()}