import torch
import torch.nn as nn

class FusionMLP(nn.Module):
    """
    Fusion module that combines image and text embeddings:
    phi = LayerNorm(MLP([Wv*z; Wt*t]))
    """

    def __init__(self, dv: int, dt: int, d_out: int = 512, hidden: int = 1024):
        super().__init__()
        self.Wv = nn.Linear(dv, d_out)
        self.Wt = nn.Linear(dt, d_out)
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_out, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
            nn.LayerNorm(d_out)
        )

    def forward(self, z_v: torch.Tensor, tau_t: torch.Tensor) -> torch.Tensor:
        p_v = self.Wv(z_v)
        p_t = self.Wt(tau_t)
        fused = torch.cat([p_v, p_t], dim=1)
        return self.mlp(fused)