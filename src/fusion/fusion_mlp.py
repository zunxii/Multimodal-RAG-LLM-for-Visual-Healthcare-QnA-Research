import torch
import torch.nn as nn


class FusionMLP(nn.Module):
    """
    Multimodal fusion module F_m: R^(dv) × R^(dt) → R^d
    
    Implements equation (3):
    φ_i = F_m(W_v · z^(I), W_t · τ_i)
    
    Architecture: Concatenate + MLP with LayerNorm
    φ = LayerNorm(MLP([W_v·z; W_t·τ]))
    """

    def __init__(
        self, 
        dv: int, 
        dt: int, 
        d_out: int = 512, 
        hidden: int = 1024
    ):
        """
        Args:
            dv: Visual embedding dimension
            dt: Text embedding dimension
            d_out: Output fused embedding dimension
            hidden: Hidden layer dimension
        """
        super().__init__()
        
        # Projection layers (W_v and W_t)
        self.proj_v = nn.Linear(dv, d_out)
        self.proj_t = nn.Linear(dt, d_out)
        
        # MLP fusion
        self.mlp = nn.Sequential(
            nn.Linear(2 * d_out, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, d_out),
        )
        
        # Layer normalization
        self.norm = nn.LayerNorm(d_out)
        
        self.d_out = d_out

    def forward(self, z_v: torch.Tensor, tau_t: torch.Tensor) -> torch.Tensor:
        """
        Fuse visual and text embeddings.
        
        Args:
            z_v: Visual embeddings of shape (N, dv)
            tau_t: Text embeddings of shape (N, dt)
            
        Returns:
            Fused embeddings of shape (N, d_out)
        """
        # Project to common dimension
        p_v = self.proj_v(z_v)  # (N, d_out)
        p_t = self.proj_t(tau_t)  # (N, d_out)
        
        # Concatenate and fuse
        concat = torch.cat([p_v, p_t], dim=1)  # (N, 2*d_out)
        fused = self.mlp(concat)  # (N, d_out)
        
        # Layer normalization
        output = self.norm(fused)  # (N, d_out)
        
        return output
