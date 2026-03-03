import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ResBlock, SelfAttention

######################## VQ-VAE-Encoder #########################

class VQVAEEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            # (3, 128, 256) -> (64, 64, 128)
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64),
            ResBlock(64),

            # (64, 64, 128) -> (128, 32, 64)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            ResBlock(128),
            ResBlock(128),

            # (128, 32, 64) -> (256, 16, 32)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            ResBlock(256),
            ResBlock(256),
            SelfAttention(256),
            ResBlock(256),

            # (256, 16, 32) -> (D, 16, 32)
            nn.GroupNorm(8, 256),
            nn.SiLU(),
            nn.Conv2d(256, config.vq_code_dim, 1)
        )

    def forward(self, x):
        return self.network(x)  # (B, D, 16, 32)

######################## VQ-VAE-Decoder #########################

class VQVAEDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            # (D, 16, 32) -> (256, 16, 32) 
            nn.Conv2d(config.vq_code_dim, 256, 3, 1, 1),
            ResBlock(256),
            ResBlock(256),
            SelfAttention(256),
            ResBlock(256),

            # (256, 16, 32) -> (128, 32, 64) 
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            ResBlock(128),
            ResBlock(128),

            # (128, 32, 64) -> (64, 64, 128)
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            ResBlock(64),
            ResBlock(64),

            # (64, 64, 128) -> (3, 128, 256) 
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        return self.network(z)
    
######################## VQ-EMA ########################

class VectorQuantizerEMA(nn.Module):
    def __init__(self, config, eps=1e-5):
        super().__init__()
        self.config = config

        self.K = config.vq_codebook_size
        self.D = config.vq_code_dim
        self.beta = config.vq_commitment_cost
        self.decay = config.vq_ema_decay
        self.eps = eps

        self.register_buffer('embedding',    torch.randn(self.K, self.D))
        self.register_buffer('cluster_size', torch.zeros(self.K))
        self.register_buffer('embed_avg',    torch.randn(self.K, self.D))
        self.register_buffer('_initialized', torch.tensor(False))

    @torch.no_grad()
    def _init_from_data(self, flat):
        index = torch.randperm(flat.size(0), device=flat.device)[:self.K]
        self.embedding.copy_(flat[index])
        self.embed_avg.copy_(flat[index])
        self.cluster_size.fill_(1.0)
        self._initialized.fill_(True)

    def forward(self, z_e):
        B, D, H, W = z_e.shape

        flat = z_e.detach().permute(0, 2, 3, 1).reshape(-1, D).float() # (B, D, H, W) -> (B*H*W, D)

        if self.training and not self._initialized:
            self._init_from_data(flat)

        distances = (
            flat.pow(2).sum(1, keepdim=True) # (B*H*W, 1)
             + self.embedding.pow(2).sum(1)  # (K,)
             - 2 * flat @ self.embedding.t() # (B*H*W, K)
        )

        indices = distances.argmin(dim=-1) # (B*H*W,)
        z_q_flat = F.embedding(indices, self.embedding) # (B*H*W, D)

        if self.training:
            encodings = F.one_hot(indices, self.K).float() # (B*H*W, K)

            self.cluster_size.mul_(self.decay).add_(
                encodings.sum(0), 
                alpha=1 - self.decay
            )

            self.embed_avg.mul_(self.decay).add_(
                encodings.t() @ flat, 
                alpha=1 - self.decay
            )

            n = self.cluster_size.sum()
            smoothed = (self.cluster_size + self.eps) / (n + self.K * self.eps) * n
            self.embedding.copy_(self.embed_avg / smoothed.unsqueeze(1))

        z_q = z_q_flat.reshape(B, H, W, D).permute(0, 3, 1, 2) # (B*H*W, D) -> (B, D, H, W)

        # Commitment loss
        commitment_loss = self.beta * F.mse_loss(z_e.float(), z_q.detach())

        # Straight through 
        z_q_st = z_e + (z_q.to(z_e.dtype) - z_e).detach()

        return z_q_st, indices.reshape(B, H, W), commitment_loss

    def get_codebook_entry(self, indices):
        return F.embedding(indices.long(), self.embedding)

    @property
    def usage(self):
        return (self.cluster_size > 1.0).sum().item()