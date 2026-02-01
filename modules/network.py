import torch
import torch.nn as nn
from torch.distributions import Independent, OneHotCategoricalStraightThrough
from torch.distributions.utils import probs_to_logits

######################## Encoder #########################

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 4, 2, 1),    # (3, 128, 256) -> (16, 64, 128)
            nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1),   # (16, 64, 128) -> (32, 32, 64)
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),   # (32, 32, 64)  -> (64, 16, 32)
            nn.SiLU(),
            nn.Conv2d(64, 128, 4, 2, 1),  # (64, 16, 32)  -> (128, 8, 16)
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1), # (128, 8, 16)  -> (256, 4, 8)
            nn.SiLU(),
            nn.Conv2d(256, 512, 4, 2, 1), # (256, 4, 8)   -> (512, 2, 4)
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(512 * 2 * 4, self.config.encoded_state_size),
        )

    def forward(self, x):
        """
        x shape      : (B, T, 3, 128, 256)        or (B, 3, 128, 256)
        output shape : (B, T, encoded_state_size) or (B, encoded_state_size)
        """
        if x.ndim == 5:
            B, T, C, H, W = x.shape
            x = x.view(B * T, C, H, W)
            x = self.network(x)
            x = x.view(B, T, -1)
        else:
            x = self.network(x)
        return x

######################## Decoder #########################

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.full_state_size, 512 * 2 * 4),
            nn.Unflatten(1, (512, 2, 4)),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), # (512, 2, 4) -> (256, 4, 8)
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # (256, 4, 8) -> (128, 8, 16)
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # (128, 8, 16) -> (64, 16, 32)
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # (64, 16, 32) -> (32, 32, 64)
            nn.SiLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),   # (32, 32, 64) -> (16, 64, 128)
            nn.SiLU(),
            nn.ConvTranspose2d(16, 3, 4, 2, 1),    # (16, 64, 128) -> (3, 128, 256)
        )

    def forward(self, x):
        """
        x shape      : (B, full_state_size) or (B, T, full_state_size)
        output shape : (B, 3, 128, 256)     or (B, T, 3, 128, 256)
        """
        if x.ndim == 3:
            B, T, C = x.shape
            x = x.view(B * T, C)
            x = self.network(x)
            x = x.view(B, T, *self.config.observation_shape)
        else:
            x = self.network(x)
        return x
    
######################## RecurrentModel #########################

"""
Custom GRUCell
Reference: https://github.com/danijar/dreamerv3
"""

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, update_bias=-1):
        super().__init__()
        self.update_bias = update_bias
        self.linear = nn.Linear(input_size + hidden_size, 3 * hidden_size)

    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        parts = self.linear(combined)

        reset, candidate, update = torch.chunk(parts, 3, dim=-1)

        reset     = torch.sigmoid(reset)
        candidate = torch.tanh(reset * candidate)
        update    = torch.sigmoid(update + self.update_bias)

        out = update * candidate + (1 - update) * h
        return out

class RecurrentModel(nn.Module):
    def __init__(self, config, hidden_size=512):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.latent_size + self.config.action_size, hidden_size),
            nn.SiLU(),
        )
        self.recurrent = GRUCell(hidden_size, self.config.recurrent_size)

    def forward(self, recurrent_state, latent_state, action):
        x = torch.cat((latent_state, action), -1)
        x = self.network(x)
        x = self.recurrent(x, recurrent_state)
        return x
    
######################## PriorNetwork #########################

class PriorNetwork(nn.Module):
    def __init__(self, config, hidden_size=512):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.recurrent_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.config.latent_size),
        )

    def forward(self, recurrent_state):
        raw_logits = self.network(recurrent_state)

        probabilities = raw_logits.view(-1, self.config.latent_length, self.config.latent_classes).softmax(-1)
        uniform = torch.ones_like(probabilities) / self.config.latent_classes
        final_probabilities = (1 - self.config.uniform_mix) * probabilities + self.config.uniform_mix * uniform
        logits = probs_to_logits(final_probabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.config.latent_size), logits
    
######################## PosteriorNetwork #########################

class PosteriorNetwork(nn.Module):
    def __init__(self, config, hidden_size=512):
        super().__init__()
        self.config = config

        self.network = nn.Sequential(
            nn.Linear(self.config.recurrent_size + self.config.encoded_state_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.config.latent_size),
        )

    def forward(self, recurrent_state, encoded_state):
        x = torch.cat((recurrent_state, encoded_state), -1)
        raw_logits = self.network(x)

        probabilities = raw_logits.view(-1, self.config.latent_length, self.config.latent_classes).softmax(-1)
        uniform = torch.ones_like(probabilities) / self.config.latent_classes
        final_probabilities = (1 - self.config.uniform_mix) * probabilities + self.config.uniform_mix * uniform
        logits = probs_to_logits(final_probabilities)

        sample = Independent(OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
        return sample.view(-1, self.config.latent_size), logits