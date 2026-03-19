import torch.nn as nn

#################### DownBlock  ####################

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.down(x)

#################### UpBlock  ####################

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, last=False):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(32, out_channels) if not last else nn.Identity(),
            nn.SiLU()                      if not last else nn.Identity()
        )

    def forward(self, x):
        return self.up(x)

######################## Residual Block #########################

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return x + self.block(x)

# old vision
# class ResBlock(nn.Module):
#     def __init__(self, channels, reduction=4):
#         super().__init__()
#         hidden = channels // reduction
        
#         self.block = nn.Sequential(
#             nn.GroupNorm(8, channels),
#             nn.SiLU(),
#             nn.Conv2d(channels, hidden, 1),     
#             nn.GroupNorm(8, hidden),
#             nn.SiLU(),
#             nn.Conv2d(hidden, hidden, 3, 1, 1), 
#             nn.GroupNorm(8, hidden),
#             nn.SiLU(),
#             nn.Conv2d(hidden, channels, 1),    
#         )

#     def forward(self, x):
#         return x + self.block(x)
