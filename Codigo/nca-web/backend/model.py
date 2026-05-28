import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceptionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        lap     = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],   dtype=torch.float32)
        for name, k in [('sx', sobel_x), ('sy', sobel_y), ('lap', lap)]:
            self.register_buffer(name, k.view(1, 1, 3, 3).repeat(channels, 1, 1, 1))

    def forward(self, x):
        gx  = F.conv2d(x, self.sx,  padding=1, groups=self.channels)
        gy  = F.conv2d(x, self.sy,  padding=1, groups=self.channels)
        lap = F.conv2d(x, self.lap, padding=1, groups=self.channels)
        return torch.cat([x, gx, gy, lap], dim=1)


class UpdateNetwork(nn.Module):
    def __init__(self, channels, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 4, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, hidden // 2, 1),
            nn.GELU(),
            nn.Conv2d(hidden // 2, channels, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class NCA(nn.Module):
    def __init__(self, channels=96, update_rate=0.5):
        super().__init__()
        self.channels    = channels
        self.update_rate = update_rate
        self.perceive    = PerceptionBlock(channels)
        self.update      = UpdateNetwork(channels)

    def forward(self, x):
        perceived = self.perceive(x)
        dx        = self.update(perceived)
        alive     = (torch.rand(x.shape[0], 1, x.shape[2], x.shape[3],
                                device=x.device) < self.update_rate).float()
        return x + dx * alive

    @staticmethod
    def init_state(img, mask, channels=96):
        B, _, H, W = img.shape
        state         = torch.zeros(B, channels, H, W, device=img.device)
        state[:, :3]  = img
        state[:, 3:4] = mask
        state[:, 4:7] = img * mask
        return state
