import torch
import torch.nn as nn
import torch.nn.functional as F

def gaussian_window(window_size, sigma, channels, dtype, device):
    coords = torch.arange(window_size, dtype=dtype, device=device) - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = (g / g.sum()).unsqueeze(0)
    w2d = (g.t() @ g)
    w2d = w2d / w2d.sum()
    window = w2d.expand(channels, 1, window_size, window_size).contiguous()
    return window

class SSIMLoss(nn.Module):
    def __init__(self, rgb_range=1, channel=None, window_size=11, sigma=1.5, reduction='mean'):
        super().__init__()
        self.rgb_range = float(rgb_range)
        self.window_size = int(window_size)
        self.sigma = float(sigma)
        self.reduction = reduction
        self.K1, self.K2 = 0.01, 0.03
        # 动态窗口缓存
        self.register_buffer('window', torch.tensor([]))
        self._window_channels = 0

    def _get_window(self, C, dtype, device):
        if (self.window.numel() == 0
            or self.window.dtype != dtype
            or self.window.device != device
            or self._window_channels != C):
            self.window = gaussian_window(self.window_size, self.sigma, C, dtype, device)
            self._window_channels = C
        return self.window

    def forward(self, x, y):
        assert x.shape == y.shape, f"SSIM input shapes must match, got {x.shape} vs {y.shape}"
        C = x.size(1)
        window = self._get_window(C, x.dtype, x.device)
        padding = self.window_size // 2

        mu_x = F.conv2d(x, window, padding=padding, groups=C)
        mu_y = F.conv2d(y, window, padding=padding, groups=C)

        mu_x2, mu_y2 = mu_x.pow(2), mu_y.pow(2)
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=C) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=C) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=padding, groups=C) - mu_xy

        L = self.rgb_range
        C1 = (self.K1 * L) ** 2
        C2 = (self.K2 * L) ** 2

        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-12)
        loss = 1.0 - ssim_map

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss