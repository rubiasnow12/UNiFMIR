import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, Callable
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

# 假设模型路径下有这些辅助模块
from model.enlcn import ENLCN 

##########################################################################
# 1. DINO v3 基础组件 (来自你的原始代码)
##########################################################################

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x):
        return x * self.gamma

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0, bias=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x

class RopePositionEmbedding(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, base: float = 100.0, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.base = base
        self.dtype = dtype
        self.inv_freq = 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, H: int, W: int, device):
        seq_len = H * W
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t_h = torch.arange(H, device=device).float() / H
            t_w = torch.arange(W, device=device).float() / W
            grid_h, grid_w = torch.meshgrid(t_h, t_w, indexing='ij')
            grid = torch.stack([grid_h, grid_w], dim=-1).reshape(-1, 2)
            
            freqs = grid[:, :, None] * self.inv_freq.to(device)[None, None, :]
            freqs = freqs.reshape(seq_len, -1)
            self._cos_cached = freqs.cos().view(1, 1, seq_len, -1).to(self.dtype)
            self._sin_cached = freqs.sin().view(1, 1, seq_len, -1).to(self.dtype)
        return self._cos_cached, self._sin_cached

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def _apply_rope(self, x, cos, sin):
        x1, x2 = x.chunk(2, dim=-1)
        rotated = torch.cat([-x2, x1], dim=-1)
        return (x * cos) + (rotated * sin)

    def forward(self, x, cos=None, sin=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if cos is not None:
            q = self._apply_rope(q, cos, sin)
            k = self._apply_rope(k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_ratio=4.0, drop_path=0.0, norm_layer=nn.LayerNorm, init_values=1e-4):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.ffn = Mlp(in_features=dim, hidden_features=int(dim * ffn_ratio))
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, cos=None, sin=None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), cos, sin)))
        x = x + self.drop_path2(self.ls2(self.ffn(self.norm2(x))))
        return x

##########################################################################
# 2. 辅助组件 (PatchEmbed, Upsample, UNetA)
##########################################################################

class PatchEmbed(nn.Module):
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=768):
        super().__init__()
        self.patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)

class PatchUnEmbed(nn.Module):
    def __init__(self, embed_dim=768):
        super().__init__()
        self.embed_dim = embed_dim
    def forward(self, x, grid_size):
        B, HW, C = x.shape
        return x.transpose(1, 2).reshape(B, C, grid_size[0], grid_size[1])

class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if scale == 1:
            m.append(nn.Identity())
        elif (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        super(Upsample, self).__init__(*m)

# 此处保留你代码中的 UNetA 用于任务 5
class UNetA(nn.Module):
    def __init__(self, num_in_ch, n_slices):
        super(UNetA, self).__init__()
        self.conv = nn.Conv2d(num_in_ch, n_slices, 3, 1, 1) # 简化占位，实际请使用你原有的 UNetA 实现
    def forward(self, x): return torch.tanh(self.conv(x))

##########################################################################
# 3. DinoUniModel: 核心多任务模型
##########################################################################

class DinoUniModel(nn.Module):
    def __init__(self, args, embed_dim=768, dino_depth=12, dino_num_heads=12, vit_patch_size=8):
        super(DinoUniModel, self).__init__()
        self.task = 1
        self.img_range = 1.0
        self.mean = torch.zeros(1, 1, 1, 1)
        self.embed_dim = embed_dim
        self.vit_patch_size = vit_patch_size

        # --- 任务特定头 (Heads) ---
        # 1. SR
        self.conv_firstsr = nn.Conv2d(1, embed_dim, 3, 1, 1)
        # 2. Denoise (5通道输入)
        self.conv_firstdT = nn.Conv2d(5, embed_dim, 3, 1, 1)
        # 3. Isotropic
        self.conv_firstiso = nn.Conv2d(1, embed_dim, 3, 1, 1)
        # 4. Projection
        self.project = ENLCN(args=args)
        self.conv_firstproj = nn.Conv2d(1, embed_dim, 3, 1, 1)
        # 5. Volumetric (2D to 3D)
        self.conv_first0 = UNetA(121, 61)
        self.conv_firstv = nn.Conv2d(61, embed_dim, 3, 1, 1)

        # --- 共享主体 (Backbone) ---
        self.patch_embed = PatchEmbed(patch_size=vit_patch_size, in_chans=embed_dim, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(embed_dim=embed_dim, num_heads=dino_num_heads)
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(dim=embed_dim, num_heads=dino_num_heads, drop_path=0.1)
            for _ in range(dino_depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.patch_unembed = PatchUnEmbed(embed_dim=embed_dim)
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)

        # --- 任务特定尾 (Tails) ---
        self.conv_before_upsample0 = nn.Sequential(nn.Conv2d(embed_dim, 32, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsamplesr = Upsample(scale=2, num_feat=32) # Task 1
        self.upsample_common = Upsample(scale=1, num_feat=32) # Task 2, 3
        self.conv_last0 = nn.Conv2d(32, 1, 3, 1, 1)
        
        # Task 5 特有尾部
        self.conv_before_upsamplev = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.conv_lastv = nn.Conv2d(embed_dim, 61, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        B, C, H, W = x.shape
        grid_h, grid_w = H // self.vit_patch_size, W // self.vit_patch_size
        x = self.patch_embed(x)
        cos, sin = self.rope_embed(grid_h, grid_w, x.device)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        x = self.patch_unembed(x, (grid_h, grid_w))
        return self.conv_after_body(x)

    def forward(self, x, tsk=0):
        if tsk > 0: self.task = tsk
        self.mean = self.mean.type_as(x)
        
        # --- Head 阶段 ---
        if self.task == 1: x = self.conv_firstsr((x - self.mean) * self.img_range)
        elif self.task == 2: x = self.conv_firstdT((x - self.mean) * self.img_range)
        elif self.task == 3: x = self.conv_firstiso((x - self.mean) * self.img_range)
        elif self.task == 4:
            x2d, _ = self.project(x)
            x = self.conv_firstproj((x2d - self.mean) * self.img_range)
        elif self.task == 5:
            xunet = self.conv_first0(x)
            x = self.conv_firstv(xunet)

        # --- Body 阶段 ---
        xfe = self.forward_features(x)

        # --- Tail 阶段 ---
        if self.task in [1, 2, 3]:
            x = self.conv_before_upsample0(xfe + x)
            x = self.upsamplesr(x) if self.task == 1 else self.upsample_common(x)
            x = self.conv_last0(x)
        elif self.task == 4:
            x = self.conv_last0(self.conv_before_upsample0(xfe))
            return x2d, x / self.img_range + self.mean + x2d
        elif self.task == 5:
            x = self.conv_lastv(self.conv_before_upsamplev(xfe))
            return xunet, x / self.img_range + self.mean

        return x / self.img_range + self.mean

def make_model(args):
    return DinoUniModel(args)