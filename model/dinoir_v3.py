import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import copy
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from model.enlcn import ENLCN
# from typing import Optional, Tuple, Union, Callable
# DINOv3 Dependencies
import logging
from torch import Tensor
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union, Callable
# --- 来自 vision_transformer.py ---
logger = logging.getLogger("dinov3")


# --- 来自 dinov3/layers/rms_norm.py ---
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

    def reset_parameters(self):
        nn.init.ones_(self.weight)

# --- 来自 dinov3/layers/layer_scale.py ---
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-4):
        super().__init__()
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma

    def reset_parameters(self):
        nn.init.ones_(self.gamma)

# --- 来自 dinov3/layers/ffn_layers.py ---
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.SiLU,
        drop=0.0,
        bias=True,
        align_to=1,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        def _align(v, n):
            return int(math.ceil(v / n)) * n

        hidden_features = _align(int(hidden_features * (2 / 3)), align_to)
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        x = self.act(x1) * x2
        x = self.drop(x)
        x = self.w3(x)
        x = self.drop(x)
        return x

# --- 来自 dinov3/layers/rope_position_encoding.py ---
class RopePositionEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        base: float = 100.0,
        min_period: Union[float, None] = None,
        max_period: Union[float, None] = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: Union[float, None] = None,
        jitter_coords: Union[float, None] = None,
        rescale_coords: Union[float, None] = None,
        dtype: torch.dtype = torch.bfloat16,
        device: Union[Any, None] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords
        self.dtype = dtype
        self.device = device
        self._seq_len_cached = -1
        self._cos_cached = None
        self._sin_cached = None
        self._init_weights()

    def _init_weights(self):
        if self.max_period is None:
            power = 2
            self.inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.head_dim, power).float().to(self.device) / self.head_dim)
            )
        else:
            freqs = torch.linspace(
                (self.min_period / (2 * torch.pi)) ** -1,
                (self.max_period / (2 * torch.pi)) ** -1,
                self.head_dim // 2,
            )
            self.inv_freq = freqs.float().to(self.device)

    def _get_course_coords(self, H, W):
        coords_h = torch.arange(H, device=self.device)
        coords_w = torch.arange(W, device=self.device)

        if self.normalize_coords == "min":
            coords_h = coords_h / min(H, W)
            coords_w = coords_w / min(H, W)
        elif self.normalize_coords == "max":
            coords_h = coords_h / max(H, W)
            coords_w = coords_w / max(H, W)
        elif self.normalize_coords == "separate":
            coords_h = coords_h / H
            coords_w = coords_w / W
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")

        if self.shift_coords is not None:
            if self.training and self.jitter_coords is not None:
                shift = self.shift_coords + torch.rand(1) * self.jitter_coords
            else:
                shift = self.shift_coords
            coords_h += shift
            coords_w += shift

        if self.rescale_coords is not None:
            coords_h *= self.rescale_coords
            coords_w *= self.rescale_coords
        return coords_h, coords_w

    def _get_grid_coords(self, H, W):
        coords_h, coords_w = self._get_course_coords(H, W)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"), dim=-1)
        coords = coords.reshape(H * W, 2)
        return coords

    def forward(self, H: int, W: int, device=None):
        # 根据实际序列长度判断是否需要重新计算
        # 优先使用传入的 device，其次使用 self.device
        if device is None:
            current_device = self.device if self.device is not None else torch.device('cpu')
        else:
            current_device = device
        seq_len = H * W
        
        # 缓存命中条件：长度相同 且 设备/dtype 匹配
        if (self._seq_len_cached == seq_len and 
            self._cos_cached is not None and
            self._cos_cached.device == current_device and 
            self._cos_cached.dtype == self.dtype):
            return self._cos_cached, self._sin_cached
        
        # 需要重新计算
        self._seq_len_cached = seq_len
        coords = self._get_grid_coords(H, W).to(current_device)  # 确保 coords 在正确设备上
        with torch.autocast(device_type="cuda", enabled=False):
            # t = coords.float() @ self.inv_freq.float().unsqueeze(-1)
            # freqs = torch.cat((t, t), dim=-1)
            t_coords = coords.float()
            # ensure inv_freq is on the current device to avoid device mismatch
            t_inv_freq = self.inv_freq.to(current_device).float()
            freqs = torch.cat([
                t_coords[:, 0:1] * t_inv_freq,
                t_coords[:, 1:2] * t_inv_freq
            ], dim=-1)
            cos = freqs.cos().to(self.dtype).to(current_device)  # 确保在正确设备上
            sin = freqs.sin().to(self.dtype).to(current_device)  # 确保在正确设备上
            self._cos_cached = cos.unsqueeze(0).unsqueeze(1)
            self._sin_cached = sin.unsqueeze(0).unsqueeze(1)
        return self._cos_cached, self._sin_cached


# ============== 原生 LoRA 实现（不依赖 peft 库） ==============
class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) 线性层
    将原始线性层分解为: W + BA，其中 B 和 A 是低秩矩阵
    """
    def __init__(self, original_linear: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.original_linear = original_linear
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # 获取原始层所在的设备和数据类型
        device = original_linear.weight.device
        dtype = original_linear.weight.dtype
        
        # LoRA 低秩矩阵（确保在同一设备上）
        self.lora_A = nn.Linear(in_features, r, bias=False, device=device, dtype=dtype)
        self.lora_B = nn.Linear(r, out_features, bias=False, device=device, dtype=dtype)
        self.lora_dropout = nn.Dropout(dropout)
        
        # 初始化：A 用 kaiming，B 用零初始化（确保初始时 LoRA 输出为 0）
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # 冻结原始权重
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False
    
    def forward(self, x):
        # 原始输出 + LoRA 输出
        original_out = self.original_linear(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return original_out + lora_out
    
    def merge_weights(self):
        """合并 LoRA 权重到原始线性层（用于推理加速）"""
        with torch.no_grad():
            self.original_linear.weight.data += (self.lora_B.weight @ self.lora_A.weight) * self.scaling
        return self.original_linear
# ============================================================


# --- 来自 dinov3/layers/attention.py (DINOv3 使用的 Attention) ---
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, proj_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def _apply_rope(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        def _rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)
        return (x * rope_cos) + (_rotate_half(x) * rope_sin)

    def forward(self, x, rope_cos: Union[torch.Tensor, None] = None, rope_sin: Union[torch.Tensor, None] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if rope_cos is not None and rope_sin is not None:
            q = self._apply_rope(q, rope_cos, rope_sin)
            k = self._apply_rope(k, rope_cos, rope_sin)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

# --- 来自 dinov3/layers/block.py (DINOv3 使用的 SelfAttentionBlock) ---
class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        ffn_ratio=4.0,
        qkv_bias=False,
        proj_bias=True,
        ffn_bias=True,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        ffn_layer=Mlp,
        init_values=1e-4,
        mask_k_bias: bool = False, # DINOv3 参数, 此处未使用
        device: Union[Any, None] = None, # DINOv3 参数, 此处未使用
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=0.0,
            proj_drop=0.0,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.ffn = ffn_layer(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            act_layer=act_layer,
            bias=ffn_bias,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, rope_cos: Union[torch.Tensor, None] = None, rope_sin: Union[torch.Tensor, None] = None):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), rope_cos, rope_sin)))
        x = x + self.drop_path2(self.ls2(self.ffn(self.norm2(x))))
        return x

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# def make_model(args):
#     try:
#         inch = int(args.inputchannel)
#     except:
#         inch = 1
#     return dinov3(upscale=int(args.scale[0]), in_chans=inch)

def make_model(args):
    """
    创建 DINOv3 模型。
    
    ViT-S 全参微调模式：模型创建时 use_lora=False，所有参数可训练。
    
    # ========== LoRA 配置已禁用，改为全参微调 ==========
    # 重要：模型创建时 use_lora=False，这样可以先加载预训练权重。
    # LoRA 注入应该在 model/__init__.py 中加载权重之后进行。
    # 
    # LoRA 相关参数保存在模型中，供后续 inject_lora() 使用:
    # - args.use_lora: 是否启用 LoRA (保存到模型，加载权重后再注入)
    # - args.lora_r: LoRA 秩 (默认 16)
    # - args.lora_alpha: LoRA alpha (默认 32)
    # - args.lora_dropout: LoRA dropout (默认 0.05)
    # ===================================================
    """
    # LoRA 配置已禁用，保留参数但不使用
    # lora_r = getattr(args, 'lora_r', 16)
    # lora_alpha = getattr(args, 'lora_alpha', 32)
    # lora_dropout = getattr(args, 'lora_dropout', 0.05)
    
    try:
        inch = int(args.inputchannel) if hasattr(args, 'inputchannel') else args.inch
    except:
        inch = 1
    
    # ViT-S 全参微调：创建时 use_lora=False
    model = dinov3(
        upscale=int(args.scale[0]), 
        in_chans=inch,
        out_chans=getattr(args, 'n_colors', 1),
        use_lora=False,  # 全参微调模式，不使用 LoRA
        lora_r=16,       # 保留默认值但不使用
        lora_alpha=32,
        lora_dropout=0.05
    )
    
    # LoRA 注入标志已禁用，始终设为 False
    # model._should_inject_lora = getattr(args, 'use_lora', True)
    model._should_inject_lora = False  # 全参微调模式
    return model


def make_modelproj(args):
    args.n_resblocks = 64
    args.n_feats = 256
    return dinoProj_stage2(upscale=int(args.scale[0]), out_chans=1, args=args)


def make_model2t3(args):
    return dinov3_2dto3d(upscale=11, in_chans=121, out_chans=61)


class dinov3(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3, out_chans=1,
                 embed_dim=384,      # ViT-S 的维度 (原 ViT-B 为 768)
                 depths=[2, 2, 6, 2],
                 num_heads=[6, 6, 6, 6],  # ViT-S 的头数 (原 ViT-B 为 12)
                 vit_patch_size=8,
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None, # qk_scale 也不再使用
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True, # ape 也不再使用
                 use_checkpoint=False, upscale=2, img_range=1., num_feat=32,
                 # === 添加 DINOv3 参数 ===
                 dino_depth=12, dino_num_heads=6, dino_ffn_ratio=4.0,  # ViT-S 的头数
                 layerscale_init=1e-4, dino_norm_layer='layernorm', 
                 dino_ffn_layer='mlp', pos_embed_rope_base=100.0,
                 dino_rope_dtype='bf16',
                 # =======================
                 # === LoRA 微调参数 ===
                 use_lora=False,  # 是否启用 LoRA，load_pretrain.py 设为 False
                 lora_r=16,       # LoRA 秩
                 lora_alpha=32,   # LoRA alpha
                 lora_dropout=0.05
                 # ====================
                 ):
        super(dinov3, self).__init__()
        num_in_ch = in_chans
        num_out_ch = out_chans
        
        # LoRA 相关状态
        self._lora_config = {'r': lora_r, 'alpha': lora_alpha, 'dropout': lora_dropout}
        self._lora_injected = False
        
        self.precision = torch.float32
        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.window_size = window_size
        
        #####################################################################################################
        ################################### 1, shallow feature extraction ###################################
        # conv_first 接收原始输入通道数（1 通道）
        # self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)
        
        #####################################################################################################
        ################################### 2, deep feature extraction ######################################
        # self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape  # absolute position embedding
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        
        # split image into non-overlapping patches 把图像分割成不重叠的补丁
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=vit_patch_size, in_chans=num_in_ch, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        
        # merge non-overlapping patches into image 合并不重叠的补丁到图像
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=vit_patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        
        self.pos_drop = nn.Dropout(p=drop_rate)

        # BEGIN: DINOv3 Block Construction
        # ===============================================================
        
        # 1. 实例化 RoPE
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=dino_num_heads,
            base=pos_embed_rope_base,
            dtype=dtype_dict.get(dino_rope_dtype, torch.bfloat16),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu") # 假设在 GPU 上运行
        )
        
        # 2. 获取 DINOv3 的 FFN 和 Norm 层
        ffn_layer_cls = ffn_layer_dict[dino_ffn_layer]
        norm_layer_cls = norm_layer_dict[dino_norm_layer]

        # 3. 计算 DINOv3 的 stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, dino_depth)]  # stochastic depth
        
        # 4. 构建 DINOv3 SelfAttentionBlocks
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=dino_num_heads,
                ffn_ratio=dino_ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=True,
                ffn_bias=True,
                drop_path=dpr[i],
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init
            )
            for i in range(dino_depth)])
        
        # 5. DINOv3 的最终 Norm
        self.norm = norm_layer_cls(embed_dim)

        # ===============================================================
        # END: DINOv3 Block Construction
        
        # build the last conv layer in deep feature extraction 构建深度特征提取中的最后一个卷积层
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
        # 这个 Upsample 会将特征图从 H/P, W/P 恢复到 H, W
        self.decoder_feat_upsampler = Upsample(scale=vit_patch_size, num_feat=embed_dim)
        #####################################################################################################
        ################################ 3, high quality image reconstruction ################################
        # for classical SR
        self.uplayers = nn.ModuleList()
        if upscale == 11:
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat * 4, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            for i in range(3):
                self.uplayers.append(nn.Sequential(Upsample2(num_feat),
                            nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1),
                            nn.BatchNorm2d(num_features=num_feat * 4), nn.ReLU(inplace=True)))
            self.uplayers.append(nn.Sequential(Upsample2(num_feat),
                         nn.Conv2d(num_feat, num_feat, 3, 1, 1)))
        else:
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)  #

        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.apply(self._init_weights)
        
        # 保存 LoRA 配置参数，供 inject_lora() 方法使用
        self._lora_config = {
            'r': lora_r,
            'alpha': lora_alpha,
            'dropout': lora_dropout
        }
        self._lora_injected = False

        # 如果启用 LoRA，在初始化时注入
        if use_lora:
            self.inject_lora()
    
    def inject_lora(self, r=None, alpha=None, dropout=None):
        """
        注入 LoRA 到 DINOv3 主干。可以在加载预训练权重后单独调用。
        使用原生实现，不依赖 peft 库。
        
        Args:
            r: LoRA 秩，默认使用初始化时的配置
            alpha: LoRA alpha，默认使用初始化时的配置  
            dropout: LoRA dropout，默认使用初始化时的配置
        """
        if self._lora_injected:
            print(">>> [跳过] LoRA 已经注入，无需重复操作")
            return
        
        # 使用传入参数或默认配置
        r = r or self._lora_config['r']
        alpha = alpha or self._lora_config['alpha']
        dropout = dropout or self._lora_config['dropout']
        
        # 1. 冻结 DINOv3 所有的主干参数（包括 blocks, rope_embed, patch_embed, norm 等）
        for param in self.blocks.parameters():
            param.requires_grad = False
        if hasattr(self, 'rope_embed'):
            for param in self.rope_embed.parameters():
                param.requires_grad = False
        # if hasattr(self, 'patch_embed'):
        #     for param in self.patch_embed.parameters():
        #         param.requires_grad = False
        # if hasattr(self, 'patch_unembed'):
        #     for param in self.patch_unembed.parameters():
        #         param.requires_grad = False
        # if hasattr(self, 'norm'):
        #     for param in self.norm.parameters():
        #         param.requires_grad = False
        # if hasattr(self, 'pos_drop'):
        #     for param in self.pos_drop.parameters():
        #         param.requires_grad = False
        
        # 2. 注入 LoRA 到每个 SelfAttentionBlock 的 qkv 层
        lora_count = 0
        for block in self.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                # 用 LoRALinear 包装原始 qkv 层
                block.attn.qkv = LoRALinear(
                    block.attn.qkv, 
                    r=r, 
                    alpha=alpha, 
                    dropout=dropout
                )
                lora_count += 1
        
        self._lora_injected = True
        
        # 统计可训练参数
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f">>> [成功] 已冻结 DINOv3 主干并注入 LoRA (r={r}, alpha={alpha})")
        print(f">>> [统计] 注入了 {lora_count} 个 LoRA 层")
        print(f">>> [统计] 可训练参数: {trainable/1e6:.2f}M / 总参数: {total/1e6:.2f}M ({trainable/total:.1%})")
    
    def merge_lora_weights(self):
        """合并 LoRA 权重到基础模型（用于推理加速）"""
        if not self._lora_injected:
            print(">>> [跳过] 未注入 LoRA")
            return
        
        merge_count = 0
        for block in self.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                if isinstance(block.attn.qkv, LoRALinear):
                    block.attn.qkv = block.attn.qkv.merge_weights()
                    merge_count += 1
        
        self._lora_injected = False
        print(f">>> [成功] 已合并 {merge_count} 个 LoRA 层到基础模型")
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        # Use ViT patch size to pad input so PatchEmbed produces integer grid
        try:
            patch_H, patch_W = make_2tuple(self.patch_embed.patch_size)
        except Exception:
            # fallback to window_size if patch info not available
            patch_H = patch_W = self.window_size
        mod_pad_h = (patch_H - h % patch_H) % patch_H
        mod_pad_w = (patch_W - w % patch_W) % patch_W
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    
    def half(self):
        self.precision = torch.half
        print('half')
        super().half()
    
    # DINOv3 深层特征提取部分
    def forward_features(self, x):
       
        # 在 patch_embed 之前，根据输入的实际 H, W 计算 patch grid 大小
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_embed.patch_size
        H_patch = H // patch_H
        W_patch = W // patch_W

        # (B, 3, H, W) -> (B, N, C) e.g., (B, 49, 768)
        x = self.patch_embed(x) 
        x = self.pos_drop(x)
        
        # 计算 DINOv3 RoPE
        self.rope_embed.device = x.device
        # 使用动态计算的 H_patch, W_patch
        rope_sincos = self.rope_embed(H=H_patch, W=W_patch) 
        
        # 运行 DINOv3 blocks 
        for i, blk in enumerate(self.blocks): 
            x = blk(x, rope_cos=rope_sincos[0], rope_sin=rope_sincos[1])
        
        # 对最后一层的输出进行归一化
        x = self.norm(x)
        
        # Un-embed features
        # 告诉 patch_unembed 使用 patch grid 大小
        x_unembedded = self.patch_unembed(x, (H_patch, W_patch))

        # 返回单个 (B, C, H/P, W/P) 特征图，不再是列表
        return x_unembedded
    

    def forward(self, x, task_id=0):
        # task_id 参数用于兼容多任务调用，单任务 SR 模型忽略它
        # 只有当模型配置为 3 通道 (预训练模式)，但输入是 1 通道时，才进行复制
        if x.dim() == 4 and x.size(1) == 1 and self.patch_embed.in_chans == 3:
            x = x.repeat(1, 3, 1, 1)

        H, W = x.shape[2:]
        x_check = self.check_image_size(x)
        self.mean = self.mean.type_as(x_check)
        x_norm = (x_check - self.mean) * self.img_range 

        # 再次检查：确保输入 PatchEmbed 的数据通道数与 PatchEmbed 层的权重匹配
        if x_norm.shape[1] == 1 and self.patch_embed.in_chans == 3:
            x_input = x_norm.repeat(1, 3, 1, 1)
        else:
            x_input = x_norm 
        
        # === 3. 深层特征提取 ===
        x_features = self.forward_features(x_input)
        
        # === 5. 主残差连接 ===
        x_body = self.conv_after_body(x_features)      

        # 将特征图恢复到 H, W (e.g., 64, 64)
        x_body_upsampled = self.decoder_feat_upsampler(x_body) 

        # === 6. 高质量重建 ===
        x_out = self.conv_before_upsample(x_body_upsampled) 
        
        if self.upscale == 11:
            for layer in self.uplayers:
                x_out = layer(x_out)
            x_out = F.interpolate(x_out, size=(H * self.upscale, W * self.upscale), mode='bilinear')
        else:
            x_out = self.upsample(x_out)
        
        x_out = self.conv_last(x_out)
        x_out = x_out / self.img_range + self.mean
        
        return x_out[:, :, :H * self.upscale, :W * self.upscale]
    
################################## Projection ##############################
# 改名为 dinoProj_stage2
class dinoProj_stage2(nn.Module):
    def __init__(self, img_size=64, patch_size=1, out_chans=1,
                 embed_dim=180 // 2, depths=[6, 6, 6], num_heads=[6, 6, 6],
                 window_size=8, mlp_ratio=2., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., num_feat=32, args=None):
        super(dinoProj_stage2, self).__init__()
        
        self.project = ENLCN(args=args)
        # 这里复用了 dinov3 类，参数保持原 SwinIR 的接口参数名，但由 dinov3 类内部映射到 DINO 参数
        self.denoise = dinov3(img_size, patch_size, 1, out_chans,
                              embed_dim, depths, num_heads, window_size, mlp_ratio, qkv_bias, qk_scale,
                              drop_rate, attn_drop_rate, drop_path_rate, norm_layer, ape, patch_norm, use_checkpoint,
                              upscale, img_range, num_feat)
        
    def half(self):
        self.denoise.half()

    def forward(self, x):
        if self.training:  
            x2d, closs = self.project(x)
            x = self.denoise(x2d) + x2d
            return x2d, x, closs
        else:
            import time
            torch.cuda.synchronize()
            st = time.time()
            x2d = self.project(x)
            torch.cuda.synchronize()
            print('project', time.time() - st)
            x = self.denoise(x2d) + x2d
            torch.cuda.synchronize()
            print('denoise', time.time() - st)
            return x2d, x


################################## Volumetric ##############################
# 重写为 DINO 版本的 VCD 模型
class dinov3_2dto3d(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=121, out_chans=61,
                 embed_dim=384, # ViT-S 默认维度 (原 ViT-B 为 768)
                 # DINOv3 配置
                 dino_depth=12, dino_num_heads=6, dino_ffn_ratio=4.0,  # ViT-S 的头数
                 layerscale_init=1e-4, dino_norm_layer='layernorm', 
                 dino_ffn_layer='mlp', pos_embed_rope_base=100.0,
                 dino_rope_dtype='bf16',
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_norm=True,
                 upscale=2, img_range=1., num_feat=32):
        super(dinov3_2dto3d, self).__init__()

        self.precision = torch.float32
        self.img_range = img_range
        self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.window_size = 8 # 这里的 window_size 主要用于 padding 计算
        
        # 1. 浅层特征提取 (UNetA + Conv)
        self.conv_first0 = UNetA(in_chans, out_chans)
        self.conv_first = nn.Conv2d(out_chans, embed_dim, 3, 1, 1)
        
        # 2. 深层特征提取 (DINOv3 Backbone)
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=8, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=8, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None)
        
        self.pos_drop = nn.Dropout(p=0.0)

        # --- DINOv3 组件 ---
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=dino_num_heads,
            base=pos_embed_rope_base,
            dtype=dtype_dict.get(dino_rope_dtype, torch.bfloat16),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        ffn_layer_cls = ffn_layer_dict[dino_ffn_layer]
        norm_layer_cls = norm_layer_dict[dino_norm_layer]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, dino_depth)]

        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=dino_num_heads,
                ffn_ratio=dino_ffn_ratio,
                drop_path=dpr[i],
                norm_layer=norm_layer_cls,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init
            )
            for i in range(dino_depth)])
        
        self.norm = norm_layer_cls(embed_dim)
        # ------------------
        
        self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        self.decoder_feat_upsampler = Upsample(scale=8, num_feat=embed_dim) # 对应 patch_size=8

        # 3. 重建模块
        self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
                                                  nn.LeakyReLU(inplace=True))
        self.conv_last = nn.Conv2d(embed_dim, out_chans, 3, 1, 1)
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def check_image_size(self, x):
        _, _, h, w = x.size()
        # 使用 patch_size (8) 进行 padding
        mod_pad_h = (8 - h % 8) % 8
        mod_pad_w = (8 - w % 8) % 8
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward_features(self, x):
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_embed.patch_size
        H_patch = H // patch_H
        W_patch = W // patch_W

        x = self.patch_embed(x)
        x = self.pos_drop(x)
        
        self.rope_embed.device = x.device
        rope_sincos = self.rope_embed(H=H_patch, W=W_patch) 
        
        for blk in self.blocks:
            x = blk(x, rope_cos=rope_sincos[0], rope_sin=rope_sincos[1])
        
        x = self.norm(x)
        x = self.patch_unembed(x, (H_patch, W_patch))
        return x
    
    def forward(self, x):
        if x.dim() == 5: # VCD data sometimes comes as 5D
             x = x.squeeze(1)

        x = self.check_image_size(x)
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range
        
        # 浅层特征 (UNetA + Conv)
        xunet = self.conv_first0(x)
        x_first = self.conv_first(xunet)

        # 深层特征 (DINOv3)
        x_deep = self.forward_features(x_first)
        x_body = self.conv_after_body(x_deep)
        
        # 上采样回原尺寸 (64->64, patch_size=8 的恢复)
        x_body_upsampled = self.decoder_feat_upsampler(x_body)

        # 重建
        x = self.conv_before_upsample(x_body_upsampled)
        x = self.conv_last(x)

        x = x / self.img_range + self.mean
        return xunet, x

# --- 这是 DINOv3 的 PatchEmbed 实现 ---
def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        # patch_H, patch_W = self.patch_size
        # assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        # assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size: Tuple[int,int]):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 1, 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if scale == 1:
            # scale=1 表示不需要上采样 (denoise 任务)
            m.append(nn.Identity())
        elif (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        elif scale == 11:
            m.append(torch.nn.Upsample(scale_factor=scale))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 1, 2^n and 3.')
        super(Upsample, self).__init__(*m)


class Upsample2(nn.Sequential):
    def __init__(self, num_feat=32):
        m = []
        m.append(nn.Conv2d(4 * num_feat, 4 * num_feat, 3, 1, 1))
        m.append(nn.PixelShuffle(2))
        super(Upsample2, self).__init__(*m)


class UNetA(nn.Module):
    def __init__(self, num_in_ch, n_slices):
        super(UNetA, self).__init__()
        self.upscale = 11
        n_interp = 4
        channels_interp = 128
        self.conv2d = nn.Conv2d(num_in_ch, channels_interp, 7, 1, 3)
        
        ## Up-scale input
        self.layers = nn.ModuleList()
        for i_layer in range(n_interp):
            channels_interp = channels_interp // 2
            self.layers.append(nn.Sequential(Upsample2(channels_interp // 2),
                                             nn.Conv2d(channels_interp // 2, channels_interp, 3, 1, 1)))
        self.conv2d1 = nn.Sequential(nn.Conv2d(channels_interp, channels_interp, 3, 1, 1),
                                     nn.BatchNorm2d(num_features=channels_interp), nn.ReLU(inplace=True))
        
        pyramid_channels = [128, 256, 512, 512, 512]
        inch = 64
        self.conv2d2 = nn.Sequential(nn.Conv2d(channels_interp, inch, 3, 1, 1),  # 64
                                     nn.BatchNorm2d(num_features=inch),
                                     nn.ReLU(inplace=True))  # [1,944,944,64]
        self.encoder_layers = nn.ModuleList()
        for idx, nc in enumerate(pyramid_channels):
            layers = nn.Sequential(nn.Conv2d(inch, nc, 3, 1, 1), nn.BatchNorm2d(num_features=nc), nn.ReLU(inplace=True))
            inch = nc
            self.encoder_layers.append(layers)
        
        # decoder
        nl = len(self.encoder_layers)  # [1,944,944,64]~ [1,59,59,512]
        self.decoder_layers = nn.ModuleList()
        for idx in range(nl - 1, -1, -1):  # idx = 4,3,2,1,0
            if idx > 0:
                inch = pyramid_channels[idx] + pyramid_channels[idx - 1]
                out_channels = pyramid_channels[idx - 1]
            else:
                inch = pyramid_channels[idx] + 64
                out_channels = n_slices
            layers = nn.Sequential(nn.Conv2d(inch, out_channels, 3, 1, 1),
                                   nn.ReLU(inplace=True),
                                   nn.BatchNorm2d(out_channels))
            self.decoder_layers.append(layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x):
        H, W = x.shape[2:]
        
        x = self.conv2d(x)
        ## Up-scale input
        for layer in self.layers:
            x = layer(x)
        x = self.conv2d1(x)
        
        encoder_layers = []
        # 'encoder':
        x = self.conv2d2(x)  # [1,944,944,64]
        id0 = 0
        for layer in self.encoder_layers:
            encoder_layers.append(x)  # append n0, n1, n2, n3, n4 (but without n5)to the layers list
            x = layer(x)
            y = torch.zeros_like(x)
            # new_channels = pyramid_channels[id0] - x.shape[1]
            y[:, :encoder_layers[-1].shape[1], :, :] = encoder_layers[-1]
            # n1 = torch.concat([x, y], dim=1)
            id0 += 1
            x = nn.MaxPool2d(kernel_size=3, stride=2)(x + y)  # [1,30,30,512]
        
        # decoder # [1,944,944,64]~ [1,59,59,512]
        x = F.interpolate(x, size=(encoder_layers[-1].shape[2:]), mode='bilinear')  # [1,59,59,512]
        idx = 4
        for layer in self.decoder_layers:  # idx = 4,3,2,1,0
            if idx > 0:
                H0, W0 = encoder_layers[idx - 1].shape[2:]  # x.shape[2:]
            else:
                H0, W0 = H * self.upscale, W * self.upscale
            x = torch.cat([encoder_layers[idx], x], dim=1)
            x = layer(x)  # n [1,944,944,61]
            x = F.interpolate(x, size=(H0, W0), mode='bilinear')
            idx -= 1
        
        x = F.interpolate(x, size=(H * self.upscale, W * self.upscale), mode='bilinear')
        x = torch.tanh(x)
        return x


class DinoUniModel(nn.Module):
    def __init__(self, args, embed_dim=384, dino_depth=12, dino_num_heads=6, vit_patch_size=8):  # ViT-S 配置
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
        # 4. Projection (50通道输入，对应50个Z切片)
        args_proj = copy.deepcopy(args)
        args_proj.inch = 50  # 修改输入通道数为50
        self.project = ENLCN(args=args_proj)
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

        # --- 关键：恢复 patch 分辨率的上采样器 ---
        self.decoder_feat_upsampler = Upsample(scale=vit_patch_size, num_feat=embed_dim)

        # --- 任务特定尾 (Tails) ---
        self.conv_before_upsample0 = nn.Sequential(nn.Conv2d(embed_dim, 32, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.upsamplesr = Upsample(scale=2, num_feat=32) # Task 1: 2x 上采样
        self.upsample_common = Upsample(scale=1, num_feat=32) # Task 2, 3: 无上采样
        self.conv_last0 = nn.Conv2d(32, 1, 3, 1, 1)
        
        # Task 5 特有尾部
        self.decoder_feat_upsampler_v = Upsample(scale=vit_patch_size, num_feat=embed_dim)
        self.conv_before_upsamplev = nn.Sequential(nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.LeakyReLU(inplace=True))
        self.conv_lastv = nn.Conv2d(embed_dim, 61, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        B, C, H_orig, W_orig = x.shape
        grid_h, grid_w = H_orig // self.vit_patch_size, W_orig // self.vit_patch_size
        x = self.patch_embed(x)
        cos, sin = self.rope_embed(grid_h, grid_w, device=x.device)  # 传递输入的 device
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = self.norm(x)
        x = self.patch_unembed(x, (grid_h, grid_w))
        x = self.conv_after_body(x)
        # 恢复到原始分辨率 H, W
        x = self.decoder_feat_upsampler(x)
        # 确保输出尺寸与输入一致（处理不能被 patch_size 整除的情况）
        if x.shape[2] != H_orig or x.shape[3] != W_orig:
            x = F.interpolate(x, size=(H_orig, W_orig), mode='bilinear', align_corners=False)
        return x

    def forward(self, x, tsk=0):
        if tsk > 0: self.task = tsk
        self.mean = self.mean.type_as(x)
        H_orig, W_orig = x.shape[2], x.shape[3]
        
        # --- Head 阶段 ---
        if self.task == 1: 
            x_first = self.conv_firstsr((x - self.mean) * self.img_range)

        elif self.task == 2: 
            # 针对去噪任务：输入已经是 5 通道（滑动窗口）
            x_input = (x - self.mean) * self.img_range
            x_first = self.conv_firstdT(x_input)
            
        elif self.task == 3: 
            x_first = self.conv_firstiso((x - self.mean) * self.img_range)
        elif self.task == 4:
            proj_out = self.project(x)
            # ENLCN 训练时返回 (x, loss)，测试时只返回 x
            x2d = proj_out[0] if isinstance(proj_out, tuple) else proj_out
            x_first = self.conv_firstproj((x2d - self.mean) * self.img_range)
        elif self.task == 5:
            xunet = self.conv_first0(x)
            x_first = self.conv_firstv(xunet)

        # --- Body 阶段 ---
        xfe = self.forward_features(x_first)  # 现在 xfe 已经恢复到原始 H, W

        # --- Tail 阶段 ---
        if self.task in [1, 2, 3]:
            x_out = self.conv_before_upsample0(xfe + x_first)  # 残差连接
            x_out = self.upsamplesr(x_out) if self.task == 1 else self.upsample_common(x_out)
            x_out = self.conv_last0(x_out)
            return x_out / self.img_range + self.mean
        elif self.task == 4:
            x_out = self.conv_last0(self.conv_before_upsample0(xfe))
            return x2d, x_out / self.img_range + self.mean + x2d
        elif self.task == 5:
            # 任务5: xunet 是 (H, W)，需要上采样到 (H*11, W*11)
            # xfe 目前是 (H, W)，需要上采样到 (H*11, W*11)
            x_out = self.conv_before_upsamplev(xfe)
            x_out = F.interpolate(x_out, size=(H_orig * 11, W_orig * 11), mode='bilinear', align_corners=False)
            x_out = self.conv_lastv(x_out)
            # 对 xunet 也进行上采样以匹配 hr 尺寸
            xunet_up = F.interpolate(xunet, size=(H_orig * 11, W_orig * 11), mode='bilinear', align_corners=False)
            return xunet_up, x_out / self.img_range + self.mean

        return x_out / self.img_range + self.mean

    def inject_lora(self, r=16, alpha=32, dropout=0.05):
        """
        注入 LoRA 到 DINOv3 主干的 attention 层。
        
        Args:
            r: LoRA 秩
            alpha: LoRA alpha 缩放系数
            dropout: LoRA dropout
        """
        # 检查是否已注入（通过检查第一个 block 的 qkv 是否为 LoRALinear）
        if hasattr(self.blocks[0].attn.qkv, 'original_linear'):
            print(">>> [跳过] LoRA 已经注入，无需重复操作")
            return
        
        # 1. 冻结主干参数
        for param in self.blocks.parameters():
            param.requires_grad = False
        if hasattr(self, 'rope_embed'):
            for param in self.rope_embed.parameters():
                param.requires_grad = False
        
        # 2. 注入 LoRA 到每个 SelfAttentionBlock 的 qkv 层
        lora_count = 0
        for block in self.blocks:
            if hasattr(block, 'attn') and hasattr(block.attn, 'qkv'):
                block.attn.qkv = LoRALinear(
                    block.attn.qkv, 
                    r=r, 
                    alpha=alpha, 
                    dropout=dropout
                )
                lora_count += 1
        
        # 统计可训练参数
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f">>> [成功] 已冻结 DINOv3 主干并注入 LoRA (r={r}, alpha={alpha})")
        print(f">>> [统计] 注入了 {lora_count} 个 LoRA 层")
        print(f">>> [统计] 可训练参数: {trainable/1e6:.2f}M / 总参数: {total/1e6:.2f}M ({trainable/total:.1%})")