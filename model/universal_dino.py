import torch
import torch.nn as nn
from model.dinoir_v3 import dinov3, UNetA, Upsample, Upsample2
from model.enlcn import ENLCN

class UniversalDino(nn.Module):
    def __init__(self, args, 
                 img_size=64, 
                 embed_dim=384,          # ViT-S (原 ViT-B 为 768)
                 patch_size=8,           # ViT Patch Size
                 dino_num_heads=6,       # ViT-S (原 ViT-B 为 12)
                 dino_depth=12,          # ViT-S/B 都是 12 层
                 # ... 其他参数保持默认 ...
                 ):
        super(UniversalDino, self).__init__()
        
        self.embed_dim = embed_dim
        self.img_range = 1.0
        self.mean = torch.zeros(1, 1, 1, 1)
        
        # 1. 共享的“身体” (Backbone)
        # 我们实例化一个 dinov3 作为基础，复用它的 blocks, norm, rope
        # 注意：这里随便传个 in_chans=3 即可，因为我们主要用它的骨架
        self.backbone_model = dinov3(img_size=img_size, patch_size=1, in_chans=3, 
                                     embed_dim=embed_dim, depths=[2,2,6,2], 
                                     dino_depth=dino_depth, dino_num_heads=dino_num_heads)
        
        # 提取共享组件
        self.pos_drop = self.backbone_model.pos_drop
        self.rope_embed = self.backbone_model.rope_embed
        self.blocks = self.backbone_model.blocks
        self.norm = self.backbone_model.norm
        self.conv_after_body = self.backbone_model.conv_after_body
        self.decoder_feat_upsampler = self.backbone_model.decoder_feat_upsampler # 恢复 patch 分辨率
        
        # 2. 定义“多头” (Task-specific Heads)
        # 我们需要为不同任务克隆 patch_embed 结构
        def make_head(in_chans):
            # 复用 dinov3 中的 PatchEmbed 类定义
            return self.backbone_model.patch_embed.__class__(
                img_size=img_size, patch_size=patch_size, 
                in_chans=in_chans, embed_dim=embed_dim, 
                norm_layer=None
            )

        self.heads = nn.ModuleDict({
            'sr': make_head(in_chans=1),       # Task 1
            'denoise': make_head(in_chans=5),  # Task 2
            'iso': make_head(in_chans=1),      # Task 3
            'proj': make_head(in_chans=1),     # Task 4 (接收 ENLCN 输出的 1 通道)
            # Task 5 VCD 比较特殊: UNetA -> Conv -> PatchEmbed(in=embed_dim)
            'vcd_patch': make_head(in_chans=embed_dim) 
        })
        
        # Task 4 特有组件
        self.proj_head = ENLCN(args=args) # 投影模块
        
        # Task 5 特有组件
        self.vcd_unet = UNetA(121, 61)
        self.vcd_conv = nn.Conv2d(61, embed_dim, 3, 1, 1)

        # 3. 定义“多尾” (Task-specific Tails)
        # 大部分任务重建结构相似，但输出通道或上采样倍率不同
        self.tails = nn.ModuleDict()
        
        # 通用重建层 (Conv + LeakyReLU)
        self.common_recon = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True)
        )
        
        # 各任务特定的输出层
        self.tail_convs = nn.ModuleDict({
            'sr': nn.Conv2d(32, 1, 3, 1, 1),      # SR 输出 1 通道
            'denoise': nn.Conv2d(32, 1, 3, 1, 1), # Denoise 输出 1 通道
            'iso': nn.Conv2d(32, 1, 3, 1, 1),     # Iso 输出 1 通道
            'proj': nn.Conv2d(32, 1, 3, 1, 1),    # Proj 输出 1 通道
            'vcd': nn.Conv2d(embed_dim, 61, 3, 1, 1) # VCD 输出 61 通道
        })
        
        # 上采样模块
        self.upsamplers = nn.ModuleDict({
            'sr': Upsample(scale=2, num_feat=32), # SR x2
            'denoise': Upsample(scale=1, num_feat=32),
            'iso': Upsample(scale=1, num_feat=32),
            'proj': Upsample(scale=1, num_feat=32),
            # VCD 没有这个显式的 upsample 模块，直接用 tail_convs
        })
        
        # SR/Denoise/Iso/Proj 都有一个 conv_before_upsample0 (dim->32)
        self.conv_before_tail = nn.Sequential(
            nn.Conv2d(embed_dim, 32, 3, 1, 1), 
            nn.LeakyReLU(inplace=True)
        )

    def check_image_size(self, x):
        return self.backbone_model.check_image_size(x)

    def forward_backbone(self, x, head_key):
        # 1. Head 处理
        # Patch Embedding
        x = self.heads[head_key](x) 
        x = self.pos_drop(x)
        
        # 2. RoPE + Transformer Blocks
        # 计算 patch grid
        B, N, C = x.shape
        # 反推 H_patch, W_patch (简化逻辑，假设方形或通过 reshape)
        # 这里需要复用 dinov3 的逻辑来获取 H, W
        patch_H, patch_W = self.heads[head_key].patch_size
        # 由于输入已经是 token 序列，RoPE 需要动态计算
        # 注意：这里需要确保 patch_embed 保留了 H,W 信息，或者重新计算
        # 简单起见，我们重新调用一次 rope (这需要知道原始 H,W)
        # *为了代码稳健，建议在 forward 里传入原始尺寸或让 PatchEmbed 返回尺寸*
        # 这里假设 x 是 (B, N, C)，为了运行 blocks，需要传入 rope_cos/sin
        
        # 临时解决方案：从 N 反推 (假设正方形)，或者让 model 记住尺寸
        H_patch = int(N**0.5) 
        W_patch = N // H_patch
        
        self.rope_embed.device = x.device
        rope_sincos = self.rope_embed(H=H_patch, W=W_patch)
        
        for blk in self.blocks:
            x = blk(x, rope_cos=rope_sincos[0], rope_sin=rope_sincos[1])
            
        x = self.norm(x)
        
        # 3. Un-embed
        x = self.backbone_model.patch_unembed(x, (H_patch, W_patch))
        
        # 4. Conv After Body & Upsample back to pixel space
        x = self.conv_after_body(x)
        x = self.decoder_feat_upsampler(x)
        return x

    def forward(self, x, task_id=1):
        # 任务映射
        task_map = {1: 'sr', 2: 'denoise', 3: 'iso', 4: 'proj', 5: 'vcd'}
        key = task_map.get(task_id, 'sr')
        
        # 预处理
        x_in = x
        if key == 'proj':
            x_proj, _ = self.proj_head(x)
            x_in = x_proj
        elif key == 'vcd':
            x_unet = self.vcd_unet(x)
            x_in = self.vcd_conv(x_unet)
        
        # 标准化与 Padding
        if x_in.dim() == 5: x_in = x_in.squeeze(1) # VCD fix
        x_in = self.check_image_size(x_in)
        self.mean = self.mean.type_as(x_in)
        x_norm = (x_in - self.mean) * self.img_range
        
        # === 核心骨干 ===
        # VCD 的头比较特殊，它输入 PatchEmbed 的是 embed_dim 通道
        head_key = 'vcd_patch' if key == 'vcd' else key
        x_feat = self.forward_backbone(x_norm, head_key)
        
        # === 尾部重建 ===
        x_out = self.common_recon(x_feat)
        
        if key == 'vcd':
            # VCD 独特尾部
            out = self.tail_convs['vcd'](x_out)
            return self.vcd_unet(x), out / self.img_range + self.mean # 返回 unet输出 和 最终输出
        else:
            # SR/Denoise/Iso/Proj 共享结构
            # 这里的 x_out 加上残差 (除了 Proj)
            if key != 'proj':
                x_out = x_out + x_norm
            
            x_out = self.conv_before_tail(x_out)
            x_out = self.upsamplers[key](x_out)
            out = self.tail_convs[key](x_out)
            
            final = out / self.img_range + self.mean
            if key == 'proj':
                return x_proj, final + x_proj
            return final

def make_model(args):
    # 实例化模型
    return UniversalDino(args)