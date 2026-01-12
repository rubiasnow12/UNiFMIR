import torch
import argparse
import sys
import os

# 1. 导入我们修改后的模型
print("正在导入 'make_model' 从 'model.dinoir_v3'...")
try:
    from model.dinoir_v3 import dinov3  # 直接导入类以便更灵活地创建
    print("...导入成功！")
except Exception as e:
    print(f"导入失败: {e}")
    sys.exit(1)

# 2. 定义 DINOv3 ViT-B 权重文件的路径
dino_checkpoint_path = 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth' 
output_checkpoint_path = 'dinoir_v3_vitb_unipreload.pth'  # ← 输出文件名（通用权重）

if not os.path.exists(dino_checkpoint_path):
    print(f"错误: 未找到 DINOv3 权重文件 '{dino_checkpoint_path}'")
    sys.exit(1)

# 3. 创建一个 "通用" 的 dinov3 模型实例
#    这里使用 upscale=2 和 in_chans=3，因为：
#    - DINOv3 backbone (blocks) 不依赖于 upscale 或 in_chans
#    - 我们只需要加载 backbone 权重，head/tail 会在微调时重新初始化
print("正在实例化 dinoir_v3 (ViT-B 尺寸) 模型...")

# 直接实例化，使用默认参数
model = dinov3(
    in_chans=3,      # 伪三通道（与 DINOv3 预训练一致）
    upscale=2,       # 任意值，backbone 不依赖这个
    embed_dim=768,   # ViT-B 的 embed_dim
    dino_depth=12,   # ViT-B 有 12 层
    dino_num_heads=12,  # ViT-B 有 12 个头
)
model_state_dict = model.state_dict()
print("...模型实例化成功。")
print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 5. 加载 DINOv3 预训练权重
print(f"正在加载 DINOv3 预训练权重从 '{dino_checkpoint_path}'...")
dino_weights = torch.load(dino_checkpoint_path, map_location='cpu')
print("...DINOv3 权重加载成功。")
print(f"DINOv3 权重包含 {len(dino_weights)} 个键。")

# 6. 核心步骤：部分加载 (Partial Load)
#    加载 DINOv3 的 blocks 和 norm 层到我们的模型
print("开始匹配权重键 (key)...")
new_state_dict = {}
loaded_keys = 0
skipped_keys = []

for dino_key, dino_value in dino_weights.items():
    # 我们关心 'blocks' (Transformer 主干) 和 'norm' (最终归一化层)
    if dino_key.startswith('blocks.') or dino_key.startswith('norm.'):
        # 检查这个键是否存在于我们的模型中
        if dino_key in model_state_dict:
            # 检查形状是否匹配
            if model_state_dict[dino_key].shape == dino_value.shape:
                new_state_dict[dino_key] = dino_value
                loaded_keys += 1
            else:
                skipped_keys.append(f"{dino_key} (形状不匹配: {dino_value.shape} vs {model_state_dict[dino_key].shape})")
        else:
            skipped_keys.append(f"{dino_key} (模型中不存在)")

print(f"...匹配完成。")
print(f"  ✅ 成功匹配并准备加载 {loaded_keys} 个键 (来自 'blocks' 和 'norm')。")
print(f"  ⏭️  跳过了 {len(skipped_keys)} 个不相关/不匹配的键。")
if skipped_keys and len(skipped_keys) <= 10:
    print("  跳过的键:")
    for k in skipped_keys:
        print(f"    - {k}")

# 7. 加载过滤后的权重到我们的模型中
#    strict=False 意味着它会忽略所有 "Missing key(s)" 
#    (例如 patch_embed, upsample, conv_last 等，这是我们期望的)
print("正在将 DINOv3 backbone 权重加载到新模型中...")
model.load_state_dict(new_state_dict, strict=False)
print("...部分加载成功！")

# 8. 保存新的混合权重文件
print(f"正在将部分加载的模型保存到 '{output_checkpoint_path}'...")
torch.save(model.state_dict(), output_checkpoint_path)

print("\n" + "="*60)
print("✅ 全部完成!")
print("="*60)
print(f"\n📁 生成的权重文件: '{output_checkpoint_path}'")
print("\n📝 使用说明:")
print("   这个权重文件包含了 DINOv3 预训练的 backbone (blocks + norm)，")
print("   以及随机初始化的 head/tail 层 (patch_embed, upsample 等)。")
print("\n   您可以将此文件用于以下任务的微调:")
print("   - SR (超分辨率): scale=2, 使用 finetune_dinoir_v3_sr.py")
print("   - Denoise (去噪): scale=1, 需要创建 finetune_dinoir_v3_denoise.py")
print("   - Projection: 使用 dinoProj_stage2")
print("   - 2D to 3D: 使用 dinov3_2dto3d")
print("\n   注意: head/tail 层的权重会在首次微调时根据具体任务自动调整。")