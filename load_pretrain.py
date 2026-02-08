import torch
import argparse
import sys
import os

# 1. 导入我们修改后的模型
print("正在导入模型...")
try:
    from model.dinoir_v3 import dinov3  # 导入 dinov3 类（用于 SR 任务）
    from model.dinoir_v3 import DinoUniModel  # 导入通用模型类 V1
    from model.dinoir_v3 import DinoUniModelV2  # 导入通用模型类 V2 (Task Embedding + FiLM)
    print("...导入成功！")
except Exception as e:
    print(f"导入失败: {e}")
    sys.exit(1)

# 2. 定义 DINOv3 ViT-S 权重文件的路径
dino_checkpoint_path = 'dinov3_vits16_pretrain_lvd1689m-08c60483.pth' 

# ========== 选择要创建的模型类型 ==========
# 'sr'     → dinov3（SR 单任务）
# 'v1'     → DinoUniModel（多头任务，每个任务独立 conv_first）
# 'v2'     → DinoUniModelV2（Task Embedding + FiLM 调制 + 统一输入层）
MODEL_TYPE = 'v2'  # ← 改这里选择模型版本

# 根据模型类型设置输出文件名
output_checkpoint_map = {
    'sr': 'dinoir_v3_vits_sr_preload.pth',
    'v1': 'dinoir_v3_vits_unipreload.pth',
    'v2': 'dinoir_v3_vits_v2preload.pth',
}
output_checkpoint_path = output_checkpoint_map[MODEL_TYPE]


# 构造模拟参数
mock_args = argparse.Namespace()
mock_args.n_resblocks = 8
mock_args.n_feats = 32
mock_args.scale = [1]
mock_args.inch = 1
mock_args.n_colors = 1
mock_args.rgb_range = 1
mock_args.res_scale = 1.0
mock_args.dilation = False

if MODEL_TYPE == 'v2':
    print("正在实例化 DinoUniModelV2 (Task Embedding + FiLM) 模型...")
    model = DinoUniModelV2(
        args=mock_args,
        embed_dim=384,
        dino_depth=12,
        dino_num_heads=6,
        task_embed_dim=64,
    )
elif MODEL_TYPE == 'v1':
    print("正在实例化 DinoUniModel V1 (ViT-S) 模型...")
    model = DinoUniModel(
        args=mock_args,
        embed_dim=384,
        dino_depth=12,
        dino_num_heads=6,
    )
else:
    print("正在实例化 dinov3 (ViT-S) SR 专用模型...")
    model = dinov3(
        in_chans=1,
        out_chans=1,
        embed_dim=384,
        dino_depth=12,
        dino_num_heads=6,
        upscale=2,
        use_lora=False,
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
print(f"\n📝 模型类型: {MODEL_TYPE}")
if MODEL_TYPE == 'v2':
    print("\n📝 V2 多任务训练流程 (Task Embedding + FiLM):")
    print("   1. 在 mainUi_pretrain.py 中设置:")
    print("      - USE_V2 = True")
    print("      - freeze_depth = 0  (冻结前0层)")
    print(f"      - preloaded_path = './{output_checkpoint_path}'")
    print("   2. 运行 python mainUi_pretrain.py")
    print("   3. 系统会自动: 加载权重 → FiLM 调制多任务训练")
else:
    print("\n📝 V1/SR 训练流程:")
    print(f"      - preloaded_path = './{output_checkpoint_path}'")
    print("   运行 python mainUi_pretrain.py 或 python mainSR_dino.py")