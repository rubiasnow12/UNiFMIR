import torch
import argparse
import sys
import os

# 1. 导入我们修改后的模型
print("正在导入 'make_model' 从 'model.dinoir_v3'...")
try:
    from model.dinoir_v3 import make_model
    print("...导入成功！")
except Exception as e:
    print(f"导入失败: {e}")
    sys.exit(1)

# 2. 定义 DINOv3 ViT-S 权重文件的路径
dino_checkpoint_path = 'dinov3_vits16_pretrain.pth'
output_checkpoint_path = 'dinoir_v3_vits_preloaded.pth'

if not os.path.exists(dino_checkpoint_path):
    print(f"错误: 未找到 DINOv3 权重文件 '{dino_checkpoint_path}'")
    sys.exit(1)

# 3. 创建一个模拟的 'args' 对象 (必须与 dinoir_v3.py 中的新默认值匹配)
args = argparse.Namespace()
args.scale = [2]  
args.inputchannel = 1    

# 4. 实例化我们新的 (ViT-B 尺寸的) dinoir_v3 模型
print("正在实例化 dinoir_v3 (ViT-s尺寸) 模型...")
# make_model 会自动使用 dinoir_v3.py 中新的默认值 (embed_dim=768, heads=12)
model = make_model(args)
model_state_dict = model.state_dict()
print("...模型实例化成功。")

# 5. 加载 DINOv3 预训练权重
print(f"正在加载 DINOv3 预训练权重从 '{dino_checkpoint_path}'...")
dino_weights = torch.load(dino_checkpoint_path, map_location='cpu')
print("...DINOv3 权重加载成功。")

# 6. 核心步骤：部分加载 (Partial Load)
print("开始匹配权重键 (key)...")
new_state_dict = {}
loaded_keys = 0
unmatched_keys = 0

for dino_key, dino_value in dino_weights.items():
    # 我们只关心 'blocks' (即 Transformer 主干)
    if dino_key.startswith('blocks.'):
        # 检查这个键是否存在于我们的模型中
        if dino_key in model_state_dict:
            # 检查形状是否匹配 (以防万一)
            if model_state_dict[dino_key].shape == dino_value.shape:
                new_state_dict[dino_key] = dino_value
                loaded_keys += 1
            else:
                print(f"  [警告] 形状不匹配，跳过: {dino_key}")
                unmatched_keys += 1
        else:
            unmatched_keys += 1
    else:
        unmatched_keys += 1

print(f"...匹配完成。")
print(f"  成功匹配并准备加载 {loaded_keys} 个键 (来自 'blocks')。")
print(f"  跳过了 {unmatched_keys} 个不相关/不匹配的键。")

# 7. 加载过滤后的权重到我们的模型中
#    strict=False 意味着它会忽略所有 "Missing key(s)" 
#    (例如 conv_first, upsample, conv_last 等，这是我们期望的)
print("正在将 DINOv3 'blocks' 权重加载到新模型中...")
model.load_state_dict(new_state_dict, strict=False)
print("...部分加载成功！")

# 8. 保存新的混合权重文件
print(f"正在将部分加载的模型保存到 '{output_checkpoint_path}'...")
torch.save(model.state_dict(), output_checkpoint_path)
print("\n--- 全部完成! ---")
print(f"您现在可以将 '{output_checkpoint_path}' 用作 'modelpath' 来开始微调。")