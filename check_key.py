import torch
import sys

# 将路径替换为您的权重文件路径，例如 'dinoir_v3_vitb_preloaded.pth'
checkpoint_path = 'dinoir_v3_vitb_preloaded.pth' 

try:
    print(f"正在加载: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # 有些 checkpoint 保存时包含 'state_dict' 键，有些直接是字典
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    print("\n--- 参数名称列表 (前 10 个) ---")
    for i, key in enumerate(state_dict.keys()):
        if i >= 10: break
        print(key)
        
    print("\n--- 查找包含 'blocks' 的参数 ---")
    for key in state_dict.keys():
        if 'blocks.0.' in key:
            print(f"找到: {key}")
            break # 找到一个就停，作为示例

except Exception as e:
    print(f"发生错误: {e}")