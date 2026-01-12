import torch
import argparse
import os
import sys

try:
    from model.universal_dino import UniversalDino
    print("导入 UniversalDino 成功！")
except ImportError:
    print("请确保 model/universal_dino.py 存在")
    sys.exit(1)

def create_universal_checkpoint():
    # 1. 配置
    dino_path = 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth'
    save_path = 'Universal_DINOv3_ViT-B.pth'
    
    if not os.path.exists(dino_path):
        print(f"错误: 找不到 DINO 权重 {dino_path}")
        return

    # 2. 实例化全能模型
    print("正在实例化全能模型...")
    # 这里创建一个 dummy args，只要能跑通 __init__ 就行
    args = argparse.Namespace()
    args.chunk_size = 144 # ENLCN 需要的参数
    
    model = UniversalDino(args)
    model_dict = model.state_dict()
    
    # 3. 加载 DINO 权重
    print("正在加载 DINOv3 原始权重...")
    dino_state = torch.load(dino_path, map_location='cpu')
    
    # 4. 智能权重分发
    new_state = {}
    print("开始分发权重...")
    
    # 辅助函数：处理 PatchEmbed 权重
    def convert_patch_embed(weight, target_ch):
        # weight: [768, 3, 16, 16]
        # 1. 取 RGB 平均 -> [768, 1, 16, 16]
        avg = weight.mean(dim=1, keepdim=True)
        # 2. 复制 -> [768, target_ch, 16, 16]
        return avg.repeat(1, target_ch, 1, 1)

    for k, v in dino_state.items():
        # === A. 共享骨干 (直接复制) ===
        # DINO 里的 key 可能是 blocks.0.xxx 或 norm.xxx
        # 我们模型里是 backbone_model.blocks.0.xxx
        # 需要加前缀 'backbone_model.'
        target_key = 'backbone_model.' + k
        
        if target_key in model_dict:
            if model_dict[target_key].shape == v.shape:
                new_state[target_key] = v
            else:
                print(f"  [Skip] Shape mismatch: {target_key}")
                
        # === B. 处理各个头 (PatchEmbed) ===
        if 'patch_embed.proj.weight' in k:
            dino_pe_weight = v
            
            # 1. SR 头 (1通道)
            if 'heads.sr.proj.weight' in model_dict:
                new_state['heads.sr.proj.weight'] = convert_patch_embed(dino_pe_weight, 1)
                new_state['heads.sr.proj.bias'] = dino_state['patch_embed.proj.bias'] # Bias直接复制
                print("  -> 初始化 SR 头 (1ch)")

            # 2. Iso 头 (1通道)
            if 'heads.iso.proj.weight' in model_dict:
                new_state['heads.iso.proj.weight'] = convert_patch_embed(dino_pe_weight, 1)
                new_state['heads.iso.proj.bias'] = dino_state['patch_embed.proj.bias']
                print("  -> 初始化 Iso 头 (1ch)")
                
            # 3. Proj 头 (1通道)
            if 'heads.proj.proj.weight' in model_dict:
                new_state['heads.proj.proj.weight'] = convert_patch_embed(dino_pe_weight, 1)
                new_state['heads.proj.proj.bias'] = dino_state['patch_embed.proj.bias']
                print("  -> 初始化 Proj 头 (1ch)")

            # 4. Denoise 头 (5通道)
            if 'heads.denoise.proj.weight' in model_dict:
                new_state['heads.denoise.proj.weight'] = convert_patch_embed(dino_pe_weight, 5)
                # Bias 不用变，因为它是加在输出通道(768)上的，跟输入通道无关
                new_state['heads.denoise.proj.bias'] = dino_state['patch_embed.proj.bias']
                print("  -> 初始化 Denoise 头 (5ch)")
                
            # 5. VCD 头 (768通道)
            # VCD 输入已经是 768 通道，DINO 权重是 3 通道，差异太大，不建议强行初始化
            # 让它保持随机初始化即可
            print("  -> VCD 头保持随机初始化")

    # 5. 加载并保存
    print(f"共匹配了 {len(new_state)} 个参数张量。")
    msg = model.load_state_dict(new_state, strict=False)
    print("未加载的键 (主要是尾部和UNet等):", len(msg.missing_keys), "个")
    
    torch.save(model.state_dict(), save_path)
    print(f"\n成功！全能预训练权重已保存至: {save_path}")
    print("以后微调任何任务，都只需要加载这个文件！")

if __name__ == '__main__':
    create_universal_checkpoint()