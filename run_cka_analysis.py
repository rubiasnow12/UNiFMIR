"""
CKA 特征相似性分析脚本。

用法：
    python run_cka_analysis.py --model_path ./experiment/Uni-DINOv3-pretrain-lora/model/model_best.pt \\
                               --model_version v2

功能：
    1. 加载训练好的模型
    2. 分别用 SR 和 Denoise 数据计算每一层 Transformer Block 的特征
    3. 计算跨任务 CKA 相似度矩阵
    4. 绘制热力图

预期结果：
    - 浅层 (Layer 1-4)：高 CKA → 共享纹理/边缘特征（打破了信息孤岛）
    - 深层 (Layer 9-12)：低 CKA → Prompt 引导了任务特异性分化
"""

import torch
import argparse
import os
import sys
import numpy as np
from torch.utils.data import DataLoader

# 项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model.dinoir_v3 import DinoUniModel, DinoUniModelV2
from mydata import SR, Flourescenedenoise
from analysis import CKAAnalyzer
import model


def parse_args():
    parser = argparse.ArgumentParser(description='CKA Feature Similarity Analysis')
    parser.add_argument('--model_path', type=str, default='./experiment/Uni-DINOv3-pretrain-lora/model/',
                        help='训练好的模型路径')
    parser.add_argument('--model_version', type=str, default='v2', choices=['v1', 'v2'],
                        help='模型版本: v1=DinoUniModel, v2=DinoUniModelV2')
    parser.add_argument('--resume', type=int, default=-2, help='-2=best, -1=latest, >0=specific epoch')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='CKA 分析使用的样本数（越多越准，但越慢）')
    parser.add_argument('--sr_dataset', type=str, default='F-actin',
                        help='SR 数据集名称')
    parser.add_argument('--dn_dataset', type=str, default='Denoising_Planaria',
                        help='Denoise 数据集名称')
    parser.add_argument('--output_dir', type=str, default='./experiment/cka_analysis',
                        help='输出目录')
    parser.add_argument('--patch_size', type=int, default=64)

    # --- Baseline 对比 ---
    parser.add_argument('--baseline_path', type=str, default='',
                        help='Baseline 模型路径（用于对比图）')

    return parser.parse_args()


def build_model(version, device):
    """构建模型实例"""
    mock_args = argparse.Namespace()
    mock_args.n_resblocks = 8
    mock_args.n_feats = 32
    mock_args.scale = [1]
    mock_args.inch = 1
    mock_args.n_colors = 1
    mock_args.rgb_range = 1
    mock_args.res_scale = 1.0
    mock_args.dilation = False
    mock_args.chop = True
    mock_args.self_ensemble = False
    mock_args.precision = 'single'
    mock_args.n_GPUs = 1
    mock_args.save_models = False
    mock_args.save = 'cka_analysis'
    mock_args.model = 'Uni-DINOv3'
    mock_args.cpu = not torch.cuda.is_available()
    mock_args.test_only = True
    mock_args.load = ''
    mock_args.resume = 0
    mock_args.pre_train = '.'
    mock_args.template = '.'

    if version == 'v2':
        unimodel = DinoUniModelV2(
            mock_args, embed_dim=384, dino_depth=12, dino_num_heads=6,
            task_embed_dim=64
        )
    else:
        unimodel = DinoUniModel(
            mock_args, embed_dim=384, dino_depth=12, dino_num_heads=6
        )

    return unimodel.to(device), mock_args


def load_model_weights(unimodel, model_path, resume=-2):
    """加载模型权重"""
    if os.path.isdir(model_path):
        if resume == -2:
            # 加载 best
            ckpt_path = os.path.join(model_path, 'model_best.pt')
            if not os.path.exists(ckpt_path):
                ckpt_path = os.path.join(model_path, 'model_latest.pt')
        elif resume == -1:
            ckpt_path = os.path.join(model_path, 'model_latest.pt')
        else:
            ckpt_path = os.path.join(model_path, f'model_{resume}.pt')
    else:
        ckpt_path = model_path

    if os.path.exists(ckpt_path):
        print(f"Loading model weights from {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location='cpu')
        # 处理可能的 model.model. 前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.model.', '').replace('model.', '')
            new_state_dict[new_key] = v

        model_state = unimodel.state_dict()
        filtered = {k: v for k, v in new_state_dict.items()
                    if k in model_state and v.shape == model_state[k].shape}
        unimodel.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)}/{len(new_state_dict)} keys")
    else:
        print(f"Warning: {ckpt_path} not found!")

    return unimodel


def get_data_loaders(sr_dataset, dn_dataset, patch_size):
    """准备 SR 和 Denoise 数据集的 DataLoader"""
    srdatapath = './CSB/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/'
    denoisedatapath = './CSB/DataSet/'

    sr_loader = DataLoader(
        SR(scale=2, name=sr_dataset, train=False, test_only=True,
           rootdatapath=srdatapath, patch_size=patch_size, length=20),
        batch_size=1, shuffle=False, num_workers=0
    )

    dn_loader = DataLoader(
        Flourescenedenoise(name=dn_dataset, istrain=False, c=1,
                           rootdatapath=denoisedatapath, test_only=True,
                           patch_size=patch_size, length=2000),
        batch_size=1, shuffle=False, num_workers=0
    )

    return sr_loader, dn_loader


class ModelWrapper:
    """简单包装器，让 CKAAnalyzer 可以统一调用"""
    def __init__(self, unimodel, device):
        self.model = unimodel
        self.device = device
        self.training = unimodel.training

    def eval(self):
        self.model.eval()
        self.training = False

    def train(self):
        self.model.train()
        self.training = True

    def __call__(self, x, tsk):
        return self.model(x, tsk)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    cka_analyzer = CKAAnalyzer(save_dir=args.output_dir)

    # ======== 构建并加载 Ours 模型 ========
    print("\n" + "="*60)
    print(f"加载 Ours 模型 ({args.model_version})...")
    print("="*60)
    unimodel, mock_args = build_model(args.model_version, device)
    unimodel = load_model_weights(unimodel, args.model_path, args.resume)
    unimodel.eval()
    wrapper = ModelWrapper(unimodel, device)

    # ======== 准备数据 ========
    print("\n加载数据集...")
    sr_loader, dn_loader = get_data_loaders(args.sr_dataset, args.dn_dataset, args.patch_size)
    print(f"  SR ({args.sr_dataset}): {len(sr_loader)} batches")
    print(f"  Denoise ({args.dn_dataset}): {len(dn_loader)} batches")

    # ======== 提取特征 ========
    print("\n" + "="*60)
    print("提取 SR 任务特征...")
    print("="*60)
    features_sr = cka_analyzer.extract_layer_features(
        wrapper, sr_loader, task_id=1,
        num_samples=args.num_samples, device=device
    )
    print(f"  提取了 {len(features_sr)} 层的特征")
    for k, v in features_sr.items():
        print(f"    Block {k}: {v.shape}")

    print("\n提取 Denoise 任务特征...")
    features_dn = cka_analyzer.extract_layer_features(
        wrapper, dn_loader, task_id=2,
        num_samples=args.num_samples, device=device
    )
    print(f"  提取了 {len(features_dn)} 层的特征")

    # ======== 计算 CKA 矩阵 ========
    print("\n" + "="*60)
    print("计算 CKA 相似度矩阵...")
    print("="*60)
    cka_matrix = cka_analyzer.compute_cross_task_cka(features_sr, features_dn)
    print(f"  CKA 矩阵形状: {cka_matrix.shape}")

    # 打印对角线值
    diag = np.diag(cka_matrix)
    print("\n  同层跨任务 CKA (对角线):")
    for i, val in enumerate(diag):
        marker = "🟢" if val > 0.5 else "🟡" if val > 0.3 else "🔴"
        print(f"    {marker} Block {i:2d}: {val:.4f}")

    print(f"\n  浅层平均 (0-3): {diag[:4].mean():.4f}")
    print(f"  中层平均 (4-7): {diag[4:8].mean():.4f}")
    print(f"  深层平均 (8-11): {diag[8:].mean():.4f}")

    # ======== 保存结果和绘图 ========
    cka_analyzer.save_cka_matrix(cka_matrix, 'SR', 'Denoise',
                                 f'cka_ours_{args.model_version}.npz')

    output_path = os.path.join(args.output_dir,
                               f'cka_heatmap_ours_{args.model_version}.png')
    CKAAnalyzer.plot_cka_heatmap(
        cka_matrix, task1_name='SR', task2_name='Denoise',
        output_path=output_path,
        title_suffix=f'(Ours: {args.model_version.upper()} Task-Prompted FiLM)'
    )

    # ======== (可选) Baseline 对比 ========
    if args.baseline_path and os.path.exists(args.baseline_path):
        print("\n" + "="*60)
        print("加载 Baseline 模型 (v1)...")
        print("="*60)
        baseline_model, _ = build_model('v1', device)
        baseline_model = load_model_weights(baseline_model, args.baseline_path, args.resume)
        baseline_model.eval()
        baseline_wrapper = ModelWrapper(baseline_model, device)

        features_sr_base = cka_analyzer.extract_layer_features(
            baseline_wrapper, sr_loader, task_id=1,
            num_samples=args.num_samples, device=device
        )
        features_dn_base = cka_analyzer.extract_layer_features(
            baseline_wrapper, dn_loader, task_id=2,
            num_samples=args.num_samples, device=device
        )
        cka_baseline = cka_analyzer.compute_cross_task_cka(features_sr_base, features_dn_base)

        comp_path = os.path.join(args.output_dir, 'cka_comparison_baseline_vs_ours.png')
        CKAAnalyzer.plot_comparison(
            cka_baseline, cka_matrix,
            task1_name='SR', task2_name='Denoise',
            output_path=comp_path
        )

    print("\n" + "="*60)
    print("✅ CKA 分析完成!")
    print("="*60)
    print(f"\n📁 结果保存在: {args.output_dir}/")
    print("📊 如需生成 Baseline 对比图，请指定 --baseline_path 参数")


if __name__ == '__main__':
    main()
