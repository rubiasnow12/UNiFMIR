"""
分析工具模块：用于证明 Task-Prompted Manifold Alignment 机制的有效性。

包含两个核心分析方案：
1. 梯度余弦相似度分析 (Gradient Cosine Similarity Analysis)
   - 证明 FiLM 调制缓解了多任务梯度冲突
2. CKA 特征相似性热力图 (Centered Kernel Alignment)
   - 证明模型"该共享的共享，该区分的区分"

数学基础：
- 梯度余弦相似度: Sim = (g_sr · g_dn) / (||g_sr|| · ||g_dn||)
- CKA: CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))
"""

import torch
import torch.nn as nn
import numpy as np
import os
import json
from collections import defaultdict


# ============================================================
# 方案一：梯度余弦相似度分析
# ============================================================
class GradientConflictAnalyzer:
    """
    梯度余弦相似度分析器。

    在训练过程中，分别计算 Backbone 参数对于不同任务 Loss 的梯度，
    然后计算梯度向量之间的余弦相似度。

    - Sim ≈ -1: 梯度方向完全相反（严重冲突）
    - Sim ≈  0: 梯度方向正交（互不干扰）
    - Sim ≈ +1: 梯度方向一致（协同）

    预期结果：
    - Baseline (Multi-Head): 负值震荡 → 梯度冲突
    - Ours (FiLM Prompt):    向 0 或正值偏移 → 冲突缓解
    """

    def __init__(self, save_dir='./experiment/gradient_analysis'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.history = defaultdict(list)  # {layer_name: [(step, sim_value), ...]}
        self.step_count = 0

    def compute_task_gradients(self, model, data_task1, data_task2, loss_fn,
                               task1_id=1, task2_id=2, device='cuda'):
        """
        计算两个任务在共享 Backbone 上的梯度向量。

        Args:
            model: DinoUniModelV2 (或被 model.Model 包装的)
            data_task1: (lr, hr) 任务1的数据 batch
            data_task2: (lr, hr) 任务2的数据 batch
            loss_fn: 损失函数
            task1_id: 任务1 ID (1=SR, 2=Denoise, ...)
            task2_id: 任务2 ID
            device: 计算设备

        Returns:
            dict: {layer_group: cosine_similarity}
        """
        # 获取实际的模型（解包 model.Model 包装）
        actual_model = model.model if hasattr(model, 'model') else model

        # --- 计算任务1的梯度 ---
        model.zero_grad()
        lr1, hr1 = data_task1
        lr1, hr1 = lr1.to(device), hr1.to(device)
        out1 = model(lr1, task1_id)
        if isinstance(out1, tuple):
            out1 = out1[-1]  # 取最终输出
        loss1 = loss_fn(out1, hr1)
        loss1.backward(retain_graph=False)

        # 收集任务1的梯度
        grads_task1 = {}
        for name, param in actual_model.named_parameters():
            if param.grad is not None and self._is_backbone_param(name):
                group = self._get_layer_group(name)
                if group not in grads_task1:
                    grads_task1[group] = []
                grads_task1[group].append(param.grad.detach().clone().flatten())

        # --- 计算任务2的梯度 ---
        model.zero_grad()
        lr2, hr2 = data_task2
        lr2, hr2 = lr2.to(device), hr2.to(device)
        out2 = model(lr2, task2_id)
        if isinstance(out2, tuple):
            out2 = out2[-1]
        loss2 = loss_fn(out2, hr2)
        loss2.backward(retain_graph=False)

        # 收集任务2的梯度
        grads_task2 = {}
        for name, param in actual_model.named_parameters():
            if param.grad is not None and self._is_backbone_param(name):
                group = self._get_layer_group(name)
                if group not in grads_task2:
                    grads_task2[group] = []
                grads_task2[group].append(param.grad.detach().clone().flatten())

        model.zero_grad()

        # --- 计算每个层组的余弦相似度 ---
        similarities = {}
        for group in grads_task1:
            if group in grads_task2:
                g1 = torch.cat(grads_task1[group])
                g2 = torch.cat(grads_task2[group])
                cos_sim = torch.nn.functional.cosine_similarity(
                    g1.unsqueeze(0), g2.unsqueeze(0)
                ).item()
                similarities[group] = cos_sim

        # 计算整体 Backbone 的余弦相似度
        all_g1 = torch.cat([torch.cat(grads_task1[g]) for g in sorted(grads_task1)])
        all_g2 = torch.cat([torch.cat(grads_task2[g]) for g in sorted(grads_task2)])
        similarities['backbone_all'] = torch.nn.functional.cosine_similarity(
            all_g1.unsqueeze(0), all_g2.unsqueeze(0)
        ).item()

        return similarities

    def log_step(self, similarities, step=None, task_pair='sr_vs_dn'):
        """记录一个 step 的梯度相似度"""
        if step is None:
            step = self.step_count
        self.step_count += 1

        for layer, sim in similarities.items():
            key = f"{task_pair}/{layer}"
            self.history[key].append((step, sim))

        return similarities

    def save_results(self, filename='gradient_cosine_similarity.json'):
        """保存所有记录到 JSON 文件"""
        # 转换为可序列化格式
        data = {}
        for key, vals in self.history.items():
            data[key] = {'steps': [v[0] for v in vals],
                         'similarities': [v[1] for v in vals]}

        path = os.path.join(self.save_dir, filename)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"梯度分析结果已保存到 {path}")
        return path

    def _is_backbone_param(self, name):
        """判断是否是共享 Backbone 的参数"""
        backbone_keywords = ['blocks.', 'norm.', 'conv_after_body.', 'patch_embed.']
        return any(kw in name for kw in backbone_keywords)

    def _get_layer_group(self, name):
        """将参数名映射到层组名"""
        if 'blocks.' in name:
            # 提取 block 编号
            parts = name.split('.')
            try:
                idx = int(parts[parts.index('blocks') + 1])
                if idx < 4:
                    return 'blocks_shallow(0-3)'
                elif idx < 8:
                    return 'blocks_middle(4-7)'
                else:
                    return 'blocks_deep(8-11)'
            except (ValueError, IndexError):
                return 'blocks_other'
        elif 'norm.' in name:
            return 'norm'
        elif 'conv_after_body' in name:
            return 'conv_after_body'
        elif 'patch_embed' in name:
            return 'patch_embed'
        return 'other'

    @staticmethod
    def plot_gradient_similarity(json_path, output_path=None):
        """
        绘制梯度余弦相似度曲线图。

        Args:
            json_path: save_results() 保存的 JSON 文件路径
            output_path: 输出图片路径。如为 None，自动生成。
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        with open(json_path, 'r') as f:
            data = json.load(f)

        if output_path is None:
            output_path = json_path.replace('.json', '.png')

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- 左图：逐层梯度相似度随训练步数变化 ---
        ax = axes[0]
        layer_colors = {
            'blocks_shallow(0-3)': '#2196F3',
            'blocks_middle(4-7)': '#FF9800',
            'blocks_deep(8-11)': '#F44336',
            'backbone_all': '#4CAF50',
        }
        for key, vals in data.items():
            layer = key.split('/')[-1]
            if layer in layer_colors:
                steps = vals['steps']
                sims = vals['similarities']
                # 平滑处理
                if len(sims) > 10:
                    window = min(20, len(sims) // 5)
                    sims_smooth = np.convolve(sims, np.ones(window)/window, mode='valid')
                    steps_smooth = steps[:len(sims_smooth)]
                else:
                    sims_smooth = sims
                    steps_smooth = steps
                ax.plot(steps_smooth, sims_smooth, label=layer,
                        color=layer_colors[layer], linewidth=2, alpha=0.8)

        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Orthogonal (0)')
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Gradient Cosine Similarity', fontsize=12)
        ax.set_title('Gradient Conflict Analysis\n(Ours: Task-Prompted FiLM)', fontsize=14)
        ax.legend(fontsize=10)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, alpha=0.3)

        # --- 右图：不同层的平均梯度相似度箱线图 ---
        ax2 = axes[1]
        box_data = []
        box_labels = []
        for key, vals in data.items():
            layer = key.split('/')[-1]
            if layer in layer_colors:
                box_data.append(vals['similarities'])
                box_labels.append(layer.replace('blocks_', 'B:'))

        if box_data:
            bp = ax2.boxplot(box_data, labels=box_labels, patch_artist=True)
            colors_list = list(layer_colors.values())[:len(box_data)]
            for patch, color in zip(bp['boxes'], colors_list):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.set_ylabel('Gradient Cosine Similarity', fontsize=12)
            ax2.set_title('Per-Layer Gradient Similarity Distribution', fontsize=14)
            ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"梯度分析图已保存到 {output_path}")
        return output_path


# ============================================================
# 方案二：CKA 特征相似性分析
# ============================================================
class CKAAnalyzer:
    """
    CKA (Centered Kernel Alignment) 特征相似性分析器。

    比较模型在处理不同任务时，每一层 Transformer Block 输出的特征分布相似度。

    CKA(K, L) = HSIC(K, L) / sqrt(HSIC(K, K) · HSIC(L, L))

    其中 HSIC 是 Hilbert-Schmidt 独立准则。

    预期结果（热力图）：
    - 浅层 (1-4): 高相似度 → 共享纹理特征
    - 深层 (9-12): 低相似度 → Prompt 引导任务分化
    """

    def __init__(self, save_dir='./experiment/cka_analysis'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.hooks = []
        self.features = {}

    @staticmethod
    def _centering_matrix(n, device):
        """生成中心化矩阵 H = I - (1/n) * 11^T"""
        H = torch.eye(n, device=device) - torch.ones(n, n, device=device) / n
        return H

    @staticmethod
    def _hsic(K, L):
        """
        计算 HSIC (Hilbert-Schmidt Independence Criterion)
        HSIC(K, L) = (1/(n-1)^2) * trace(K H L H)
        """
        n = K.shape[0]
        H = CKAAnalyzer._centering_matrix(n, K.device)
        KH = K @ H
        LH = L @ H
        return (KH * LH.T).sum() / ((n - 1) ** 2)

    @staticmethod
    def linear_cka(X, Y):
        """
        计算线性 CKA。

        Args:
            X: [n, d1] 特征矩阵（n 个样本，d1 维特征）
            Y: [n, d2] 特征矩阵

        Returns:
            float: CKA 相似度值 ∈ [0, 1]
        """
        # 线性核: K = X X^T, L = Y Y^T
        K = X @ X.T
        L = Y @ Y.T

        hsic_kl = CKAAnalyzer._hsic(K, L)
        hsic_kk = CKAAnalyzer._hsic(K, K)
        hsic_ll = CKAAnalyzer._hsic(L, L)

        denom = torch.sqrt(hsic_kk * hsic_ll)
        if denom < 1e-10:
            return 0.0
        return (hsic_kl / denom).item()

    def extract_layer_features(self, model, data_loader, task_id, num_samples=200, device='cuda'):
        """
        提取模型在指定任务下，每一层 Transformer Block 的输出特征。

        Args:
            model: 模型 (model.Model 包装的)
            data_loader: 数据加载器
            task_id: 任务 ID
            num_samples: 使用的样本数量
            device: 设备

        Returns:
            dict: {layer_idx: features_tensor [N, D]}
        """
        actual_model = model.model if hasattr(model, 'model') else model
        model.eval()

        layer_features = defaultdict(list)

        # 注册 hook 到每个 Transformer Block
        hooks = []
        for idx, block in enumerate(actual_model.blocks):
            def make_hook(layer_idx):
                def hook_fn(module, input, output):
                    # output: [B, N, C] token 序列
                    if isinstance(output, torch.Tensor):
                        # 对 token 维度取平均，得到 [B, C]
                        feat = output.mean(dim=1).detach().cpu()
                        layer_features[layer_idx].append(feat)
                return hook_fn
            h = block.register_forward_hook(make_hook(idx))
            hooks.append(h)

        # 前向传播收集特征
        sample_count = 0
        with torch.no_grad():
            for batch_data in data_loader:
                if sample_count >= num_samples:
                    break
                lr_data, hr_data = batch_data[0], batch_data[1]
                lr_data = lr_data.to(device)
                _ = model(lr_data, task_id)
                sample_count += lr_data.shape[0]

        # 移除 hooks
        for h in hooks:
            h.remove()

        # 合并特征: {layer_idx: [N, C]}
        result = {}
        for idx in layer_features:
            result[idx] = torch.cat(layer_features[idx], dim=0)[:num_samples]

        model.train()
        return result

    def compute_cross_task_cka(self, features_task1, features_task2):
        """
        计算跨任务的 CKA 矩阵。

        Args:
            features_task1: {layer_idx: [N, D]} 任务1的各层特征
            features_task2: {layer_idx: [N, D]} 任务2的各层特征

        Returns:
            np.ndarray: [num_layers, num_layers] CKA 矩阵
        """
        layers1 = sorted(features_task1.keys())
        layers2 = sorted(features_task2.keys())
        n_layers = max(len(layers1), len(layers2))

        cka_matrix = np.zeros((n_layers, n_layers))

        for i, l1 in enumerate(layers1):
            for j, l2 in enumerate(layers2):
                X = features_task1[l1].float()
                Y = features_task2[l2].float()
                # 确保样本数一致
                n = min(X.shape[0], Y.shape[0])
                X, Y = X[:n], Y[:n]
                cka_matrix[i, j] = self.linear_cka(X, Y)

        return cka_matrix

    def compute_same_layer_cka(self, features_task1, features_task2):
        """
        计算同一层在不同任务下的 CKA 相似度（对角线值）。

        Returns:
            dict: {layer_idx: cka_value}
        """
        result = {}
        layers = sorted(set(features_task1.keys()) & set(features_task2.keys()))
        for l in layers:
            X = features_task1[l].float()
            Y = features_task2[l].float()
            n = min(X.shape[0], Y.shape[0])
            result[l] = self.linear_cka(X[:n], Y[:n])
        return result

    def save_cka_matrix(self, cka_matrix, task1_name='SR', task2_name='Denoise',
                        filename='cka_matrix.npz'):
        """保存 CKA 矩阵"""
        path = os.path.join(self.save_dir, filename)
        np.savez(path, cka_matrix=cka_matrix,
                 task1=task1_name, task2=task2_name)
        print(f"CKA 矩阵已保存到 {path}")
        return path

    @staticmethod
    def plot_cka_heatmap(cka_matrix, task1_name='SR', task2_name='Denoise',
                         output_path=None, title_suffix='(Ours: Task-Prompted FiLM)'):
        """
        绘制 CKA 特征相似性热力图。

        Args:
            cka_matrix: [L, L] CKA 相似度矩阵
            task1_name: 任务1名称（X 轴）
            task2_name: 任务2名称（Y 轴）
            output_path: 输出图片路径
            title_suffix: 标题后缀
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if output_path is None:
            output_path = f'./experiment/cka_analysis/cka_{task1_name}_vs_{task2_name}.png'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        n_layers = cka_matrix.shape[0]
        layer_labels = [f'L{i+1}' for i in range(n_layers)]

        fig, axes = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={'width_ratios': [1.2, 1]})

        # --- 左图：完整 CKA 热力图 ---
        ax = axes[0]
        im = ax.imshow(cka_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(n_layers))
        ax.set_yticks(range(n_layers))
        ax.set_xticklabels(layer_labels, fontsize=9)
        ax.set_yticklabels(layer_labels, fontsize=9)
        ax.set_xlabel(f'{task1_name} Layers', fontsize=13)
        ax.set_ylabel(f'{task2_name} Layers', fontsize=13)
        ax.set_title(f'CKA Feature Similarity\n{task1_name} vs {task2_name} {title_suffix}', fontsize=14)

        # 标注数值
        for i in range(n_layers):
            for j in range(n_layers):
                val = cka_matrix[i, j]
                color = 'white' if val > 0.6 or val < 0.2 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)

        plt.colorbar(im, ax=ax, shrink=0.8, label='CKA Similarity')

        # 添加区域标注框
        # 浅层区域 (0-3)
        rect1 = plt.Rectangle((-0.5, -0.5), 4, 4, linewidth=2,
                               edgecolor='#2196F3', facecolor='none', linestyle='--')
        ax.add_patch(rect1)
        ax.text(1.5, -1.2, 'Shared\n(Shallow)', ha='center', fontsize=9,
                color='#2196F3', fontweight='bold')

        # 深层区域 (8-11)
        if n_layers >= 12:
            rect2 = plt.Rectangle((7.5, 7.5), 4, 4, linewidth=2,
                                   edgecolor='#F44336', facecolor='none', linestyle='--')
            ax.add_patch(rect2)
            ax.text(9.5, 12.5, 'Differentiated\n(Deep)', ha='center', fontsize=9,
                    color='#F44336', fontweight='bold')

        # --- 右图：对角线 CKA 曲线 (同层跨任务相似度) ---
        ax2 = axes[1]
        diag = np.diag(cka_matrix)
        x = np.arange(1, n_layers + 1)

        # 绘制柱状图 + 折线图
        colors = ['#2196F3'] * 4 + ['#FF9800'] * 4 + ['#F44336'] * (n_layers - 8)
        ax2.bar(x, diag, color=colors, alpha=0.5, edgecolor='gray', linewidth=0.5)
        ax2.plot(x, diag, 'ko-', linewidth=2, markersize=6, label='Same-layer CKA')

        # 添加区域背景
        ax2.axvspan(0.5, 4.5, alpha=0.08, color='#2196F3', label='Shallow (shared)')
        ax2.axvspan(4.5, 8.5, alpha=0.08, color='#FF9800', label='Middle')
        ax2.axvspan(8.5, n_layers + 0.5, alpha=0.08, color='#F44336',
                    label='Deep (task-specific)')

        ax2.set_xlabel('Transformer Block Layer', fontsize=13)
        ax2.set_ylabel('CKA Similarity (same layer, cross-task)', fontsize=12)
        ax2.set_title(f'Same-Layer Cross-Task Similarity\n'
                      f'{task1_name} vs {task2_name}', fontsize=14)
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_labels, fontsize=9)
        ax2.set_ylim(0, 1.05)
        ax2.legend(fontsize=9, loc='upper right')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"CKA 热力图已保存到 {output_path}")
        return output_path

    @staticmethod
    def plot_comparison(cka_baseline, cka_ours, task1_name='SR', task2_name='Denoise',
                        output_path=None):
        """
        绘制 Baseline vs Ours 对比热力图。

        Args:
            cka_baseline: [L, L] Baseline 的 CKA 矩阵
            cka_ours: [L, L] Ours 的 CKA 矩阵
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if output_path is None:
            output_path = f'./experiment/cka_analysis/cka_comparison_{task1_name}_vs_{task2_name}.png'
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

        n_layers = cka_baseline.shape[0]
        layer_labels = [f'L{i+1}' for i in range(n_layers)]

        fig, axes = plt.subplots(1, 3, figsize=(22, 6),
                                 gridspec_kw={'width_ratios': [1, 1, 0.9]})

        # --- 左图: Baseline ---
        im1 = axes[0].imshow(cka_baseline, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[0].set_title(f'Baseline (Multi-Head)\n{task1_name} vs {task2_name}', fontsize=13)
        axes[0].set_xticks(range(n_layers))
        axes[0].set_yticks(range(n_layers))
        axes[0].set_xticklabels(layer_labels, fontsize=8)
        axes[0].set_yticklabels(layer_labels, fontsize=8)
        axes[0].set_xlabel(f'{task1_name} Layers', fontsize=11)
        axes[0].set_ylabel(f'{task2_name} Layers', fontsize=11)
        for i in range(n_layers):
            for j in range(n_layers):
                val = cka_baseline[i, j]
                color = 'white' if val > 0.6 or val < 0.2 else 'black'
                axes[0].text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=6, color=color)
        plt.colorbar(im1, ax=axes[0], shrink=0.8)

        # --- 中图: Ours ---
        im2 = axes[1].imshow(cka_ours, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[1].set_title(f'Ours (Task-Prompted FiLM)\n{task1_name} vs {task2_name}', fontsize=13)
        axes[1].set_xticks(range(n_layers))
        axes[1].set_yticks(range(n_layers))
        axes[1].set_xticklabels(layer_labels, fontsize=8)
        axes[1].set_yticklabels(layer_labels, fontsize=8)
        axes[1].set_xlabel(f'{task1_name} Layers', fontsize=11)
        axes[1].set_ylabel(f'{task2_name} Layers', fontsize=11)
        for i in range(n_layers):
            for j in range(n_layers):
                val = cka_ours[i, j]
                color = 'white' if val > 0.6 or val < 0.2 else 'black'
                axes[1].text(j, i, f'{val:.2f}', ha='center', va='center',
                             fontsize=6, color=color)
        plt.colorbar(im2, ax=axes[1], shrink=0.8)

        # --- 右图: 对角线对比 ---
        diag_base = np.diag(cka_baseline)
        diag_ours = np.diag(cka_ours)
        x = np.arange(1, n_layers + 1)

        axes[2].plot(x, diag_base, 's--', color='#9E9E9E', linewidth=2,
                     markersize=7, label='Baseline', alpha=0.8)
        axes[2].plot(x, diag_ours, 'o-', color='#E91E63', linewidth=2,
                     markersize=7, label='Ours (FiLM)')
        axes[2].fill_between(x, diag_base, diag_ours,
                             where=(diag_ours > diag_base),
                             alpha=0.15, color='#4CAF50', label='Improvement')
        axes[2].fill_between(x, diag_base, diag_ours,
                             where=(diag_ours <= diag_base),
                             alpha=0.15, color='#2196F3', label='Differentiation')

        axes[2].axvspan(0.5, 4.5, alpha=0.05, color='#2196F3')
        axes[2].axvspan(8.5, n_layers + 0.5, alpha=0.05, color='#F44336')
        axes[2].set_xlabel('Layer', fontsize=12)
        axes[2].set_ylabel('Same-Layer CKA', fontsize=12)
        axes[2].set_title('Diagonal Comparison\n(Same-layer cross-task)', fontsize=13)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(layer_labels, fontsize=8)
        axes[2].set_ylim(0, 1.05)
        axes[2].legend(fontsize=9)
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"CKA 对比图已保存到 {output_path}")
        return output_path


# ============================================================
# 辅助函数：在训练循环中使用
# ============================================================
def compute_gradient_similarity_in_training(model, loss_fn, data_task1, data_task2,
                                             task1_id=1, task2_id=2, device='cuda'):
    """
    在训练过程中快速计算两个任务的梯度余弦相似度。
    轻量版：只计算整体 Backbone 和分层的相似度。

    用法示例 (在训练循环中每 100 步调用一次):
        if step % 100 == 0:
            sim = compute_gradient_similarity_in_training(
                model, loss_fn, (lr_sr, hr_sr), (lr_dn, hr_dn),
                task1_id=1, task2_id=2
            )
            wandb.log({'grad_sim/backbone': sim['backbone_all']})

    Returns:
        dict: {'backbone_all': float, 'shallow': float, 'middle': float, 'deep': float}
    """
    actual_model = model.model if hasattr(model, 'model') else model

    # 存储当前模式
    was_training = model.training
    model.train()

    # --- 任务1梯度 ---
    model.zero_grad()
    lr1, hr1 = data_task1[0].to(device), data_task1[1].to(device)
    out1 = model(lr1, task1_id)
    if isinstance(out1, tuple):
        out1 = out1[-1]
    loss1 = loss_fn(out1, hr1)
    loss1.backward()

    grads1 = {'shallow': [], 'middle': [], 'deep': []}
    for name, param in actual_model.named_parameters():
        if param.grad is not None and 'blocks.' in name:
            parts = name.split('.')
            try:
                idx = int(parts[parts.index('blocks') + 1])
                g = param.grad.detach().clone().flatten()
                if idx < 4:
                    grads1['shallow'].append(g)
                elif idx < 8:
                    grads1['middle'].append(g)
                else:
                    grads1['deep'].append(g)
            except (ValueError, IndexError):
                pass

    # --- 任务2梯度 ---
    model.zero_grad()
    lr2, hr2 = data_task2[0].to(device), data_task2[1].to(device)
    out2 = model(lr2, task2_id)
    if isinstance(out2, tuple):
        out2 = out2[-1]
    loss2 = loss_fn(out2, hr2)
    loss2.backward()

    grads2 = {'shallow': [], 'middle': [], 'deep': []}
    for name, param in actual_model.named_parameters():
        if param.grad is not None and 'blocks.' in name:
            parts = name.split('.')
            try:
                idx = int(parts[parts.index('blocks') + 1])
                g = param.grad.detach().clone().flatten()
                if idx < 4:
                    grads2['shallow'].append(g)
                elif idx < 8:
                    grads2['middle'].append(g)
                else:
                    grads2['deep'].append(g)
            except (ValueError, IndexError):
                pass

    model.zero_grad()

    # 计算余弦相似度
    result = {}
    all_g1, all_g2 = [], []
    for group in ['shallow', 'middle', 'deep']:
        if grads1[group] and grads2[group]:
            g1 = torch.cat(grads1[group])
            g2 = torch.cat(grads2[group])
            cos = torch.nn.functional.cosine_similarity(
                g1.unsqueeze(0), g2.unsqueeze(0)).item()
            result[group] = cos
            all_g1.append(g1)
            all_g2.append(g2)

    if all_g1 and all_g2:
        g1_all = torch.cat(all_g1)
        g2_all = torch.cat(all_g2)
        result['backbone_all'] = torch.nn.functional.cosine_similarity(
            g1_all.unsqueeze(0), g2_all.unsqueeze(0)).item()

    if not was_training:
        model.eval()

    return result
