"""
批量测试 model_*.pt 文件，找到 PSNR/SSIM 最优的 epoch

用法:
    python batch_test_models.py --exp_dir experiment/DINOIRv3F-actin-frozen --start 11 --end 200

或指定 GPU:
    CUDA_VISIBLE_DEVICES=1 python batch_test_models.py --exp_dir experiment/DINOIRv3F-actin-frozen
"""
import os
import sys
import glob
import argparse
import torch
import numpy as np
from tqdm import tqdm

# 复用现有模块
import utility
import model
from div2k import DIV2K, normalize
from torch.utils.data import DataLoader

try:
    from skimage.metrics import structural_similarity as compare_ssim
    from skimage.metrics import peak_signal_noise_ratio as compare_psnr
except:
    from skimage.measure import compare_psnr, compare_ssim


def compute_psnr_ssim(sr, hr, data_range=255):
    """计算 PSNR 和 SSIM"""
    psnr = compare_psnr(hr, sr, data_range=data_range)
    ssim = compare_ssim(hr, sr, data_range=data_range)
    return psnr, ssim


def test_single_model(model_obj, loader_test, args, device):
    """测试单个模型，返回平均 PSNR 和 SSIM - 完全复制 trainer.py 的逻辑"""
    model_obj.eval()
    psnr_list = []
    ssim_list = []
    
    with torch.no_grad():
        for lr, hr, filename in loader_test:
            lr = lr.to(device)
            hr = hr.to(device)
            
            # 前向推理
            sr = model_obj(lr, 0)
            
            # 完全复制 trainer.py 的处理流程
            # 1. quantize
            sr = utility.quantize(sr, args.rgb_range)
            hr = utility.quantize(hr, args.rgb_range)
            
            # 2. 第一种 PSNR：utility.calc_psnr (带边界裁剪)
            pst = utility.calc_psnr(sr, hr, args.scale[0], args.rgb_range, dataset=None)
            
            # 3. 转换到 [0, 255] 范围
            sr_np = sr.mul(255 / args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            hr_np = hr.mul(255 / args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            
            # 4. 第二种计算：直接在 [0, 255] 上计算
            ps, ss = utility.compute_psnr_and_ssim(sr_np, hr_np)
            
            # 5. 第三种计算：normalize(x, 0, 100) 后再计算
            sr_255 = np.float32(normalize(sr_np, 0, 100, clip=True)) * 255
            hr_255 = np.float32(normalize(hr_np, 0, 100, clip=True)) * 255
            ps255, ss255 = utility.compute_psnr_and_ssim(sr_255, hr_255)
            
            # 6. 取三种 PSNR 的最大值（与 trainer.py 完全一致）
            psnr_list.append(np.max([ps, pst, ps255]))
            ssim_list.append(ss)  # SSIM 用第二种
    
    return np.mean(psnr_list), np.mean(ssim_list)


def main():
    parser = argparse.ArgumentParser(description='批量测试模型找最优 epoch')
    parser.add_argument('--exp_dir', type=str, required=True, 
                        help='实验目录，如 experiment/DINOIRv3ER-frozen')
    parser.add_argument('--data_test', type=str, default='ER',
                        help='测试数据集名称')
    parser.add_argument('--start', type=int, default=1, help='起始 epoch')
    parser.add_argument('--end', type=int, default=None, help='结束 epoch (None=自动检测)')
    parser.add_argument('--step', type=int, default=1, help='epoch 步长')
    parser.add_argument('--scale', type=int, default=2, help='超分辨率倍数')
    parser.add_argument('--model', type=str, default='DINOIRv3', help='模型名称')
    
    args = parser.parse_args()
    
    # 检测可用的模型文件
    model_dir = os.path.join(args.exp_dir, 'model')
    if not os.path.exists(model_dir):
        print(f"❌ 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    model_files = sorted(glob.glob(os.path.join(model_dir, 'model_*.pt')))
    if not model_files:
        print(f"❌ 未找到 model_*.pt 文件")
        sys.exit(1)
    
    # 解析可用的 epoch 列表
    available_epochs = []
    for f in model_files:
        basename = os.path.basename(f)
        if basename.startswith('model_') and basename.endswith('.pt'):
            try:
                ep = int(basename[6:-3])  # model_XXX.pt -> XXX
                available_epochs.append(ep)
            except ValueError:
                continue
    
    available_epochs.sort()
    print(f"📁 找到 {len(available_epochs)} 个模型文件")
    print(f"   范围: epoch {available_epochs[0]} ~ {available_epochs[-1]}")
    
    # 筛选要测试的 epoch
    start_ep = args.start
    end_ep = args.end if args.end else available_epochs[-1]
    test_epochs = [ep for ep in available_epochs if start_ep <= ep <= end_ep and (ep - start_ep) % args.step == 0]
    
    print(f"🔍 将测试 {len(test_epochs)} 个模型 (epoch {start_ep}~{end_ep}, step={args.step})")
    
    # 构建测试用的 args
    class TestArgs:
        pass
    
    test_args = TestArgs()
    test_args.model = args.model
    test_args.scale = [args.scale]
    test_args.data_test = args.data_test
    test_args.rgb_range = 1
    test_args.n_colors = 1
    test_args.inch = 1
    test_args.inputchannel = 3
    test_args.patch_size = 128
    test_args.cpu = False
    test_args.n_GPUs = 1
    test_args.chop = True
    test_args.precision = 'single'
    test_args.self_ensemble = False
    test_args.save_models = False
    test_args.test_only = True
    test_args.save = args.exp_dir.replace('experiment/', '')
    test_args.modelpath = '.'
    test_args.resume = 0
    test_args.freeze_backbone = False
    test_args.local_rank = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  使用设备: {device}")
    
    # 加载测试数据
    print(f"📊 加载测试集: {args.data_test}")
    test_dataset = DIV2K(test_args, name=args.data_test, train=False, benchmark=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"   测试图片数: {len(test_dataset)}")
    
    # 创建一个 dummy checkpoint 用于初始化模型
    class DummyCheckpoint:
        def __init__(self):
            self.ok = True
            self.log_file = open(os.devnull, 'w')
        def get_path(self, *args):
            return '/tmp'
    
    dummy_ckp = DummyCheckpoint()
    
    # 初始化模型结构（只需一次）
    print(f"🔧 初始化模型结构: {args.model}")
    _model = model.Model(test_args, dummy_ckp)
    
    # 记录结果
    results = []
    
    # 遍历测试每个 epoch
    print(f"\n{'='*60}")
    print("开始批量测试...")
    print(f"{'='*60}\n")
    
    for epoch in tqdm(test_epochs, desc="测试进度"):
        model_path = os.path.join(model_dir, f'model_{epoch}.pt')
        
        try:
            # 加载权重
            state_dict = torch.load(model_path, map_location=device)
            _model.model.load_state_dict(state_dict, strict=True)
            _model.model.to(device)
            
            # 测试
            avg_psnr, avg_ssim = test_single_model(_model, test_loader, test_args, device)
            results.append({
                'epoch': epoch,
                'psnr': avg_psnr,
                'ssim': avg_ssim
            })
            
            tqdm.write(f"  Epoch {epoch:4d}: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
            
        except Exception as e:
            tqdm.write(f"  Epoch {epoch:4d}: ❌ 加载失败 - {e}")
            continue
    
    # 找最优
    if results:
        best_psnr = max(results, key=lambda x: x['psnr'])
        best_ssim = max(results, key=lambda x: x['ssim'])
        
        print(f"\n{'='*60}")
        print("📈 测试完成！结果汇总:")
        print(f"{'='*60}")
        print(f"\n🏆 PSNR 最优: Epoch {best_psnr['epoch']}")
        print(f"   PSNR = {best_psnr['psnr']:.4f}, SSIM = {best_psnr['ssim']:.4f}")
        print(f"   模型: {model_dir}/model_{best_psnr['epoch']}.pt")
        
        print(f"\n🏆 SSIM 最优: Epoch {best_ssim['epoch']}")
        print(f"   PSNR = {best_ssim['psnr']:.4f}, SSIM = {best_ssim['ssim']:.4f}")
        print(f"   模型: {model_dir}/model_{best_ssim['epoch']}.pt")
        
        # 保存结果到文件
        result_file = os.path.join(args.exp_dir, 'batch_test_results.txt')
        with open(result_file, 'w') as f:
            f.write(f"Dataset: {args.data_test}\n")
            f.write(f"Tested epochs: {len(results)}\n")
            f.write(f"\nBest PSNR: Epoch {best_psnr['epoch']} (PSNR={best_psnr['psnr']:.4f}, SSIM={best_psnr['ssim']:.4f})\n")
            f.write(f"Best SSIM: Epoch {best_ssim['epoch']} (PSNR={best_ssim['psnr']:.4f}, SSIM={best_ssim['ssim']:.4f})\n")
            f.write(f"\n{'Epoch':<8}{'PSNR':<12}{'SSIM':<12}\n")
            f.write("-" * 32 + "\n")
            for r in sorted(results, key=lambda x: x['epoch']):
                f.write(f"{r['epoch']:<8}{r['psnr']:<12.4f}{r['ssim']:<12.4f}\n")
        
        print(f"\n💾 详细结果已保存到: {result_file}")
    else:
        print("❌ 没有成功测试任何模型")


if __name__ == '__main__':
    main()
