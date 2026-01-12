"""
DINOv3-based Denoising 微调脚本

用于训练基于 DINOv3 backbone 的 3D 去噪模型。
支持 Planaria 和 Tribolium 数据集。
"""
import torch
torch.backends.cudnn.enabled = False
import utility
import loss
import argparse
from div2k import Flourescenedenoise
from trainer import Trainer
from torch.utils.data import dataloader
import model
import os
import wandb


def options():
    parser = argparse.ArgumentParser(description='DINOv3 Denoising')
    parser.add_argument('--model', default=modelname, help='model name')
    parser.add_argument('--test_only', action='store_true', default=test_only,
                        help='set this option to test the model')
    
    scale = 1  # Denoise 任务 scale=1
    parser.add_argument('--modelpath', type=str, default=modelpath, help='预训练模型路径')
    parser.add_argument('--resume', type=int, default=resume, help='-2:best;-1:latest.ptb; 0:pretrain; >0: resume')
    parser.add_argument('--save', type=str, default=savepath, help='保存路径')
    parser.add_argument('--pre_train', type=str, default=modelpath)
    parser.add_argument('--prune', action='store_true', help='prune layers')

    # Data specifications
    parser.add_argument('--data_test', type=str, default=testset, help='demo image directory')
    parser.add_argument('--epochs', type=int, default=epoch, help='number of epochs to train')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGBn_colors')
    parser.add_argument('--n_colors', type=int, default=1, help='')
    parser.add_argument('--inputchannel', type=int, default=inputchannel, help='输入通道数')
    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)
    parser.add_argument('--condition', type=int, default=condition)

    parser.add_argument('--batch_size', type=int, default=batchsize, help='input batch size for training')
    parser.add_argument('--patch_size', type=int, default=patch_size, help='input batch size for training')
    parser.add_argument('--cpu', action='store_true', default=False, help='')
    parser.add_argument('--print_every', type=int, default=400)
    parser.add_argument('--test_every', type=int, default=30000)
    
    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
    
    parser.add_argument('--chop', action='store_true', default=True, help='enable memory-efficient forward')
    parser.add_argument('--load', type=str, default='', help='file name to load')
    parser.add_argument('--debug', action='store_true', help='Enables debug mode')
    parser.add_argument('--scale', type=str, default='%d' % scale, help='super resolution scale')
    parser.add_argument('--chunk_size', type=int, default=144, help='attention bucket size')
    parser.add_argument('--n_hashes', type=int, default=4, help='number of hash rounds')
    
    # Model specifications
    parser.add_argument('--extend', type=str, default='.', help='pre-trained model directory')
    parser.add_argument('--shift_mean', default=True, help='subtract pixel mean from the input')
    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'), 
                        help='FP precision for test (single | half)')

    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--local_rank', type=int, default=0)
    
    # 冻结 backbone 选项
    # ========== 关键修改 3: 默认开启部分冻结 ==========
    parser.add_argument('--freeze_backbone', action='store_true', default=True,
                        help='Freeze the DINOv3 backbone (blocks) and only train the head/tail.')

    # Hardware specifications
    parser.add_argument('--n_threads', type=int, default=0, help='number of threads for data loading')
    # Training specifications
    parser.add_argument('--reset', action='store_true', help='reset the training')
    parser.add_argument('--split_batch', type=int, default=1, help='split the batch into smaller chunks')
    parser.add_argument('--self_ensemble', action='store_true', help='use self-ensemble method for test')
    
    # Optimization specifications
    parser.add_argument('--lr', type=float, default=lr, help='learning rate')
    parser.add_argument('--decay', type=str, default='cosine', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
    parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'),
                        help='optimizer to use (SGD | ADAM | RMSprop)')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    # ========== 关键修改 2: 开启梯度裁剪 ==========
    parser.add_argument('--gclip', type=float, default=1.0, help='gradient clipping threshold (0 = no clipping)')
    
    # Loss specifications
    # ========== 关键修改 5: 使用混合损失函数 ==========
    # L1 + SSIM 组合可以提供更稳定的训练
    parser.add_argument('--loss', type=str, default='1*L1+0.1*SSIM', help='loss function configuration')
    parser.add_argument('--skip_threshold', type=float, default='1e8', help='skipping batch that has large error')
    
    # Log specifications
    parser.add_argument('--save_models', action='store_true', default=True, help='save all intermediate models')
    parser.add_argument('--save_results', action='store_true', default=True, help='save output results')
    parser.add_argument('--patience', type=int, default=5000, help='Early stopping patience')
    
    parser.add_argument('--wandb_id', type=str, default=None, help='wandb run id to resume')
    
    args = parser.parse_args()
    
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
    
    return args


def main():
    _model = model.Model(args, checkpoint)

    if args.prune:
        prune_layers = 1
        print(f"Pruning {prune_layers} layers...")
        # 注意：DINOv3 使用 'blocks' 而不是 'layers'
        if hasattr(_model.model, 'blocks'):
            del _model.model.blocks[prune_layers]
        elif hasattr(_model.model, 'layers'):
            del _model.model.layers[prune_layers]

    if not args.test_only:
        loader_train = dataloader.DataLoader(
            Flourescenedenoise(args, istrain=True),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=not args.cpu,
            num_workers=4,
        )
    else:
        loader_train = None
    
    loader_test = [dataloader.DataLoader(
        Flourescenedenoise(args, istrain=False, c=condition),
        batch_size=1,
        shuffle=False,
        pin_memory=not args.cpu,
        num_workers=args.n_threads,
    )]
    
    _loss = loss.Loss(args, checkpoint) if not args.test_only else None
    t = Trainer(args, loader_train, loader_test, args.data_test, _model, _loss, checkpoint)
    
    if test_only:
        t.test3DdenoiseInchannel5(condition=condition)
    else:
        while t.terminate():
            t.train()
    
    if hasattr(t, 'done'):
        t.done()
    checkpoint.done()

    
if __name__ == '__main__':
    test_only = False
    normrange = 'Norm_0-100'
    
    # 选择数据集: 'Denoising_Planaria' 或 'Denoising_Tribolium'
    testsetlst = ['Denoising_Planaria']  # ['Denoising_Tribolium']
    
    # 根据数据集调整参数
    if testsetlst[0] == 'Denoising_Planaria':
        modelname = 'DINOIRv3'  # 改为 DINOv3
        inputchannel = 1  # Planaria 是单通道输入
        resume = 0  # 从预训练开始
    else:
        modelname = 'DINOIRv3mto1'  # Tribolium 多通道输入
        inputchannel = 5
        resume = 0

    # 训练超参数
    batchsize = 8  # Denoise 任务通常 batch 较小
    patch_size = 64
    epoch = 500
    # ========== 关键修改 1: 降低学习率 ==========
    # DINOv3 预训练模型微调建议使用较小的学习率
    lr = 2e-5  # 原来 1e-4 太大，导致震荡
    datamin, datamax = 0, 100
    
    # 预训练权重路径
    initial_weights_path = './dinoir_v3_vitb_unipreload.pth'
    
    for condition in range(1, 4):  # 遍历不同的 condition
        for testset in testsetlst:
            savepath = '%s%s/' % (modelname, testset)
            modelpath = initial_weights_path
            
            args = options()
            
            # ========== 关键修改 4: 正确设置通道数 ==========
            # 数据本身是单通道，但模型需要三通道输入（预训练要求）
            # args.inputchannel 用于数据集加载，应与实际数据匹配
            # 模型会在 forward 中自动复制单通道到三通道
            if testsetlst[0] == 'Denoising_Planaria':
                args.inputchannel = 1  # Planaria 数据是单通道
            else:
                args.inputchannel = 5  # Tribolium 是多通道
            
            # 模型内部通道设置（模型会自动处理 1->3 的转换）
            args.inch = 3
            args.n_colors = 3
            
            # 覆盖参数
            args.modelpath = modelpath
            args.resume = resume
            args.test_only = test_only
            args.epochs = epoch
            args.lr = lr
            args.batch_size = batchsize
            args.patch_size = patch_size
            args.save = savepath
            args.condition = condition
            
            print(f"\n{'='*60}")
            print(f"🚀 开始训练: {testset}, Condition={condition}")
            print(f"   模型: {modelname}")
            print(f"   保存路径: {savepath}")
            print(f"{'='*60}\n")
            
            torch.manual_seed(args.seed)
            checkpoint = utility.checkpoint(args)
            main()
