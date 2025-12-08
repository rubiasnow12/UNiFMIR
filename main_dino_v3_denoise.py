import torch
import utility
import loss
from trainer import Trainer
import os
import argparse
from div2k import Flourescenedenoise
from torch.utils.data import dataloader
import model

# 禁用 cudnn 以避免某些版本的不兼容问题，可视情况开启
torch.backends.cudnn.enabled = False

def options():
    parser = argparse.ArgumentParser(description='DINOv3 Denoising')
    parser.add_argument('--model', default=modelname, help='model name')
    parser.add_argument('--test_only', action='store_true', default=test_only,
                        help='set this option to test the model')
    
    # 核心路径参数
    parser.add_argument('--modelpath', type=str, default=initial_weights_path, help='Pretrained model path')
    parser.add_argument('--save', type=str, default=savepath, help='Save path')
    parser.add_argument('--resume', type=int, default=resume, help='Resume epoch')
    
    # 数据参数
    parser.add_argument('--data_test', type=str, default=testset, help='Dataset name')
    parser.add_argument('--inputchannel', type=int, default=inputchannel, help='Input channels (1 or 5)')
    parser.add_argument('--n_colors', type=int, default=1, help='Output channels')
    parser.add_argument('--rgb_range', type=int, default=1, help='RGB range')
    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)
    parser.add_argument('--condition', type=int, default=condition)
    parser.add_argument('--scale', type=str, default='1', help='Scale factor (1 for denoising)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=batch_size, help='Batch size')
    parser.add_argument('--patch_size', type=int, default=patch_size, help='Patch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'))
    parser.add_argument('--loss', type=str, default='1*L1', help='Loss function')
    
    # 硬件与优化
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--n_GPUs', type=int, default=1)
    parser.add_argument('--n_threads', type=int, default=4)
    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'))
    parser.add_argument('--chop', action='store_true', default=False)
    parser.add_argument('--self_ensemble', action='store_true',help='use self-ensemble method for test')
    parser.add_argument('--seed', type=int, default=1)
    
    # 调度器与其他
    parser.add_argument('--decay', type=str, default='cosine', help='Learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--test_every', type=int, default=1000)
    parser.add_argument('--save_models', action='store_true', default=True)
    parser.add_argument('--save_results', action='store_true', default=True)
    
    # 补充参数 (Trainer 需要)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gclip', type=float, default=0)
    parser.add_argument('--skip_threshold', type=float, default='1e8')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--patience', type=int, default=50, help='Early stopping patience')
    
    # DINOv3 特有
    parser.add_argument('--freeze_backbone', action='store_true', default=False, help='Freeze DINO backbone')

    args = parser.parse_args()
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    
    # 布尔值处理
    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False
            
    return args

def main():
    # 初始化 Checkpoint 和 模型
    _model = model.Model(args, checkpoint)

    # 这里的 Dataset 类是您原文件中的 Flourescenedenoise
    # 注意：该类内部根据 args.data_test (即 testset) 自动加载 Planaria 或 Tribolium 数据
    if not args.test_only:
        loader_train = dataloader.DataLoader(
            Flourescenedenoise(args, istrain=True, c=condition),
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=not args.cpu,
            num_workers=args.n_threads,
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
        # 使用 Trainer 中专门的 3D 去噪测试函数
        t.test3DdenoiseInchannel5(condition=condition)
    else:
        while t.terminate():
            t.train()
    
    t.done()
    checkpoint.done()

if __name__ == '__main__':
    # === 全局配置 ===
    # 设置为 False 进行微调训练，设置为 True 进行测试
    test_only = False  
    
    # 选择数据集 (取消注释以切换)
    # testsetlst = ['Denoising_Planaria'] 
    testsetlst = ['Denoising_Tribolium']
    
    # 预加载的 DINOv3 权重路径
    initial_weights_path = './dinoir_v3_vitb_preloaded.pth' 
    
    modelname = 'DINOIRv3'
    batch_size = 16 # 显存允许的话可以调大
    patch_size = 64
    
    # 循环不同的 condition (1, 2, 3) 进行训练或测试
    for condition in range(1, 4):
        for testset in testsetlst:
            
            # 根据数据集自动设置通道数和保存路径
            if testset == 'Denoising_Planaria':
                inputchannel = 1
                # 如果是微调，resume=0；如果是测试，设为训练好的 epoch
                resume = 0 if not test_only else -1 
            else: # Tribolium
                inputchannel = 5
                resume = 0 if not test_only else -1

            savepath = '%s%s/' % (modelname, testset)
            
            # 只有在测试模式且没有指定路径时，才尝试自动寻找权重
            # 训练模式下，我们使用 initial_weights_path
            if test_only:
                # 尝试加载该 condition 下训练好的模型
                pass 
            
            print(f"\n=== Running {testset} | Condition {condition} | Input Channels {inputchannel} ===")
            
            args = options()
            modelpath = './experiment/%s/model/model_best.pt' % savepath
            # 强制覆盖部分参数以确保正确
            args.modelpath = initial_weights_path if not test_only else args.modelpath
            
            torch.manual_seed(args.seed)
            checkpoint = utility.checkpoint(args)
            main()