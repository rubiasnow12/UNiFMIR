import torch
torch.backends.cudnn.enabled = False
import utility
import loss
import argparse
from div2k import Flourescenedenoise
from trainer import Trainer
from torch.utils.data import dataloader
import model

# ######### 在这里修改参数 #########
modelname = 'DINOIRv3'
testset = 'Denoising_Planaria' 
test_only = True
condition = 1
resume = 0 
batchsize = 4 
patch_size = 64
inputchannel = 1
# #################################

def options():
    parser = argparse.ArgumentParser(description='DINOIRv3 Test')
    parser.add_argument('--self_ensemble', action='store_true', default=False,
                        help='use self-ensemble during test')
    parser.add_argument('--chop', action='store_true', default=False,
                        help='use chopping to forward large images (compatible flag)')
    parser.add_argument('--model', default=modelname, help='model name')
    parser.add_argument('--test_only', action='store_true', default=test_only)
    parser.add_argument('--resume', type=int, default=resume)
    parser.add_argument('--save', type=str, default=f'{modelname}{testset}')
    
    # Data specifications
    parser.add_argument('--data_test', type=str, default=testset)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--rgb_range', type=int, default=1)
    parser.add_argument('--n_colors', type=int, default=1)
    parser.add_argument('--inputchannel', type=int, default=inputchannel)
    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)
    parser.add_argument('--condition', type=int, default=condition)
    parser.add_argument('--batch_size', type=int, default=batchsize)
    parser.add_argument('--patch_size', type=int, default=patch_size)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--n_GPUs', type=int, default=1)
    parser.add_argument('--n_threads', type=int, default=0)
    parser.add_argument('--scale', type=str, default='1') # 去噪任务 scale 为 1
    
    # Model specifications
    parser.add_argument('--precision', type=str, default='single', choices=('single', 'half'))
    
    # Log specifications
    parser.add_argument('--save_results', action='store_true', default=True)
    
    # 兼容性参数
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--loss', type=str, default='1*L1')
    parser.add_argument('--optimizer', default='ADAM')
    parser.add_argument('--decay', type=str, default='200')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--gclip', type=float, default=0)
    parser.add_argument('--skip_threshold', type=float, default='1e8')
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_models', action='store_true', default=False)
    parser.add_argument('--print_every', type=int, default=100)
    parser.add_argument('--test_every', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=1)
    
    args = parser.parse_args()
    args.scale = list(map(lambda x: int(x), args.scale.split('+')))
    return args

def main():
    _model = model.Model(args, checkpoint)
    
    loader_test = [dataloader.DataLoader(
        Flourescenedenoise(args, istrain=False, c=condition),
        batch_size=1,
        shuffle=False,
        pin_memory=not args.cpu,
        num_workers=args.n_threads,
    )]
    
    _loss = None
    t = Trainer(args, None, loader_test, args.data_test, _model, _loss, checkpoint)
    
    # 使用 trainer 中的去噪测试函数
    t.test3DdenoiseInchannel5(condition=condition)
    
    checkpoint.done()

if __name__ == '__main__':
    args = options()
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    main()