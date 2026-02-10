"""
V3 多任务预训练脚本
改进要点:
  1. DinoUniModelV3: 任务专用输入头 + FiLM 调制（替代 zero-pad 统一输入）
  2. Batch 级任务切换: 每个 mini-batch 随机选任务（而非 epoch 级）
  3. task_embed_dim 增大到 128
  4. FiLM 参数独立高学习率（5x base lr）
"""
from csbdeep.models import pretrained
import torch
torch.backends.cudnn.enabled = False
import utility
from utility import savecolorim
import loss
import argparse
import wandb
from mydata import SR, FlouresceneVCD, Flouresceneproj, Flouresceneiso, Flourescenedenoise, \
    normalize, PercentileNormalizer
from torch.utils.data import dataloader
import torch.nn.utils as utils
import model
import os
from decimal import Decimal
import imageio
import numpy as np
from tifffile import imsave
import random
from model.dinoir_v3 import DinoUniModel, DinoUniModelV2, DinoUniModelV3
from analysis import GradientConflictAnalyzer, compute_gradient_similarity_in_training
import itertools

gpu = torch.cuda.is_available()


def options():
    parser = argparse.ArgumentParser(description='FMIR Model V3')
    parser.add_argument('--task', type=int, default=-1)
    parser.add_argument('--model', default='Uni-SwinIR', help='model name')
    parser.add_argument('--save', type=str, default='Uni-DINOv3-pretrain-v3', help='file name to save')
    parser.add_argument('--test_only', action='store_true', default=testonly, help='set this option to test the model')
    parser.add_argument('--cpu', action='store_true', default=not gpu, help='cpu only')
    parser.add_argument('--resume', type=int, default=0, help='-2:best;-1:latest; 0:pretrain; >0: resume')
    parser.add_argument('--pre_train', type=str, default=pretrain, help='pre-trained model directory')
    parser.add_argument('--modelpath', type=str, default='.', help='base path to load model checkpoints')

    # Data specifications
    parser.add_argument('--epochs', type=int, default=2000, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=8, help='input batch size for training')
    parser.add_argument('--patch_size', type=int, default=64, help='input batch size for training')
    parser.add_argument('--rgb_range', type=int, default=1, help='maximum value of RGBn_colors')
    parser.add_argument('--n_colors', type=int, default=1, help='')
    parser.add_argument('--inch', type=int, default=1, help='input channel number')

    parser.add_argument('--datamin', type=int, default=0)
    parser.add_argument('--datamax', type=int, default=100)

    parser.add_argument('--print_every', type=int, default=200, help='')
    parser.add_argument('--test_every', type=int, default=1000)
    parser.add_argument('--load', type=str, default='', help='file name to load')
    parser.add_argument('--lr', type=float, default=0.00005, help='learning rate')
    parser.add_argument('--film_lr_mult', type=float, default=5.0,
                        help='FiLM/TaskEmbed 参数的学习率倍数（相对于 base lr）')

    parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')
    parser.add_argument('--n_resblocks', type=int, default=8, help='number of residual blocks')
    parser.add_argument('--n_feats', type=int, default=32, help='number of feature maps')

    parser.add_argument('--save_models', action='store_true', default=True, help='save all intermediate models')
    parser.add_argument('--save_every', type=int, default=20, help='save checkpoint every N epochs')

    parser.add_argument('--template', default='.', help='You can set various templates in option.py')
    parser.add_argument('--scale', type=str, default='1', help='super resolution scale')
    parser.add_argument('--chop', action='store_true', default=True, help='enable memory-efficient forward')
    parser.add_argument('--self_ensemble', action='store_true', help='use self-ensemble method for test')
    parser.add_argument('--act', type=str, default='relu', help='activation function')
    parser.add_argument('--res_scale', type=float, default=0.1, help='residual scaling')
    parser.add_argument('--dilation', action='store_true', help='use dilated convolution')
    parser.add_argument('--precision', type=str, default='single',
                        choices=('single', 'half'), help='FP precision for test')

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    # Optimization specifications
    parser.add_argument('--decay', type=str, default='200', help='learning rate decay type')
    parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor')
    parser.add_argument('--optimizer', default='ADAM',
                        choices=('SGD', 'ADAM', 'RMSprop'),
                        help='optimizer to use')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold')

    parser.add_argument('--use_lora', type=bool, default=False, help='是否启用 LoRA 微调')

    # Loss specifications
    parser.add_argument('--loss', type=str, default='1*L1+1*L2', help='loss function configuration')

    # V3 特有参数
    parser.add_argument('--task_embed_dim', type=int, default=128,
                        help='任务嵌入维度（V3 默认 128，V2 为 64）')
    parser.add_argument('--batches_per_task', type=int, default=4,
                        help='每个任务连续训练多少个 batch 后切换（mini-batch 级）')

    args = parser.parse_args()

    args.scale = list(map(lambda x: int(x), args.scale.split('+')))

    for arg in vars(args):
        if vars(args)[arg] == 'True':
            vars(args)[arg] = True
        elif vars(args)[arg] == 'False':
            vars(args)[arg] = False

    return args


# ============================================================
# 分参数组优化器：FiLM + TaskEmbed 使用更高学习率
# ============================================================
def make_optimizer_v3(args, target):
    """
    为 V3 模型创建分参数组优化器:
    - 参数组1 (film_params): FiLM 调制器 + Task Embedding → film_lr_mult * lr
    - 参数组2 (other_params): 其余可训练参数 → lr
    """
    film_params = []
    other_params = []

    for name, param in target.named_parameters():
        if not param.requires_grad:
            continue
        if 'film_' in name or 'task_embedding' in name:
            film_params.append(param)
        else:
            other_params.append(param)

    film_lr = args.lr * args.film_lr_mult
    print(f"\n--- 分参数组优化器 ---")
    print(f"  FiLM/TaskEmbed 参数: {sum(p.numel() for p in film_params)/1e6:.3f}M, lr={film_lr:.6f}")
    print(f"  其他可训练参数:      {sum(p.numel() for p in other_params)/1e6:.3f}M, lr={args.lr:.6f}")

    param_groups = [
        {'params': film_params, 'lr': film_lr},
        {'params': other_params, 'lr': args.lr},
    ]

    kwargs_optimizer = {'weight_decay': args.weight_decay}
    if args.optimizer == 'ADAM':
        optimizer = torch.optim.Adam(param_groups, betas=args.betas, eps=args.epsilon, **kwargs_optimizer)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, **kwargs_optimizer)
    elif args.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(param_groups, eps=args.epsilon, **kwargs_optimizer)

    # scheduler
    if args.decay == 'cosine':
        import torch.optim.lr_scheduler as lrs
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    else:
        import torch.optim.lr_scheduler as lrs
        milestones = list(map(lambda x: int(x), args.decay.split('-')))
        scheduler = lrs.MultiStepLR(optimizer, milestones=milestones, gamma=args.gamma)

    # 封装为与原始接口兼容的对象
    class OptimizerWrapper:
        def __init__(self, optimizer, scheduler):
            self.optimizer = optimizer
            self.scheduler = scheduler

        def zero_grad(self):
            self.optimizer.zero_grad()

        def step(self):
            self.optimizer.step()

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch

        def save(self, save_dir):
            torch.save(self.optimizer.state_dict(), os.path.join(save_dir, 'optimizer.pt'))

        def load(self, load_dir, epoch=1):
            self.optimizer.load_state_dict(torch.load(os.path.join(load_dir, 'optimizer.pt')))
            if epoch > 1:
                for _ in range(epoch):
                    self.scheduler.step()

        @property
        def param_groups(self):
            return self.optimizer.param_groups

    return OptimizerWrapper(optimizer, scheduler)


# ============================================================
# 多任务数据加载器管理器
# ============================================================
class MultiTaskDataManager:
    """
    管理所有任务的数据加载器，支持 batch 级任务切换。
    """
    def __init__(self, args, srdatapath, denoisedatapath, isodatapath, prodatapath, voldatapath):
        self.args = args
        self.loaders = {}       # {task_id: train_loader}
        self.iterators = {}     # {task_id: iterator}
        self.test_loaders = {}  # {task_id: [test_loader]}

        print("\n--- 加载所有任务的数据 ---")

        # Task 1: SR (随机选一个子数据集)
        srlst = ['F-actin', 'ER', 'Microtubules', 'CCPs']
        testset_sr = srlst[random.randint(0, 3)]
        self.loaders[1] = dataloader.DataLoader(
            SR(scale=2, name=testset_sr, train=True, rootdatapath=srdatapath,
               patch_size=args.patch_size, length=20),
            batch_size=args.batch_size, shuffle=True,
            pin_memory=not args.cpu, num_workers=0)
        self.test_loaders[1] = [dataloader.DataLoader(
            SR(scale=2, name=testset_sr, train=False, test_only=args.test_only,
               rootdatapath=srdatapath, patch_size=args.patch_size, length=20),
            batch_size=1, shuffle=False,
            pin_memory=not args.cpu, num_workers=0)]
        print(f"  Task 1 (SR): {testset_sr}")

        # Task 2: Denoise
        nlst = ['Denoising_Planaria', 'Denoising_Tribolium']
        testset_dn = nlst[random.randint(0, 1)]
        self.loaders[2] = dataloader.DataLoader(
            Flourescenedenoise(name=testset_dn, istrain=True, c=1, rootdatapath=denoisedatapath,
                               patch_size=args.patch_size, length=2000),
            batch_size=args.batch_size, shuffle=True,
            pin_memory=not args.cpu, num_workers=0)
        self.test_loaders[2] = [dataloader.DataLoader(
            Flourescenedenoise(name=testset_dn, istrain=False, c=1, rootdatapath=denoisedatapath,
                               test_only=args.test_only, patch_size=args.patch_size, length=2000),
            batch_size=1, shuffle=False,
            pin_memory=not args.cpu, num_workers=0)]
        print(f"  Task 2 (Denoise): {testset_dn}")

        # Task 3: Isotropic
        testset_iso = 'Isotropic_Liver'
        self.loaders[3] = dataloader.DataLoader(
            Flouresceneiso(name=testset_iso, istrain=True, rootdatapath=isodatapath,
                           patch_size=args.patch_size, length=2000),
            batch_size=args.batch_size, shuffle=True,
            pin_memory=not args.cpu, num_workers=0)
        self.test_loaders[3] = [dataloader.DataLoader(
            Flouresceneiso(name=testset_iso, istrain=False, rootdatapath=isodatapath,
                           patch_size=args.patch_size, test_only=args.test_only, length=2000),
            batch_size=1, shuffle=False,
            pin_memory=not args.cpu, num_workers=0)]
        print(f"  Task 3 (Iso): {testset_iso}")

        # Task 4: Projection
        testset_proj = 'Projection_Flywing'
        self.loaders[4] = dataloader.DataLoader(
            Flouresceneproj(name=testset_proj, istrain=True, condition=1,
                            rootdatapath=prodatapath, patch_size=args.patch_size, length=2000),
            batch_size=args.batch_size, shuffle=True,
            pin_memory=not args.cpu, num_workers=0)
        self.test_loaders[4] = [dataloader.DataLoader(
            Flouresceneproj(name=testset_proj, istrain=False, condition=1, test_only=args.test_only,
                            rootdatapath=prodatapath, patch_size=args.patch_size, length=2000),
            batch_size=1, shuffle=False,
            pin_memory=not args.cpu, num_workers=0)]
        print(f"  Task 4 (Proj): {testset_proj}")

        # Task 5: Volume
        testset_vol = 'to_predict'
        self.loaders[5] = dataloader.DataLoader(
            FlouresceneVCD(istrain=True, subtestset=testset_vol, test_only=False,
                           rootdatapath=voldatapath, patch_size=args.patch_size, length=2000),
            batch_size=args.batch_size, shuffle=False,
            pin_memory=not args.cpu, num_workers=0)
        self.test_loaders[5] = [dataloader.DataLoader(
            FlouresceneVCD(istrain=False, subtestset=testset_vol, test_only=args.test_only,
                           rootdatapath=voldatapath, patch_size=args.patch_size, length=2000),
            batch_size=1, shuffle=False,
            pin_memory=not args.cpu, num_workers=0)]
        print(f"  Task 5 (Vol): {testset_vol}")

        # 初始化所有迭代器
        for t in self.loaders:
            self.iterators[t] = iter(self.loaders[t])

    def get_batch(self, task_id):
        """获取指定任务的一个 batch，自动处理 epoch 结束后重新循环"""
        try:
            batch = next(self.iterators[task_id])
        except StopIteration:
            self.iterators[task_id] = iter(self.loaders[task_id])
            batch = next(self.iterators[task_id])
        return batch

    def reload_task(self, task_id, new_loader):
        """重新加载某个任务的数据（例如切换 SR 子数据集）"""
        self.loaders[task_id] = new_loader
        self.iterators[task_id] = iter(new_loader)


class PreTrainerV3:
    """
    V3 预训练器:
    - Batch 级任务切换: 每 batches_per_task 个 batch 随机切换任务
    - 分参数组优化器: FiLM/TaskEmbed 使用 5x 学习率
    - 每个 epoch 内所有 5 个任务都会被训练到
    """
    def __init__(self, args, my_model, my_loss, ckp, data_manager):
        self.args = args
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.scale = args.scale
        self.bestpsnr = 0
        self.bestep = 0
        self.ckp = ckp
        self.model = my_model
        self.loss = my_loss
        self.optimizer = make_optimizer_v3(args, self.model)
        self.normalizer = PercentileNormalizer(2, 99.8)
        self.normalizerhr = PercentileNormalizer(2, 99.8)
        self.sepoch = args.resume
        self.epoch = 0
        self.epoch_tsk5 = 0
        self.epoch_tsk4 = 0
        self.test_only = args.test_only
        self.data_manager = data_manager
        self.batches_per_task = args.batches_per_task

        # 每个 epoch 包含的总 batch 数
        # 取所有任务中最长 loader 的长度 × 任务数
        max_loader_len = max(len(loader) for loader in data_manager.loaders.values())
        self.batches_per_epoch = max_loader_len  # 一个 epoch 的 batch 数

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.global_step = 0

        self.error_last = 1e8
        rp = os.path.dirname(__file__)
        self.dir = os.path.join(rp, 'experiment', self.args.save)
        os.makedirs(self.dir, exist_ok=True)
        if not self.args.test_only:
            self.testsave = self.dir + '/Valid/'
            os.makedirs(self.testsave, exist_ok=True)
            self.file = open(self.testsave + "TrainPsnr.txt", 'w')

    def pretrain(self):
        self.pslst = []
        self.sslst = []

        self.loss.step()
        if self.sepoch > 0:
            epoch = self.sepoch
            self.sepoch = 0
            self.epoch = epoch
        else:
            epoch = self.epoch

        lr = self.optimizer.get_lr()
        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        timer_data, timer_model = utility.timer(), utility.timer()

        self.model.train()

        # ====================================================
        # 核心改进: Batch 级任务切换
        # 每 batches_per_task 个 batch 随机切换到新任务
        # 保证每个 epoch 内所有任务都有机会被训练
        # ====================================================
        task_list = [1, 2, 3, 4, 5]
        current_task = random.choice(task_list)
        task_batch_counter = 0

        # 跟踪每个任务在本 epoch 被训练的 batch 数
        task_train_counts = {t: 0 for t in task_list}

        for batch in range(self.batches_per_epoch):
            # 每 batches_per_task 个 batch 切换任务
            if task_batch_counter >= self.batches_per_task:
                current_task = random.choice(task_list)
                task_batch_counter = 0

            self.tsk = current_task
            task_batch_counter += 1
            task_train_counts[current_task] += 1

            if self.tsk == 4:
                self.epoch_tsk4 += 1
            if self.tsk == 5:
                self.epoch_tsk5 += 1

            # 获取当前任务的 batch 数据
            lr_batch, hr_batch, _ = self.data_manager.get_batch(self.tsk)
            lr_batch, hr_batch = self.prepare(lr_batch, hr_batch)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()

            if self.tsk in [1, 2, 3]:
                sr = self.model(lr_batch, self.tsk)
                batch_loss = self.loss(sr, hr_batch)
            elif self.tsk == 4:
                sr_stg1, sr, eacm_loss = self.model(lr_batch, self.tsk)
                lambda_eacm = 0.1
                if self.epoch_tsk4 <= 30 * self.batches_per_epoch:
                    batch_loss = 0.001 * self.loss(sr_stg1, hr_batch) + self.loss(sr, hr_batch) + lambda_eacm * eacm_loss
                else:
                    batch_loss = self.loss(sr, hr_batch) + lambda_eacm * eacm_loss
            elif self.tsk == 5:
                sr_stg1, sr = self.model(lr_batch, self.tsk)
                if self.epoch_tsk5 <= 30 * self.batches_per_epoch:
                    batch_loss = self.loss(sr_stg1, hr_batch)
                else:
                    batch_loss = 0.1 * self.loss(sr_stg1, hr_batch) + self.loss(sr, hr_batch)

            batch_loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            self.optimizer.step()

            self.global_step += 1

            timer_model.hold()
            if batch % self.args.print_every == 0:
                sr2dim = np.float32(normalize(np.squeeze(sr[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                hr2dim = np.float32(normalize(np.squeeze(hr_batch[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                print(f'[Task {self.tsk}] training patch- PSNR/SSIM = {psm:.4f}/{ssmm:.4f}')

                log_dict = {
                    'epoch': epoch,
                    'batch': batch,
                    'task': self.tsk,
                    'loss': batch_loss.item(),
                    'train_psnr': psm,
                    'train_ssim': ssmm,
                    'lr': self.optimizer.get_lr()
                }
                if self.tsk == 4:
                    log_dict['eacm_loss'] = eacm_loss.item()
                wandb.log(log_dict)

                if self.tsk in [4, 5]:
                    sr2dimu = np.float32(
                        normalize(np.squeeze(sr_stg1[0].cpu().detach().numpy()), 0, 100, clip=True)) * 255
                    psm_s1, ssmm_s1 = utility.compute_psnr_and_ssim(sr2dimu, hr2dim)
                    print(f'sr_stg1 training patch = {psm_s1:.4f}/{ssmm_s1:.4f}')
                    wandb.log({'stg1_psnr': psm_s1, 'stg1_ssim': ssmm_s1})

                print(f'Batch{batch}/Epoch{epoch}, Loss = {batch_loss.item():.6f}')
                print(f'  Task distribution: {task_train_counts}')
                print('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    self.batches_per_epoch * self.args.batch_size,
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

            if batch > 0 and batch % self.args.test_every == 0:
                self._run_validation(epoch, batch)

        # Epoch 结束
        self.loss.end_log(self.batches_per_epoch)
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

        self.file.write('Name \n PSNR \n' + str(self.pslst) + '\n SSIM \n' + str(self.sslst))
        save_numbered = (epoch % self.args.save_every == 0) or (epoch == self.args.epochs)
        self.model.save(self.dir + '/model/', epoch, is_best=False, save_numbered=save_numbered)
        self.model.scale = 1
        print(f'save model Epoch{epoch} (numbered={save_numbered}), Loss = {batch_loss.item():.6f}')
        print(f'  Epoch task distribution: {task_train_counts}')

    def _run_validation(self, epoch, batch):
        """运行当前任务的验证"""
        self.loss.end_log(self.batches_per_epoch)
        self.error_last = self.loss.log[-1, -1]

        # 暂时设置 test loader
        self.loader_test = self.data_manager.test_loaders.get(self.tsk, None)
        if self.loader_test is None:
            return

        if self.tsk == 1:
            psnr, ssim = self.testSR(epoch)
        elif self.tsk == 2:
            psnr, ssim = self.test3Ddenoise(epoch, condition=1)
        elif self.tsk == 3:
            psnr, ssim = self.testiso(epoch)
        elif self.tsk == 4:
            psnr, ssim = self.testproj(epoch)
        elif self.tsk == 5:
            psnr, ssim = self.test2to3(epoch)
        else:
            return

        self.pslst.append(psnr)
        self.sslst.append(ssim)

        wandb.log({
            'val_psnr': psnr,
            'val_ssim': ssim,
            'best_psnr': self.bestpsnr,
            'best_epoch': self.bestep,
            'val_task': self.tsk,
        })

        self.model.train()
        self.loss.step()
        lr = self.optimizer.get_lr()
        print(f'Evaluation -- Batch{batch}/Epoch{epoch}')
        self.ckp.write_log(f'Batch{batch}/Epoch{epoch}' +
                           '\tLearning rate: {:.2e}'.format(Decimal(lr)))
        self.loss.start_log()

    def testall(self, tsk, subd=-1, condition=1):
        self.tsk = tsk
        self.loader_test = self.data_manager.test_loaders.get(tsk, None)
        if self.loader_test is None:
            print(f"No test loader for task {tsk}")
            return 0, 0

        if tsk == 1:
            p, s = self.testSR()
        elif tsk == 2:
            p, s = self.test3Ddenoise(condition=condition)
        elif tsk == 3:
            p, s = self.testiso()
        elif tsk == 4:
            p, s = self.testproj(condition=condition)
        elif tsk == 5:
            p, s = self.test2to3()
        return p, s

    # ============== 以下测试函数与 V1/V2 完全一致 ==============

    def testSR(self, epoch=0):
        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/' % self.args.resume
        os.makedirs(self.testsave, exist_ok=True)
        self.model.scale = 2

        torch.set_grad_enabled(False)

        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        num = 0
        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lr, hr, filename) in enumerate(self.loader_test[0]):
            nmlst.append(filename)
            if not self.args.test_only and num >= 5:
                break
            num += 1
            lr, hr = self.prepare(lr, hr)
            sr = self.model(lr, 1)
            sr = utility.quantize(sr, self.args.rgb_range)
            hr = utility.quantize(hr, self.args.rgb_range)

            pst = utility.calc_psnr(sr, hr, self.scale[0], self.args.rgb_range, dataset=None)
            sr = sr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            hr = hr.mul(255 / self.args.rgb_range).detach().cpu().numpy()[0, 0, :, :]
            ps, ss = utility.compute_psnr_and_ssim(sr, hr)
            sr255 = np.float32(normalize(sr, 0, 100, clip=True)) * 255
            hr255 = np.float32(normalize(hr, 0, 100, clip=True)) * 255
            ps255, ss255 = utility.compute_psnr_and_ssim(sr255, hr255)

            pslst.append(np.max([ps, pst, ps255]))
            sslst.append(ss255)
            print('pst, ps, ss, ps255, ss255 = ', pst, ps, ss, ps255, ss255)
            if self.args.test_only:
                name = '{}.png'.format(filename[0][:-4])
                imageio.imwrite(self.testsave + name, sr)
                savecolorim(self.testsave + name[:-4] + '-Color.png', sr, norm=False)
                sr = np.round(np.maximum(0, np.minimum(255, sr)))
                hr2 = np.round(np.maximum(0, np.minimum(255, hr)))
                res = np.clip(np.abs(sr - hr2), 0, 255)
                savecolorim(self.testsave + name[:-4] + '-MeandfnoNormC.png', res, norm=False)

        psnrmean = np.mean(pslst)
        ssimmean = np.mean(sslst)

        if self.args.test_only:
            file = open(self.testsave + "Psnrssim100_norm.txt", 'w')
            file.write('Mean = ' + str(psnrmean) + str(ssimmean))
            file.write('\nName \n' + str(nmlst) + '\n PSNR t \n'
                       + '\n PSNR max \n' + str(pslst)
                       + '\n SSIM \n' + str(sslst))
            file.close()
        else:
            if psnrmean > self.bestpsnr:
                self.bestpsnr = psnrmean
                self.bestep = epoch
                self.model.save(self.dir, epoch, is_best=(self.bestep == epoch), save_numbered=False)

        print('num = ', num, 'psnrmean SSIM = ', psnrmean, ssimmean)
        torch.set_grad_enabled(True)
        return psnrmean, ssimmean

    def test3Ddenoise(self, epoch=0, condition=1, data_test='Denoising_Tribolium'):
        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/condition_%d/' % (self.args.resume, condition)
            os.makedirs(self.testsave, exist_ok=True)
            file = open(self.testsave + '/Psnrssim_Im_patch_c%d.txt' % condition, 'w')

        datamin, datamax = self.args.datamin, self.args.datamax
        patchsize = 600
        torch.set_grad_enabled(False)
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        num = 0
        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            if not self.args.test_only and num >= 1:
                break
            num += 1

            nmlst.append(filename)
            print('filename = ', filename)
            if filename[0] == '':
                name = 'im%d' % idx_data
            else:
                name = '{}'.format(filename[0])
            if not self.args.test_only:
                name = 'EP{}_{}'.format(epoch, filename[0])

            lrt = self.normalizer.before(lrt, 'CZYX')
            hrt = self.normalizerhr.before(hrt, 'CZYX')
            lrt, hrt = self.prepare(lrt, hrt)

            lr = np.squeeze(lrt.cpu().detach().numpy())
            hr = np.squeeze(hrt.cpu().detach().numpy())
            print('hr.shape = ', hr.shape)
            denoiseim = torch.zeros_like(hrt, dtype=hrt.dtype)

            batchstep = 2
            inputlst = []
            for ch in range(0, len(hr)):
                if ch < 5 // 2:
                    lr1 = [lrt[:, ch:ch + 1, :, :] for _ in range(5 // 2 - ch)]
                    lr1.append(lrt[:, :5 // 2 + ch + 1])
                    lrt1 = torch.concat(lr1, 1)
                elif ch >= (len(hr) - 5 // 2):
                    lr1 = []
                    lr1.append(lrt[:, ch - 5 // 2:])
                    numa = (5 // 2 - (len(hr) - ch)) + 1
                    lr1.extend([lrt[:, ch:ch + 1, :, :] for _ in range(numa)])
                    lrt1 = torch.concat(lr1, 1)
                else:
                    lrt1 = lrt[:, ch - 5 // 2:ch + 5 // 2 + 1]
                assert lrt1.shape[1] == 5
                inputlst.append(lrt1)

            torch.cuda.empty_cache()

            for dp in range(0, len(inputlst), batchstep):
                slice_end = min(dp + batchstep, len(inputlst))
                lrtn = torch.concat(inputlst[dp:slice_end], 0)
                with torch.no_grad():
                    a = self.model(lrtn, 2)
                a = torch.transpose(a, 1, 0)
                denoiseim[:, dp:slice_end, :, :] = a
                del lrtn, a
                torch.cuda.empty_cache()
                print(f'Processed slice {dp} to {slice_end}')

            sr = np.float32(denoiseim.cpu().detach().numpy())
            sr = np.squeeze(self.normalizer.after(sr))
            hr = np.squeeze(self.normalizerhr.after(hr))

            sr255 = np.squeeze(np.float32(normalize(sr, datamin, datamax, clip=True))) * 255
            hr255 = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255
            lr255 = np.float32(normalize(lr, datamin, datamax, clip=True)) * 255

            cpsnrlst = []
            cssimlst = []
            step = 1
            if self.args.test_only:
                imsave(self.testsave + name + '.tif', sr)
                if 'Planaria' in data_test:
                    if condition == 1:
                        randcs = 10
                        randce = hr.shape[0] - 10
                        step = (hr.shape[0] - 20) // 5
                    else:
                        randcs = 85
                        randce = 87
                        step = 1
                        if randce >= hr.shape[0]:
                            randcs = hr.shape[0] - 3
                            randce = hr.shape[0]

                    for dp in range(randcs, randce, step):
                        savecolorim(self.testsave + name + '-dfnoNormC%d.png' % dp, sr[dp] - hr[dp], norm=False)
                        savecolorim(self.testsave + name + '-C%d.png' % dp, sr[dp])
                        savecolorim(self.testsave + name + '-HRC%d.png' % dp, hr[dp])
                        srpatch255 = sr255[dp, :patchsize, :patchsize]
                        hrpatch255 = hr255[dp, :patchsize, :patchsize]
                        lrpatch255 = lr255[dp, :patchsize, :patchsize]
                        psm, ssmm = utility.compute_psnr_and_ssim(srpatch255, hrpatch255)
                        psml, ssmml = utility.compute_psnr_and_ssim(lrpatch255, hrpatch255)
                        print('SR Image %s - C%d- PSNR/SSIM/MSE = %f/%f' % (name, dp, psm, ssmm))
                        print('LR PSNR/SSIM = %f/%f' % (psml, ssmml))
                        cpsnrlst.append(psm)
                        cssimlst.append(ssmm)
                elif 'Tribolium' in data_test:
                    if condition == 1:
                        randcs = 2
                        randce = hr.shape[0] - 2
                        step = (hr.shape[0] - 4) // 6
                    else:
                        randcs = hr.shape[0] // 2 - 1
                        randce = randcs + 3
                        step = 1
                    for randc in range(randcs, randce, step):
                        hrpatch = normalize(hr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        srpatchour = normalize(sr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                        psm, ssmm = utility.compute_psnr_and_ssim(srpatchour, hrpatch)
                        cpsnrlst.append(psm)
                        cssimlst.append(ssmm)
                file.write('Image \"%s\" Channel = %d-%d \n' % (name, randcs, randce) + 'PSNR = ' + str(
                    cpsnrlst) + '\n SSIM = ' + str(cssimlst))
            else:
                randcs = 0
                randce = hr.shape[0]
                for randc in range(randcs, randce, step):
                    hrpatch = normalize(hr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                    srpatchour = normalize(sr255[randc, :patchsize, :patchsize], datamin, datamax, clip=True) * 255
                    psm, ssmm = utility.compute_psnr_and_ssim(srpatchour, hrpatch)
                    cpsnrlst.append(psm)
                    cssimlst.append(ssmm)

            psnr1, ssim = np.mean(np.array(cpsnrlst)), np.mean(np.array(cssimlst))
            print('SR im:', psnr1, ssim)
            sslst.append(ssim)
            pslst.append(psnr1)

        psnrm = np.mean(np.array(pslst))
        ssimm = np.mean(np.array(sslst))
        if self.args.test_only:
            print('+++++++++ condition%d meanSR++++++++++++' % condition, sum(pslst) / len(pslst),
                  sum(sslst) / len(sslst))
            file.write('\n \n +++++++++ condition%d meanSR ++++++++++++ \n PSNR/SSIM \n  patchsize = %d \n' % (
                condition, patchsize))
            file.write('Name \n' + str(nmlst) + '\n PSNR = ' + str(pslst) + '\n SSIM = ' + str(sslst))
            file.close()
        else:
            if psnrm > self.bestpsnr:
                self.bestpsnr = psnrm
                self.bestep = epoch
            self.model.save(self.dir, epoch, is_best=(self.bestep == epoch), save_numbered=False)
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrm, self.bestpsnr, self.bestep)
        print('psnrm, np.mean(np.array(sslst)) = ', psnrm, ssimm)
        torch.set_grad_enabled(True)
        return psnrm, ssimm

    def testiso(self, epoch=0):
        def _rotate(arr, k=1, axis=1, copy=True):
            if copy:
                arr = arr.copy()
            k = k % 4
            arr = np.rollaxis(arr, axis, arr.ndim)
            if k == 0:
                res = arr
            elif k == 1:
                res = arr[::-1].swapaxes(0, 1)
            elif k == 2:
                res = arr[::-1, ::-1]
            else:
                res = arr.swapaxes(0, 1)[::-1]
            res = np.rollaxis(res, -1, axis)
            return res

        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/' % self.args.resume
            os.makedirs(self.testsave, exist_ok=True)
        datamin, datamax = self.args.datamin, self.args.datamax

        torch.set_grad_enabled(False)
        if epoch is None:
            epoch = self.optimizer.get_last_epoch()

        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        num = 0
        pslst = []
        sslst = []
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            num += 1
            nmlst.append(filename)
            name = '{}'.format(filename[0])

            lrt = self.normalizer.before(lrt, 'CZYX')
            hrt = self.normalizerhr.before(hrt, 'CZYX')
            lrt, hrt = self.prepare(lrt, hrt)

            lr = np.float32(np.squeeze(lrt.cpu().detach().numpy()))
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))

            if len(lr.shape) <= 3:
                lr = np.expand_dims(lr, -1)
                hr = np.expand_dims(hr, -1)
            isoim1 = np.zeros_like(hr, dtype=np.float32)
            isoim2 = np.zeros_like(hr, dtype=np.float32)

            batchstep = 30
            torch.cuda.empty_cache()
            for wp in range(0, hr.shape[2], batchstep):
                if wp + batchstep >= hr.shape[2]:
                    wp = hr.shape[2] - batchstep
                x_rot1 = _rotate(lr[:, :, wp:wp + batchstep, :], axis=1, copy=False)
                x_rot1 = np.expand_dims(np.squeeze(x_rot1), 1)
                x_rot1 = torch.from_numpy(np.ascontiguousarray(x_rot1)).float()
                mini_bs = 8
                a1_chunks = []
                with torch.no_grad():
                    for i in range(0, x_rot1.shape[0], mini_bs):
                        batch_in = self.prepare(x_rot1[i:i + mini_bs])[0]
                        batch_out = self.model(batch_in, 3)
                        a1_chunks.append(batch_out.cpu())
                        del batch_in, batch_out
                        torch.cuda.empty_cache()
                a1 = torch.cat(a1_chunks, dim=0)
                a1 = np.expand_dims(np.squeeze(a1.detach().numpy()), -1)
                u1 = _rotate(a1, -1, axis=1, copy=False)
                isoim1[:, :, wp:wp + batchstep, :] = u1
                del x_rot1, a1, a1_chunks
                torch.cuda.empty_cache()
            for hp in range(0, hr.shape[1], batchstep):
                if hp + batchstep >= hr.shape[1]:
                    hp = hr.shape[1] - batchstep
                x_rot2 = _rotate(_rotate(lr[:, hp:hp + batchstep, :, :], axis=2, copy=False), axis=0, copy=False)
                x_rot2 = np.expand_dims(np.squeeze(x_rot2), 1)
                x_rot2 = torch.from_numpy(np.ascontiguousarray(x_rot2)).float()
                mini_bs = 8
                a2_chunks = []
                with torch.no_grad():
                    for i in range(0, x_rot2.shape[0], mini_bs):
                        batch_in = self.prepare(x_rot2[i:i + mini_bs])[0]
                        batch_out = self.model(batch_in, 3)
                        a2_chunks.append(batch_out.cpu())
                        del batch_in, batch_out
                        torch.cuda.empty_cache()
                a2 = torch.cat(a2_chunks, dim=0)
                a2 = np.expand_dims(np.squeeze(a2.detach().numpy()), -1)
                u2 = _rotate(_rotate(a2, -1, axis=0, copy=False), -1, axis=2, copy=False)
                isoim2[:, hp:hp + batchstep, :, :] = u2
                del x_rot2, a2, a2_chunks
                torch.cuda.empty_cache()

            sr = np.sqrt(np.maximum(isoim1, 0) * np.maximum(isoim2, 0))
            sr = np.squeeze(self.normalizer.after(sr))
            lr = np.squeeze(self.normalizer.after(lr))
            imsave(self.testsave + name + '.tif', sr)
            hr = np.squeeze(self.normalizerhr.after(hr))
            c, h, w = hr.shape

            cpsnrlst = []
            cssimlst = []
            for dp in range(1, h, h // 5):
                if self.args.test_only:
                    savecolorim(self.testsave + name + '-dfnoNormCz%d.png' % dp, sr[:, dp, :] - hr[:, dp, :], norm=False)
                    savecolorim(self.testsave + name + '-C%d.png' % dp, sr[:, dp, :])
                    savecolorim(self.testsave + name + '-GTC%d.png' % dp, hr[:, dp, :])
                    savecolorim(self.testsave + name + '-LRC%d.png' % dp, lr[:, dp, :])
                hrpatch = normalize(hr[:, dp, :], datamin, datamax, clip=True) * 255
                lrpatch = normalize(lr[:, dp, :], datamin, datamax, clip=True) * 255
                srpatch = normalize(sr[:, dp, :], datamin, datamax, clip=True) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(srpatch, hrpatch)
                psml, ssmml = utility.compute_psnr_and_ssim(lrpatch, hrpatch)
                print('Normalized Patch %s - C%d- PSNR/SSIM = %f/%f' % (name, dp, psm, ssmm))
                print('Normalized LR PSNR/SSIM = %f/%f' % (psml, ssmml))
                cpsnrlst.append(psm)
                cssimlst.append(ssmm)
            psnr1, ssim = np.mean(np.array(cpsnrlst)), np.mean(np.array(cssimlst))
            sslst.append(ssim)
            pslst.append(psnr1)

        psnrm = np.mean(np.array(pslst))
        ssmm = np.mean(np.array(sslst))
        if self.args.test_only:
            file = open(self.testsave + "Psnrssim.txt", 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(pslst) + '\n SSIM \n' + str(sslst))
            file.close()
        else:
            if psnrm > self.bestpsnr:
                self.bestpsnr = psnrm
                self.bestep = epoch
                self.model.save(self.dir, epoch, is_best=(self.bestep == epoch), save_numbered=False)
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
        return psnrm, ssmm

    def testproj(self, epoch=0, condition=2):
        if self.args.test_only:
            self.testsave = self.dir + '/results/model%d/c%d/' % (self.args.resume, condition)
            os.makedirs(self.testsave, exist_ok=True)
        print('save to', self.testsave)

        datamin, datamax = self.args.datamin, self.args.datamax

        torch.set_grad_enabled(False)
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        psnrall, ssimall = [], []
        psnralls1, ssimalls1 = [], []
        num = 0
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            if not self.args.test_only and num >= 1:
                break
            num += 1
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            lrt, hrt = self.prepare(lrt, hrt)
            a_stg1, a, _ = self.model(lrt, 4)

            sr_stg1 = np.float32(np.squeeze(a_stg1.cpu().detach().numpy()))
            sr = np.float32(np.squeeze(a.cpu().detach().numpy()))
            srtf = sr
            if self.args.test_only:
                axes_restored = 'YX'
                utility.save_tiff_imagej_compatible(self.testsave + name + '.tif', srtf, axes_restored)
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))

            hr2dim = np.float32(normalize(hr, datamin, datamax, clip=True)) * 255
            sr2dim = np.float32(normalize(np.float32(srtf), datamin, datamax, clip=True)) * 255
            sr2dim_stg1 = np.float32(normalize(np.float32(sr_stg1), datamin, datamax, clip=True)) * 255
            psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
            psm_stg1, ssmm_stg1 = utility.compute_psnr_and_ssim(sr2dim_stg1, hr2dim)
            print('2D img Norm-%s - PSNR/SSIM = %f/%f / Output of StageI = %f/%f' % (
                name, psm, ssmm, psm_stg1, ssmm_stg1))

            psnralls1.append(psm_stg1)
            ssimalls1.append(ssmm_stg1)
            psnrall.append(psm_stg1)
            ssimall.append(ssmm_stg1)

        psnrallm = np.mean(np.array(psnrall))
        ssimallm = np.mean(np.array(ssimall))
        if self.args.test_only:
            file = open(self.testsave + "Psnrssim.txt_c%d.txt" % condition, 'w')
            file.write('Name \n' + str(nmlst) + '\n PSNR \n' + str(psnrall) + '\n SSIM \n' + str(ssimall))
            file.close()
        else:
            if psnrallm > self.bestpsnr:
                self.bestpsnr = psnrallm
                self.bestep = epoch
            self.model.save(self.dir, epoch, is_best=(self.bestep == epoch), save_numbered=False)

        print('+++++++++ condition %d ++++++++++++' % condition, psnrallm, ssimallm)
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrallm, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
        return psnrallm, ssimallm

    def test2to3(self, epoch=0, subtestset='to_predict'):
        if self.args.test_only:
            self.testsave = self.dir + 'results/model_%d/%s/' % (self.args.resume, subtestset)
            os.makedirs(self.testsave, exist_ok=True)

        datamin, datamax = self.args.datamin, self.args.datamax
        torch.set_grad_enabled(False)
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        self.model.eval()

        psnralls1 = []
        psnrall = []
        ssimalls1 = []
        ssimall = []
        num = 0
        nmlst = []
        for idx_data, (lrt, hrt, filename) in enumerate(self.loader_test[0]):
            if not self.args.test_only and num >= 2:
                break
            nmlst.append(filename)
            name = '{}'.format(filename[0])
            if name == '':
                name = 'im%d' % idx_data
            lrt, hrt = self.prepare(lrt, hrt)
            as1, a = self.model(lrt, 5)

            sr = np.float32(a.cpu().detach().numpy())
            srs1 = np.float32(as1.cpu().detach().numpy())
            imsave(self.testsave + name + 'norm.tif', np.squeeze(sr))

            sr = (np.clip(np.squeeze(sr), -1, 1) + 1) / 2
            srs1 = (np.clip(np.squeeze(srs1), -1, 1) + 1) / 2
            hr = np.float32(np.squeeze(hrt.cpu().detach().numpy()))
            hr = (np.clip(hr, -1, 1) + 1) / 2

            savecolorim(self.testsave + 'OriIm' + name + '-Result.png', sr[0])
            savecolorim(self.testsave + 'OriIm' + name + '-HR.png', hr[0])
            for i in range(0, len(hr), 10):
                num += 1
                hr2dim = np.float32(normalize(hr[i], datamin, datamax, clip=True)) * 255
                sr2dim = np.float32(normalize(sr[i], datamin, datamax, clip=True)) * 255
                sr2dims1 = np.float32(normalize(srs1[i], datamin, datamax, clip=True)) * 255
                psm, ssmm = utility.compute_psnr_and_ssim(sr2dim, hr2dim)
                psms1, ssmms1 = utility.compute_psnr_and_ssim(sr2dims1, hr2dim)
                psms1 = np.max([0, np.min([100, psms1])])
                psnralls1.append(psms1)
                ssimalls1.append(ssmms1)
                psnrall.append(psms1)
                ssimall.append(ssmms1)
                print('Stage I/II 2D img Norm-%s - PSNR/SSIM = %f/%f / %f/%f' % (name, psms1, ssmms1, psm, ssmm))

        psnrmean = np.mean(psnrall)
        ssmean = np.mean(ssimall)
        if psnrmean > self.bestpsnr:
            self.bestpsnr = psnrmean
            self.bestep = epoch
        if not self.args.test_only:
            self.model.save(self.dir, epoch, is_best=(self.bestep == epoch), save_numbered=False)
        print('%%% ~~~~~~~~~~~~ %%% psnrm, self.bestpsnr, self.bestep ', psnrmean, self.bestpsnr, self.bestep)
        torch.set_grad_enabled(True)
        return psnrmean, ssmean

    def prepare(self, *args):
        def _prepare(tensor):
            if self.args.precision == 'half':
                tensor = tensor.half()
            return tensor.to(self.device)
        return [_prepare(a) for a in args]

    def terminate(self):
        if self.test_only:
            return False
        else:
            self.epoch = self.epoch + 1
            if self.epoch > self.args.epochs:
                self.file.close()
            return self.epoch <= self.args.epochs


if __name__ == '__main__':
    srdatapath = './CSB/DataSet/BioSR_WF_to_SIM/DL-SR-main/dataset/'
    denoisedatapath = './CSB/DataSet/'
    isodatapath = './CSB/DataSet/Isotropic/'
    prodatapath = './CSB/DataSet/'
    voldatapath = './VCD/vcdnet/'

    pretrain = '.'
    testonly = False

    use_lora = False

    args = options()
    torch.manual_seed(args.seed)

    # 初始化 wandb
    wandb.init(
        project="UniFMIR-Pretrain",
        name=f"Uni-DINOv3-V3-{args.save}",
        config=vars(args),
        resume="allow"
    )

    checkpoint = utility.checkpoint(args)
    assert checkpoint.ok

    # ========== V3 模型 ==========
    print("\n" + "=" * 60)
    print("🚀 使用 DinoUniModelV3: 任务专用输入头 + FiLM 调制（改进版）")
    print(f"   task_embed_dim={args.task_embed_dim}, film_lr_mult={args.film_lr_mult}")
    print(f"   batches_per_task={args.batches_per_task} (batch 级任务切换)")
    print("=" * 60)

    unimodel = DinoUniModelV3(
        args, embed_dim=384, dino_depth=12, dino_num_heads=6,
        task_embed_dim=args.task_embed_dim
    )

    # 加载预训练权重（V3 的 backbone + task heads 与 V1 兼容）
    preloaded_path = './dinoir_v3_vits_unipreload.pth'
    if os.path.exists(preloaded_path):
        print(f"\nLoading preloaded DINO weights from {preloaded_path}")
        state_dict = torch.load(preloaded_path)
        model_state = unimodel.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"  Skipping '{k}': shape {v.shape} vs {model_state[k].shape}")
            else:
                print(f"  Skipping '{k}': not in model")
        unimodel.load_state_dict(filtered_state_dict, strict=False)
        print(f"  Loaded {len(filtered_state_dict)}/{len(state_dict)} keys from checkpoint")
    else:
        print("Warning: Preloaded weights not found, starting from scratch!")

    # ========== 部分冻结 ==========
    freeze_depth = 0
    print(f"\n--- [部分冻结] 冻结 RoPE + Patch Embed + 前 {freeze_depth} 层 Block ---")

    for name, param in unimodel.named_parameters():
        if "rope_embed" in name:
            param.requires_grad = False
        elif name.startswith("patch_embed"):
            param.requires_grad = False
        elif "blocks." in name:
            parts = name.split('.')
            try:
                block_idx = int(parts[parts.index("blocks") + 1])
                if block_idx < freeze_depth:
                    param.requires_grad = False
            except (ValueError, IndexError):
                pass
        elif name == "norm.weight" or name == "norm.bias":
            param.requires_grad = False

    # 统计参数
    total_params = sum(p.numel() for p in unimodel.parameters())
    trainable_params = sum(p.numel() for p in unimodel.parameters() if p.requires_grad)
    film_params = sum(p.numel() for n, p in unimodel.named_parameters()
                      if p.requires_grad and ('film_' in n or 'task_embedding' in n))
    frozen_params = total_params - trainable_params
    print(f"\n--- 参数统计 ---")
    print(f"  总参数:         {total_params / 1e6:.2f}M")
    print(f"  冻结参数:       {frozen_params / 1e6:.2f}M ({100 * frozen_params / total_params:.1f}%)")
    print(f"  可训练参数:     {trainable_params / 1e6:.2f}M ({100 * trainable_params / total_params:.1f}%)")
    print(f"  FiLM/TaskEmbed: {film_params / 1e6:.2f}M (使用 {args.film_lr_mult}x 学习率)")

    _model = model.Model(args, checkpoint, unimodel)
    _loss = loss.Loss(args, checkpoint) if not args.test_only else None

    # ========== 多任务数据管理器 ==========
    data_manager = MultiTaskDataManager(
        args, srdatapath, denoisedatapath, isodatapath, prodatapath, voldatapath
    )

    t = PreTrainerV3(args, _model, _loss, checkpoint, data_manager)

    if testonly:
        for i in range(1, 6):
            t.testall(tsk=i)
    else:
        while t.terminate():
            t.pretrain()

    checkpoint.done()
