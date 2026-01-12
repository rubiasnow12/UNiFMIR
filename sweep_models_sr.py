import argparse
import glob
import os
import re

import torch

import utility
import model
from div2k import DIV2K
from trainer import Trainer
from torch.utils.data import dataloader


def parse_args():
    p = argparse.ArgumentParser(description="Sweep SR checkpoints and pick best PSNR/SSIM")
    p.add_argument("--exp", type=str, required=True,
                   help="Experiment folder under ./experiment, e.g. DINOIRv3F-actin-frozen")
    p.add_argument("--testset", type=str, required=True,
                   help="Dataset name for DIV2K, e.g. F-actin / CCPs / ER / Microtubules")
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--inputchannel", type=int, default=3,
                   help="Model in_chans; keep 3 for DINOv3 checkpoints")
    p.add_argument("--rgb_range", type=int, default=1)
    p.add_argument("--datamin", type=int, default=0)
    p.add_argument("--datamax", type=int, default=100)
    p.add_argument("--cpu", action="store_true", default=False)
    p.add_argument("--chop", action="store_true", default=True)
    p.add_argument("--n_threads", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1000,
                   help="Only used by scheduler; irrelevant for test")
    p.add_argument("--start", type=int, default=None)
    p.add_argument("--end", type=int, default=None)
    p.add_argument("--pattern", type=str, default="model_*.pt",
                   help="Glob pattern under experiment/<exp>/model/")
    return p.parse_args()


def build_test_args(ns, resume_epoch: int):
    # Build a lightweight args object compatible with Model/Trainer/DIV2K
    class A:
        pass

    args = A()
    args.model = "DINOIRv3"
    args.test_only = True
    args.resume = int(resume_epoch)

    # IMPORTANT: args.save determines experiment path: ./experiment/<save>/
    args.save = ns.exp.rstrip("/") + "/"
    args.load = ""  # keep empty; checkpoint() will use args.save

    args.modelpath = "."  # force load by resume from experiment/<save>/model/model_<resume>.pt

    args.task = -1
    args.dir_data = None
    args.dir_demo = None
    args.data_test = ns.testset

    args.epochs = ns.epochs
    args.batch_size = 1
    args.patch_size = 128

    args.rgb_range = ns.rgb_range
    args.n_colors = 1
    args.inch = 1
    args.inputchannel = ns.inputchannel
    args.datamin = ns.datamin
    args.datamax = ns.datamax

    args.cpu = ns.cpu
    args.print_every = 1000
    args.test_every = 2000
    args.lr = 5e-5

    args.n_GPUs = 1
    args.n_resblocks = 8
    args.n_feats = 32

    args.save_models = False
    args.save_results = False  # key: avoid saving images during sweep
    args.save_gt = False

    args.debug = False
    args.scale = [int(ns.scale)]
    args.chunk_size = 144
    args.n_hashes = 4
    args.chop = ns.chop
    args.self_ensemble = False
    args.no_augment = True

    args.act = "relu"
    args.extend = "."
    args.res_scale = 0.1
    args.shift_mean = True
    args.dilation = False
    args.precision = "single"

    args.seed = 1
    args.local_rank = 0

    args.n_threads = ns.n_threads
    args.reset = False
    args.split_batch = 1
    args.gan_k = 1

    args.decay = "200"
    args.gamma = 0.5
    args.optimizer = "ADAM"
    args.momentum = 0.9
    args.betas = (0.9, 0.999)
    args.epsilon = 1e-8
    args.weight_decay = 0
    args.gclip = 0

    args.loss = "1*L1"
    args.skip_threshold = 1e8

    # DINO finetune flag used in model/__init__.py; keep false for pure evaluation
    args.freeze_backbone = False

    return args


def list_epochs(model_dir: str, pattern: str, start=None, end=None):
    paths = glob.glob(os.path.join(model_dir, pattern))
    epochs = []
    rx = re.compile(r"model_(\d+)\.pt$")
    for p in paths:
        m = rx.search(p)
        if not m:
            continue
        e = int(m.group(1))
        if start is not None and e < start:
            continue
        if end is not None and e > end:
            continue
        epochs.append(e)
    return sorted(set(epochs))


def main():
    ns = parse_args()

    torch.backends.cudnn.enabled = False

    model_dir = os.path.join("experiment", ns.exp, "model")
    if not os.path.isdir(model_dir):
        raise SystemExit(f"找不到模型目录: {model_dir}")

    epochs = list_epochs(model_dir, ns.pattern, ns.start, ns.end)
    if not epochs:
        raise SystemExit(f"未找到任何 checkpoint: {model_dir}/{ns.pattern}")

    best_psnr = (-1e9, None)
    best_ssim = (-1e9, None)

    # Reuse one test loader across epochs (dataset independent of epoch)
    # NOTE: we must rebuild checkpoint/model per epoch because weights differ.
    for e in epochs:
        args = build_test_args(ns, e)
        torch.manual_seed(args.seed)

        ckp = utility.checkpoint(args)
        loader_test = [dataloader.DataLoader(
            DIV2K(args, name=ns.testset, train=False, benchmark=False),
            batch_size=1,
            shuffle=False,
            pin_memory=not args.cpu,
            num_workers=args.n_threads,
        )]

        net = model.Model(args, ckp)
        t = Trainer(args, loader_train=None, loader_test=loader_test,
                    datasetname=args.data_test, my_model=net, my_loss=None, ckp=ckp)

        psnr_mean, ssim_mean = t.test(batch=0, epoch=e)
        t.done()
        ckp.done()

        print(f"[model_{e:04d}] PSNR={psnr_mean:.6f} SSIM={ssim_mean:.6f}")

        if psnr_mean > best_psnr[0]:
            best_psnr = (psnr_mean, e)
        if ssim_mean > best_ssim[0]:
            best_ssim = (ssim_mean, e)

    print("=" * 60)
    print(f"Best PSNR: epoch={best_psnr[1]}  value={best_psnr[0]:.6f}")
    print(f"Best SSIM: epoch={best_ssim[1]}  value={best_ssim[0]:.6f}")


if __name__ == "__main__":
    main()
