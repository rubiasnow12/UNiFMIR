import os, time
import torch
import torch.nn as nn
from model.enlcn import ProjectionUpdater
import model.swinir as module
# 添加
import model.dinoir_v3 as dinoir_v3
import model.multi_dinov3 as multi_dinoir_v3
import model.universal_dino as module_dino

class Model(nn.Module):
    def __init__(self, args, ckp, unimodel=None):
        super(Model, self).__init__()
        print('Making model...')

        self.scale = args.scale[0]
        self.idx_scale = 0
        self.self_ensemble = args.self_ensemble
        self.chop = args.chop
        self.precision = args.precision
        self.cpu = not (torch.cuda.is_available())
        self.device = torch.device('cpu' if self.cpu else 'cuda')
        self.n_GPUs = args.n_GPUs
        self.save_models = args.save_models
        self.args = args
        
        # 如果外部传入了预创建的 unimodel（如 DinoUniModel），直接使用
        if unimodel is not None:
            print('********** Using pre-initialized Universal DINO model ***********')
            self.model = unimodel.to(self.device)
            # 检查外部是否已经注入了 LoRA（通过检查 blocks 的 qkv 层）
            self._lora_already_injected = False
            if hasattr(self.model, 'blocks') and len(self.model.blocks) > 0:
                first_qkv = getattr(self.model.blocks[0].attn, 'qkv', None)
                if first_qkv is not None and hasattr(first_qkv, 'original_linear'):
                    self._lora_already_injected = True
        elif 'proj' in args.model:
            print('********** %s ***********' % args.model.lower())
            self.model = module.make_modelproj(args).to(self.device)
        elif 'SwinIR2t3' in args.model:
            print('********** %s ***********' % args.model.lower())
            self.model = module.make_model2t3(args).to(self.device)
        elif 'UniDINOv3SR' in args.model:  # V3 多任务模型用于 SR 微调
            print('********** %s (DinoUniModelV3 → SR Fine-tuning) ***********' % args.model.lower())
            self.model = dinoir_v3.DinoUniModelV3(
                args, embed_dim=384, dino_depth=12,
                dino_num_heads=6, vit_patch_size=8,
                task_embed_dim=128
            ).to(self.device)
            self._should_inject_lora = False
            self._lora_enabled = False
        elif 'DINOIRv3' in args.model:  # 添加这个分支
            print('********** %s ***********' % args.model.lower())
            self.model = dinoir_v3.make_model(args).to(self.device)
            # 标记：需要在加载权重后注入 LoRA
            self._should_inject_lora = getattr(self.model, '_should_inject_lora', False)
            self._lora_enabled = False  # 此时还未注入
        elif 'MultiDINOv3' in args.model:  # 多尺度 DINOv3 分支
            print('********** %s ***********' % args.model.lower())
            self.model = multi_dinoir_v3.make_model(args).to(self.device)
        elif 'universal' in args.model.lower() or 'dino' in args.model.lower():
            print('********** %s (Universal DINO) ***********' % args.model.lower())
            # 调用 universal_dino.py 里的 make_model 函数
            self.model = module_dino.make_model(args).to(self.device)
        else:
            print('********** %s ***********' % args.model.lower())
            self.model = module.make_model(args).to(self.device)
        
        # ========== 参数训练设置（LoRA 注入前的临时设置） ==========
        # 注意：如果启用 LoRA，真正的冻结会在 load() 之后的 inject_lora() 中完成
        self._should_inject_lora = getattr(self, '_should_inject_lora', False)
        
        # 检查是否外部已经注入了 LoRA（如 mainUi_pretrain.py 中预先注入的情况）
        if getattr(self, '_lora_already_injected', False):
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"--- [LoRA 已注入] 可训练 {trainable_params/1e6:.2f}M / 总参数 {total_params/1e6:.2f}M ({trainable_params/total_params:.1%}) ---")
        elif not self._should_inject_lora:
            # 全参数微调模式：所有参数可训练
            for param in self.model.parameters():
                param.requires_grad = True
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"--- [全参数微调] 可训练 {trainable_params/1e6:.2f}M / 总参数 {total_params/1e6:.2f}M ({trainable_params/total_params:.1%}) ---")
        else:
            # LoRA 模式：先让所有参数可训练，等 load() 后 inject_lora() 再冻结主干
            for param in self.model.parameters():
                param.requires_grad = True
            print("--- [LoRA 模式] 权重加载后将注入 LoRA 并冻结主干 ---")

        # if getattr(args, 'freeze_backbone', False):
        #     print("--- 策略调整: 启用部分微调 (Partial Fine-tuning) ---")
        #     print("--- 冻结前 9 层 (Blocks 0-8)，训练后 3 层 (Blocks 9-11) + Head/Tail ---")
            
        #     # 1. 首先，让所有参数默认可训练 (包括 head, tail, conv_first 等)
        #     for param in self.model.parameters():
        #         param.requires_grad = True

        #     # 2. 遍历参数，精确冻结 DINOv3 内部层
        #     for name, param in self.model.named_parameters():
                
        #         # 冻结位置编码 (Position Embedding)，显微图像尺寸固定的话通常不需要微调
        #         if "rope_embed" in name or "pos_embed" in name:
        #             param.requires_grad = False
                
        #         # 核心逻辑：处理 Transformer Blocks
        #         if name.startswith('blocks.'):# 可能 'model.' 前缀，或者直接 'blocks.' 
        #             try:
        #                 # 解析层号，例如 "model.blocks.5.attn..." -> 5
        #                 # 这里的 split('.') 索引取决于你的参数名结构，通常是第 2 个或第 3 个
        #                 parts = name.split('.')
        #                 layer_idx = -1
        #                 for p in parts:
        #                     if p.isdigit():
        #                         layer_idx = int(p)
        #                         break
                        
        #                 if layer_idx != -1:
        #                     # 策略：冻结前 9 层 (0,1,2,3,4,5,6,7,8)
        #                     if layer_idx < 9:
        #                         param.requires_grad = False
        #                     else:
        #                         # 后 3 层 (9,10,11) 保持 True
        #                         pass 
        #             except ValueError:
        #                 pass
            
        #     # 打印一下验证，确保冻结正确
        #     trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        #     total_params = sum(p.numel() for p in self.model.parameters())
        #     print(f"--- 统计: 可训练参数 {trainable_params/1e6:.2f}M / 总参数 {total_params/1e6:.2f}M ({trainable_params/total_params:.1%}) ---")
# # === 全量微调 (Full Fine-tuning) ===
#         # 所有参数可训练，仅冻结位置编码
#         for param in self.model.parameters():
#             param.requires_grad = True
        
#         # 冻结位置编码 (RoPE/Position Embedding)
#         for name, param in self.model.named_parameters():
#             if "rope_embed" in name or "pos_embed" in name:
#                 param.requires_grad = False
        
#         trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
#         total_params = sum(p.numel() for p in self.model.parameters())
#         print(f"--- 全量微调 (冻结位置编码): 可训练 {trainable_params/1e6:.2f}M / 总参数 {total_params/1e6:.2f}M ({trainable_params/total_params:.1%}) ---")

        
        
        self.proj_updater = ProjectionUpdater(self.model, feature_redraw_interval=640)
        if args.precision == 'half':
            self.model.half()

        self.load(os.path.join('experiment', args.save), modelpath=args.modelpath, resume=args.resume)
        
        # ========== 关键：在加载权重后注入 LoRA（仅当未在 load() 中提前注入时） ==========
        if getattr(self, '_should_inject_lora', False) and hasattr(self.model, 'inject_lora'):
            # 检查是否已经在 load() 中注入过（针对包含 LoRA 权重的 checkpoint）
            if not getattr(self.model, '_lora_injected', False):
                print("\n--- [LoRA] 权重加载完成，正在注入 LoRA... ---")
                self.model.inject_lora()
                self._lora_enabled = True
            
            # 重新统计可训练参数
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"--- [LoRA] 注入后参数统计: 可训练 {trainable_params/1e6:.2f}M / 总参数 {total_params/1e6:.2f}M ({trainable_params/total_params:.1%}) ---\n")

        if not args.cpu and args.n_GPUs > 1:
            self.model = nn.DataParallel(self.model, range(args.n_GPUs))

        print(self.model, file=ckp.log_file)

    def forward(self, x, *args, **kwargs):
        # 提取任务 ID（如果有）
        task_id = args[0] if len(args) > 0 else kwargs.get('tsk', 0)
        
        # UniDINOv3SR: 固定使用 task=1 (SR)，并设置 scale=2
        if 'UniDINOv3SR' in self.args.model:
            task_id = 1
            self.model.scale = self.scale  # 使用 args.scale[0] 的值
        
        # 任务5 (Volume) 使用特殊的 11x 上采样，不适合 forward_chop
        # 直接调用模型
        if self.args.chop and task_id != 5:
            result = self.forward_chop(x, task_id=task_id, min_size=160000)
        else:
            result = self.model(x, task_id)
        return result

    def get_model(self):
        return self.model

    def state_dict(self, **kwargs):
        target = self.get_model()
        return target.state_dict(**kwargs)

    def save(self, apath, epoch, is_best=False, save_numbered=True):
        target = self.get_model()
        os.makedirs(apath, exist_ok=True)
        torch.save(
            target.state_dict(), 
            os.path.join(apath, 'model_latest.pt')
        )
        if is_best:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_best.pt')
            )
        
        if self.save_models and save_numbered:
            torch.save(
                target.state_dict(),
                os.path.join(apath, 'model_{}.pt'.format(epoch))
            )

    def load(self, apath, modelpath='.', resume=-1):
        if self.cpu:
            kwargs = {'map_location': lambda storage, loc: storage}
        else:
            kwargs = {}

        if resume == -1:
            print('Load Model from ', os.path.join(apath, 'model_latest.pt'))
            self.model.load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_latest.pt'),
                    **kwargs
                ),
                strict=True
            )
        elif resume == -2:
            m = os.path.join(apath, 'model_best.pt')
            print('Load Model from ', m)
            self.model.load_state_dict(torch.load(m, **kwargs), strict=True)
        elif resume < -2:
            m = os.path.join(apath, 'model_best%d.pt' % -resume)
            print('Load Model from ', m)
            self.model.load_state_dict(torch.load(m, **kwargs), strict=True)
        elif resume == 0 and modelpath != '.':
            # print('Loading UNet model from {}'.format(modelpath))
            # if ('2stg_enlcn' in self.args.model) or ('2stg_proj_care' in self.args.model):
            #     self.model.project.load_state_dict(torch.load(modelpath, **kwargs), strict=True)
            # elif ('stage2' in self.args.model) or ('_unfix' in self.args.model):
            #     self.model.conv_first0.load_state_dict(torch.load(modelpath, **kwargs), strict=True)
            # else:
            #     self.get_model().load_state_dict(torch.load(modelpath, **kwargs), strict=True)
            print('Loading pretrained model from {}'.format(modelpath))
            pretrained_dict = torch.load(modelpath, **kwargs)
            
            # ========== 检测 checkpoint 是否包含 LoRA 权重 ==========
            # 如果 checkpoint 包含 'original_linear' 或 'lora_A/lora_B'，说明保存时已注入 LoRA
            has_lora_in_checkpoint = any(
                'original_linear' in k or 'lora_A' in k or 'lora_B' in k 
                for k in pretrained_dict.keys()
            )
            
            if has_lora_in_checkpoint and getattr(self, '_should_inject_lora', False):
                # checkpoint 包含 LoRA 权重，需要先注入 LoRA 结构再加载
                print(">>> [检测到] Checkpoint 包含 LoRA 权重，先注入 LoRA 结构...")
                if hasattr(self.model, 'inject_lora') and not getattr(self.model, '_lora_injected', False):
                    self.model.inject_lora()
                    self._lora_enabled = True
                    # 标记已注入，避免后续重复注入
                    self._should_inject_lora = False
            
            model_dict = self.model.state_dict()

            # 通用处理：形状匹配才加载；对 conv_first.weight 做特殊适配
            pretrained_dict_filtered = {}
            shape_mismatch_keys = []

            # 特判 conv_first.weight 的 Cin 适配（1↔3）
            if 'conv_first.weight' in pretrained_dict and 'conv_first.weight' in model_dict:
                w_src = pretrained_dict['conv_first.weight']
                w_tgt = model_dict['conv_first.weight']
                if w_src.shape != w_tgt.shape:
                    if w_src.dim() == 4 and w_tgt.dim() == 4 and w_src.shape[2:] == w_tgt.shape[2:]:
                        if w_src.shape[1] == 1 and w_tgt.shape[1] == 3:
                            # 1→3：复制并平均，保持幅度
                            w_adapt = w_src.repeat(1, 3, 1, 1) / 3.0
                            print("✅ Adapt conv_first.weight: (out,1,k,k) → (out,3,k,k)")
                            pretrained_dict_filtered['conv_first.weight'] = w_adapt
                        elif w_src.shape[1] == 3 and w_tgt.shape[1] == 1:
                            # 3→1：通道平均
                            w_adapt = w_src.mean(dim=1, keepdim=True)
                            print("✅ Adapt conv_first.weight: (out,3,k,k) → (out,1,k,k)")
                            pretrained_dict_filtered['conv_first.weight'] = w_adapt
                        else:
                            shape_mismatch_keys.append('conv_first.weight')
                    else:
                        shape_mismatch_keys.append('conv_first.weight')
                else:
                    pretrained_dict_filtered['conv_first.weight'] = w_src

            # 其余键：仅接受形状完全匹配的
            for k, v in pretrained_dict.items():
                if k == 'conv_first.weight':
                    continue
                if k in model_dict and v.shape == model_dict[k].shape:
                    pretrained_dict_filtered[k] = v
                elif k in model_dict:
                    shape_mismatch_keys.append(k)

            missing_keys = set(model_dict.keys()) - set(pretrained_dict_filtered.keys())
            unexpected_keys = set(pretrained_dict.keys()) - set(model_dict.keys())

            print(f'✅ Successfully matched {len(pretrained_dict_filtered)}/{len(model_dict)} layers')
            if missing_keys:
                print(f'⚠️  Missing keys (randomly initialized): {len(missing_keys)}')
                print(f'    Example: {list(missing_keys)[:5]}')
            if shape_mismatch_keys:
                print(f'⚠️  Shape mismatched keys (skipped or adapted): {len(shape_mismatch_keys)}')
                print(f'    Example: {shape_mismatch_keys[:5]}')
            if unexpected_keys:
                print(f'⚠️  Unexpected keys in checkpoint (ignored): {len(unexpected_keys)}')
                print(f'    Example: {list(unexpected_keys)[:5]}')

            # 加载过滤后的字典（不再会触发 size mismatch）
            self.get_model().load_state_dict({**model_dict, **pretrained_dict_filtered}, strict=False)
            print('✅ Pretrained weights loaded. Trainable params ready for finetuning.')

        elif (resume > 0) and os.path.exists(os.path.join(apath, 'model_{}.pt'.format(resume))):
            print('Load Model from ', os.path.join(apath, 'model_{}.pt'.format(resume)))
            self.get_model().load_state_dict(
                torch.load(
                    os.path.join(apath, 'model_{}.pt'.format(resume)),
                    **kwargs
                ),
                strict=True
            )
        else:
            print('!!!!!!!!  Not Load Model  !!!!!!')
        
    def load_network(self, load_path, strict=True, param_key=None):  # 'params'params_ema
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(load_net, strict=strict)
        print(f'Loading {self.model.__class__.__name__} model from {load_path}.')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        # logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            print('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                print('warning', f'  {v}')
            print('warning', 'Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                print('warning', f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    print('warning', f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def forward_chop(self, x, task_id=0, shave=10, min_size=6000):
        scale = self.scale
        n_GPUs = min(self.n_GPUs, 4)
        b, _, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        h_size, w_size = h_half + 16, w_half + 16
        h_size += 8
        w_size += 8

        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
        # print('lr_list[0].size = ', lr_list[0].size())

        if w_size * h_size < min_size:
            sr_list = []
            sr_stg1_list = []  # 用于存储第一阶段输出（任务4和5）
            eacm_loss_list = []  # 用于存储 EACM 对比损失（任务4）
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                result = self.model(lr_batch, task_id)
                # 检查是否是 tuple（任务4返回3个输出，任务5返回2个输出）
                if isinstance(result, tuple):
                    if len(result) == 3:
                        # task 4: (x2d, sr, eacm_loss)
                        sr_stg1_list.extend(result[0].chunk(n_GPUs, dim=0))
                        sr_list.extend(result[1].chunk(n_GPUs, dim=0))
                        eacm_loss_list.append(result[2])
                    else:
                        # task 5: (stg1, sr)
                        sr_stg1_list.extend(result[0].chunk(n_GPUs, dim=0))
                        sr_list.extend(result[1].chunk(n_GPUs, dim=0))
                else:
                    sr_list.extend(result.chunk(n_GPUs, dim=0))
        else:
            results = [
                self.forward_chop(patch, task_id=task_id, shave=shave, min_size=min_size)\
                for patch in lr_list]
            # 分离可能的 tuple 结果
            if isinstance(results[0], tuple):
                if len(results[0]) == 3:
                    sr_stg1_list = [r[0] for r in results]
                    sr_list = [r[1] for r in results]
                    eacm_loss_list = [r[2] for r in results]
                else:
                    sr_stg1_list = [r[0] for r in results]
                    sr_list = [r[1] for r in results]
                    eacm_loss_list = []
            else:
                sr_list = results
                sr_stg1_list = []
                eacm_loss_list = []

        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale

        output = x.new(b, sr_list[0].shape[1], h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

        # 如果有第一阶段输出，组合第一阶段的 output
        if sr_stg1_list:
            output_stg1 = x.new(b, sr_stg1_list[0].shape[1], h, w)
            output_stg1[:, :, 0:h_half, 0:w_half] \
                = sr_stg1_list[0][:, :, 0:h_half, 0:w_half]
            output_stg1[:, :, 0:h_half, w_half:w] \
                = sr_stg1_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
            output_stg1[:, :, h_half:h, 0:w_half] \
                = sr_stg1_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
            output_stg1[:, :, h_half:h, w_half:w] \
                = sr_stg1_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
            # 如果有 EACM loss（task 4），取平均并返回
            if eacm_loss_list:
                avg_eacm_loss = sum(eacm_loss_list) / len(eacm_loss_list)
                return output_stg1, output, avg_eacm_loss
            return output_stg1, output

        return output

    def forward_chopProj(self, x, shave=10, min_size=1e60):
        scale = self.scale
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()

        h_half, w_half = h // 2, w // 2
        h_size, w_size = h_half + shave, w_half + shave
        h_size += 16-h_size % 16  # 272
        w_size += 16-w_size % 16  #
        
        lr_list = [
                x[:, :, 0:h_size, 0:w_size],  # 272 360
                x[:, :, 0:h_size, (w - w_size):w],  # 272 360
                x[:, :, (h - h_size):h, 0:w_size],  # 272 360
                x[:, :, (h - h_size):h, (w - w_size):w]]  # 272 360
            
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                # print('Proj Output')
                if '2stg_enlcn' in self.args.model:
                    sr_batchgvt, sr_batch = self.model(lr_batch)
                
                # print('sr_batch.size() = ', sr_batch.size())
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [self.forward_chopProj(patch, shave=shave, min_size=min_size) for patch in lr_list]
    
        # print('1  sr_list[0].size ', sr_list[0].size())
        h_half, w_half = scale * h_half, scale * w_half
        h_size, w_size = scale * h_size, scale * w_size
        shave *= scale
    
        # print('1 x.size', x.size())
        output = x.new(b, 1, h, w)
        # print('1 output', output.shape)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    
        return output

    def forward_chop2to3(self, x, shave=2, min_size=120000):
        scale = 11
        n_GPUs = min(self.n_GPUs, 4)
        b, c, h, w = x.size()
        h_half, w_half = h // 2, w // 2
        
        h_size, w_size = h_half + shave, w_half + shave
      
        h_size += 8-h_size % 8
        w_size += 8-w_size % 8
        
        print('0 x.size = ', x.size(), 'h/w_size = ', h_size, w_size)
        
        lr_list = [
            x[:, :, 0:h_size, 0:w_size],
            x[:, :, 0:h_size, (w - w_size):w],
            x[:, :, (h - h_size):h, 0:w_size],
            x[:, :, (h - h_size):h, (w - w_size):w]]
    
        if w_size * h_size < min_size:
            sr_list = []
            for i in range(0, 4, n_GPUs):
                lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
                if self.idx_scale < 0:
                    print('UNet 2to3 output')
                    sr_batch, _ = self.model(lr_batch)
                else:
                    print('SWinIR 2to3 output')
                    sr_batchu, sr_batch = self.model(lr_batch)
                sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
        else:
            sr_list = [self.forward_chop2to3(patch, shave=shave, min_size=min_size) for patch in lr_list]
    
        # print('1  sr_list[0].size ', sr_list[0].size())  # [1, 61, 176, 176]
        h, w = scale * h, scale * w
        h_half, w_half = scale * h_half, scale * w_half  # 88, 88
        h_size, w_size = scale * h_size, scale * w_size  # 352 352
        
        output = x.new(b, 61, h, w)
        output[:, :, 0:h_half, 0:w_half] \
            = sr_list[0][:, :, 0:h_half, 0:w_half]
        output[:, :, 0:h_half, w_half:w] \
            = sr_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
        output[:, :, h_half:h, 0:w_half] \
            = sr_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
        output[:, :, h_half:h, w_half:w] \
            = sr_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]
    
        return output

    def forward_x8(self, x, forward_function):
        def _transform(v, op):
            # if self.precision != 'single': v = v.float()

            v2np = v.data.cpu().numpy()
            if op == 'v':
                tfnp = v2np[:, :, :, ::-1].copy()
            elif op == 'h':
                tfnp = v2np[:, :, ::-1, :].copy()
            elif op == 't':
                tfnp = v2np.transpose((0, 1, 3, 2)).copy()

            ret = torch.Tensor(tfnp).to(self.device)
            # if self.precision == 'half': ret = ret.half()

            return ret

        lr_list = [x]
        for tf in 'v', 'h', 't':
            lr_list.extend([_transform(t, tf) for t in lr_list])

        sr_list = [forward_function(aug) for aug in lr_list]
        for i in range(len(sr_list)):
            if i > 3:
                sr_list[i] = _transform(sr_list[i], 't')
            if i % 4 > 1:
                sr_list[i] = _transform(sr_list[i], 'h')
            if (i % 4) % 2 == 1:
                sr_list[i] = _transform(sr_list[i], 'v')

        output_cat = torch.cat(sr_list, dim=0)
        output = output_cat.mean(dim=0, keepdim=True)

        return output