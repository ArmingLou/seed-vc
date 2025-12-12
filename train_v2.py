import os
import sys
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import random
import numpy as np
import librosa
import yaml
import argparse
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import glob
import time
from tqdm import tqdm
import shutil
import accelerate
from optimizers import build_optimizer
from data.ft_dataset import build_ft_dataloader
import hydra
from omegaconf import DictConfig
from contextlib import nullcontext

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.logging import get_logger
import datetime


class Trainer:
    def __init__(
            self,
            config_path,
            pretrained_cfm_ckpt_path,
            pretrained_ar_ckpt_path,
            data_dir,
            run_name,
            batch_size=0,
            num_workers=0,
            steps=1000,
            save_interval=500,
            max_epochs=1000,
            train_cfm=False,
            train_ar=False,
            mixed_precision=None,
            fp16=False,
            val_dataset_dir=None,
            patience=20,
            validation_interval=50,
            min_lr=1e-7,
            lr_adjust_interval=50,
            initial_lr=1e-5,
            warmup_steps=1000,
            # 添加新的知识蒸馏参数
            distill_ar=0.0,
            distill_cfm=0.0,
            grad_clip_norm=1.0,
            distill_temperature=1.0,
            resume_lr=0.0,  # 添加resume_lr参数，默认值为0.0
            language=None,  # 添加language参数，默认值为None
        ):
        self.config_path = config_path
        self.mixed_precision = mixed_precision
        self.requested_fp16 = fp16  # 保存用户请求的fp16设置
        # 保存新的知识蒸馏参数
        self.distill_ar_weight = distill_ar
        self.distill_cfm_weight = distill_cfm
        self.use_distill_ar = distill_ar > 0.0
        self.use_distill_cfm = distill_cfm > 0.0
        # 保存梯度裁剪参数
        self.grad_clip_norm = grad_clip_norm
        # 保存蒸馏温度参数
        self.distill_temperature = distill_temperature
        # 保存预训练检查点路径（同时用于预训练和蒸馏）
        self.pretrained_ar_ckpt_path = pretrained_ar_ckpt_path
        self.pretrained_cfm_ckpt_path = pretrained_cfm_ckpt_path
        # 保存resume_lr参数
        self.resume_lr = resume_lr
        # 保存language参数
        self.language = language
        
        # Check FORCE_CPU environment variable
        force_cpu = os.environ.get('FORCE_CPU', '0') == '1'
        
        # If XPU is available, override force_cpu setting
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            force_cpu = False

        # Load configuration
        self.config = yaml.safe_load(open(config_path))

        # Setup logging directory
        # 使用数据集目录名作为后缀，保持与V1版本一致的输出目录结构
        dataset_name = os.path.basename(os.path.normpath(data_dir))
        version_run_name = f"{run_name}_{dataset_name}"
        self.log_dir = os.path.join("runs", version_run_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))

        # Setup accelerator with fp16 error handling
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True, broadcast_buffers=False)
        try:
            self.accelerator = Accelerator(
                project_dir=self.log_dir,
                split_batches=True,
                kwargs_handlers=[ddp_kwargs],
                mixed_precision=self.mixed_precision,
                cpu=force_cpu  # 根据FORCE_CPU环境变量决定是否使用CPU
            )
        except Exception as e:
            print(f"Warning: Failed to initialize accelerator with mixed precision '{self.mixed_precision}': {e}")
            # 如果fp16初始化失败，回退到no mixed precision
            if self.mixed_precision == 'fp16':
                print("Falling back to no mixed precision...")
                self.mixed_precision = 'no'
                self.requested_fp16 = False  # 更新请求的fp16设置
                self.accelerator = Accelerator(
                    project_dir=self.log_dir,
                    split_batches=True,
                    kwargs_handlers=[ddp_kwargs],
                    mixed_precision=self.mixed_precision,
                    cpu=force_cpu
                )
            else:
                raise e  # 如果不是fp16相关的错误，重新抛出异常
        self.device = self.accelerator.device

        # 早停机制参数
        self.patience = patience
        self.validation_interval = validation_interval
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        # 初始化损失缩放因子
        self.loss_scaling_factors = {
            'ar': 1.0,
            'cfm': 1.0,
            'distill_ar': 0.0,
            'distill_cfm': 0.0
        }
        
        # 学习率调度相关参数
        self.initial_lr = initial_lr  # 初始学习率
        self.min_lr = min_lr      # 最小学习率
        self.lr_adjust_interval = lr_adjust_interval  # 学习率调整间隔
        self.warmup_steps = warmup_steps  # 学习率预热步数
        self.best_train_loss = float('inf')  # 用于学习率调度的最佳训练损失
        self.switched_to_val_scheduler = False  # 学习率调度器切换状态

        # 初始化ema_loss和其他相关属性
        self.ema_loss = 0
        self.loss_smoothing_rate = 0.99

        # Initialize training parameters
        self._init_dataloader(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            spect_params=self.config['mel_fn'],
            sr=self.config['sr'],
        )
        
        # 初始化验证集数据加载器（如果提供了验证集路径）
        self.val_dataloader = None
        if val_dataset_dir:
            self.val_dataloader = build_ft_dataloader(
                val_dataset_dir,
                self.config['mel_fn'],
                self.config['sr'],
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,  # 验证集不打乱
            )

        # 训练状态参数
        self.iters = 0
        self.epoch = 0
        self.log_interval = 10
        self.max_steps = steps
        self.save_interval = save_interval
        self.max_epochs = max_epochs
        self.training_completed = False  # 训练完成标志位
        
        # Initialize models and optimizers
        self._init_models(train_cfm=train_cfm, train_ar=train_ar)

        # Load checkpoint if available
        self._load_checkpoint(pretrained_cfm_ckpt_path, pretrained_ar_ckpt_path)
        
        # 初始化教师模型（如果启用了知识蒸馏）
        self.teacher_model = None
        # 根据新的参数决定是否加载教师模型
        # 只有在对应模型需要训练且启用了蒸馏时才加载教师模型
        if (self.use_distill_ar and self.train_ar) or (self.use_distill_cfm and self.train_cfm):
            self.set_teacher_model()
    
    def compute_kl_distill_loss(self, student_logits, teacher_logits, temperature=1.0):
        """使用KL散度计算蒸馏损失，支持温度参数"""
        # 避免数值不稳定，添加小的epsilon
        epsilon = 1e-7
        
        # 软化分布
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        # 添加epsilon防止log(0)
        student_probs = torch.clamp(student_probs, epsilon, 1.0 - epsilon)
        teacher_probs = torch.clamp(teacher_probs, epsilon, 1.0 - epsilon)
        
        # 计算KL散度
        kl_loss = F.kl_div(torch.log(student_probs), teacher_probs, reduction='batchmean')
        
        # 温度系数缩放
        return kl_loss * (temperature ** 2)
    
    def _compute_initial_loss_scaling_factors(self):
        """计算初始损失缩放因子，使各损失组件按指定比例调整"""
        # 只保存初次计算的值，避免续训练重置缩放因子，也避免被 0 覆盖原有非 0 的值，确保缩放因子即使断点续训练从头到尾保持最早的值，而不是续训练时新计算。
        distill_ar_need_comp = False
        distill_cfm_need_comp = False
        if self.loss_scaling_factors['distill_cfm'] == 0.0:
            distill_cfm_need_comp = True
        else:
            print("断点续训练恢复，将使用继承的蒸馏权重，忽视参数中 --distill-cfm 重新指定的值，只可以关闭蒸馏。")
        if self.loss_scaling_factors['distill_ar'] == 0.0:
            distill_ar_need_comp = True
        else:
            print("断点续训练恢复，将使用继承的蒸馏权重，忽视参数中 --distill-ar 重新指定的值，只可以关闭蒸馏。")
            
        if self.accelerator.is_main_process:
            print("计算初始损失缩放因子...")
            
        # 临时设置模型为评估模式以计算初始损失
        self.model.eval()
        # 如果有教师模型且使用蒸馏，也确保其处于评估模式
        if self.teacher_model is not None and (self.use_distill_ar or self.use_distill_cfm):
            self.teacher_model.eval()
            
        try:
            # 获取一个样本批次计算初始损失
            sample_batch = next(iter(self.train_dataloader))
            sample_batch = [b.to(self.device) for b in sample_batch]
            
            with torch.no_grad():
                # 解包样本批次
                waves, mels, wave_lens, mel_lens = sample_batch
                # Resample to 16kHz for ASR models
                waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
                wave_lengths_16k = (wave_lens.float() * 16000 / self.sr).long()
                
                # 计算各损失组件的原始值
                original_loss_ar, original_loss_cfm = self.model(
                    waves_16k.to(self.device),
                    mels.to(self.device),
                    wave_lengths_16k.to(self.device),
                    mel_lens.to(self.device),
                    forward_ar=self.train_ar,
                    forward_cfm=self.train_cfm,
                )
                
                original_ar_loss_val = original_loss_ar.item() if isinstance(original_loss_ar, torch.Tensor) else 0.0
                original_cfm_loss_val = original_loss_cfm.item() if isinstance(original_loss_cfm, torch.Tensor) else 0.0
                
                # 计算蒸馏损失（如果有教师模型）
                original_distill_ar_loss = 0.0
                original_distill_cfm_loss = 0.0
                
                if self.teacher_model is not None and (self.use_distill_ar or self.use_distill_cfm):
                    # 使用教师模型生成目标输出
                    # 只计算需要蒸馏的模型部分的输出，提高效率
                    teacher_forward_ar = self.train_ar and self.use_distill_ar
                    teacher_forward_cfm = self.train_cfm and self.use_distill_cfm
                    
                    teacher_loss_ar, teacher_loss_cfm = self.teacher_model(
                        waves_16k.to(self.device),
                        mels.to(self.device),
                        wave_lengths_16k.to(self.device),
                        mel_lens.to(self.device),
                        forward_ar=teacher_forward_ar,
                        forward_cfm=teacher_forward_cfm,
                )
                    
                    # 计算学生模型和教师模型输出之间的蒸馏损失
                    # 只有在对应模型被训练且启用了蒸馏时才计算蒸馏损失
                    if self.train_cfm and self.use_distill_cfm:
                        # CFM蒸馏损失
                        # 确保loss_cfm和teacher_loss_cfm都是张量且形状匹配
                        if isinstance(original_loss_cfm, torch.Tensor) and isinstance(teacher_loss_cfm, torch.Tensor):
                            if original_loss_cfm.size() == teacher_loss_cfm.size():
                                # 确保数据类型一致
                                if original_loss_cfm.dtype != teacher_loss_cfm.dtype:
                                    if self.accelerator.is_main_process:
                                        print(f"警告: CFM蒸馏损失数据类型不一致 - student: {original_loss_cfm.dtype}, teacher: {teacher_loss_cfm.dtype}")
                                    teacher_loss_cfm = teacher_loss_cfm.to(original_loss_cfm.dtype)
                                # 使用KL散度计算蒸馏损失，添加温度参数支持
                                kl_loss = self.compute_kl_distill_loss(original_loss_cfm, teacher_loss_cfm.detach(), temperature=self.distill_temperature)
                                original_distill_cfm_loss = kl_loss.item()
                            else:
                                if self.accelerator.is_main_process:
                                    print(f"Warning: Shape mismatch in CFM distillation loss - student: {original_loss_cfm.size()}, teacher: {teacher_loss_cfm.size()}")
                        else:
                            if self.accelerator.is_main_process:
                                print(f"Warning: Type mismatch in CFM distillation loss - student: {type(original_loss_cfm)}, teacher: {type(teacher_loss_cfm)}")
                    if self.train_ar and self.use_distill_ar:
                        # AR蒸馏损失
                        # 确保loss_ar和teacher_loss_ar都是张量且形状匹配
                        if isinstance(original_loss_ar, torch.Tensor) and isinstance(teacher_loss_ar, torch.Tensor):
                            if original_loss_ar.size() == teacher_loss_ar.size():
                                # 确保数据类型一致
                                if original_loss_ar.dtype != teacher_loss_ar.dtype:
                                    if self.accelerator.is_main_process:
                                        print(f"警告: AR蒸馏损失数据类型不一致 - student: {original_loss_ar.dtype}, teacher: {teacher_loss_ar.dtype}")
                                    teacher_loss_ar = teacher_loss_ar.to(original_loss_ar.dtype)
                                # 使用KL散度计算蒸馏损失，添加温度参数支持
                                kl_loss = self.compute_kl_distill_loss(original_loss_ar, teacher_loss_ar.detach(), temperature=self.distill_temperature)
                                original_distill_ar_loss = kl_loss.item()
                            else:
                                if self.accelerator.is_main_process:
                                    print(f"Warning: Shape mismatch in AR distillation loss - student: {original_loss_ar.size()}, teacher: {teacher_loss_ar.size()}")
                        else:
                            if self.accelerator.is_main_process:
                                print(f"Warning: Type mismatch in AR distillation loss - student: {type(original_loss_ar)}, teacher: {type(teacher_loss_ar)}")
                
                
                # 计算总原始损失
                original_total_loss_ar = original_ar_loss_val + original_distill_ar_loss          
                original_total_loss_cfm = original_cfm_loss_val + original_distill_cfm_loss                
                original_total_loss = original_total_loss_ar + original_total_loss_cfm               
                if self.accelerator.is_main_process:
                    print(f"原始损失值:")
                    print(f"  AR损失: {original_ar_loss_val:.6f}")
                    print(f"  CFM损失: {original_cfm_loss_val:.6f}")
                    print(f"  AR蒸馏损失: {original_distill_ar_loss:.6f}")
                    print(f"  CFM蒸馏损失: {original_distill_cfm_loss:.6f}")
                    print(f"  总损失: {original_total_loss:.6f}")
                
                # 目标比例 1:1:self.distill_ar_weight:self.distill_cfm_weight
                target_ar_ratio = 1.0
                target_cfm_ratio = 1.0
                # 如果原始损失值为0，则对应的目标比例也应为0
                # 如果distill权重为0或原始蒸馏损失值为0，则对应的目标蒸馏比例应为0
                target_distill_ar_ratio = 0.0 if original_distill_ar_loss == 0 else self.distill_ar_weight
                target_distill_cfm_ratio = 0.0 if original_distill_cfm_loss == 0 else self.distill_cfm_weight
                
                # AR模型和CFM模型分别独立计算目标比例
                # AR模型组：target_ar_ratio 和 target_distill_ar_ratio
                # ar_group_ratio_sum = sum([ratio for ratio in [target_ar_ratio, target_distill_ar_ratio] if ratio > 0])
                # CFM模型组：target_cfm_ratio 和 target_distill_cfm_ratio
                # cfm_group_ratio_sum = sum([ratio for ratio in [target_cfm_ratio, target_distill_cfm_ratio] if ratio > 0])
                
                # 根据各自组内的比例分配目标损失值
                # target_ar_loss = original_total_loss_ar * target_ar_ratio / ar_group_ratio_sum if ar_group_ratio_sum > 0 and target_ar_ratio > 0 else 0
                # target_distill_ar_loss = original_total_loss_ar * target_distill_ar_ratio / ar_group_ratio_sum if ar_group_ratio_sum > 0 and target_distill_ar_ratio > 0 else 0
                
                # target_cfm_loss = original_total_loss_cfm * target_cfm_ratio / cfm_group_ratio_sum if cfm_group_ratio_sum > 0 and target_cfm_ratio > 0 else 0
                # target_distill_cfm_loss = original_total_loss_cfm * target_distill_cfm_ratio / cfm_group_ratio_sum if cfm_group_ratio_sum > 0 and target_distill_cfm_ratio > 0 else 0
                
                
                # 调整为永远以 main 的因子为 1.0
                # arScale = target_ar_loss / original_ar_loss_val if original_ar_loss_val > 0 else 1.0
                # target_ar_loss = original_ar_loss_val
                # target_distill_ar_loss = target_distill_ar_loss / arScale
                # cfmScale = target_cfm_loss / original_cfm_loss_val if original_cfm_loss_val > 0 else 1.0
                # target_cfm_loss = original_cfm_loss_val
                # target_distill_cfm_loss = target_distill_cfm_loss / cfmScale
                
                # 计算缩放因子
                self.loss_scaling_factors['ar'] = 1.0
                self.loss_scaling_factors['cfm'] = 1.0
                if distill_ar_need_comp:
                    self.loss_scaling_factors['distill_ar'] = target_ar_loss * target_distill_ar_ratio / original_distill_ar_loss  if original_distill_ar_loss > 0 else 0.0
                if distill_cfm_need_comp:
                    self.loss_scaling_factors['distill_cfm'] = target_cfm_loss * target_distill_cfm_ratio / original_distill_cfm_loss if original_distill_cfm_loss > 0 else 0.0
                
                if self.accelerator.is_main_process:
                    print(f"缩放因子:")
                    print(f"  AR损失缩放因子: {self.loss_scaling_factors['ar']:.6f}")
                    print(f"  CFM损失缩放因子: {self.loss_scaling_factors['cfm']:.6f}")
                    print(f"  {'计算的' if distill_ar_need_comp else '继承的'} AR蒸馏损失缩放因子: {self.loss_scaling_factors['distill_ar']:.6f}")
                    print(f"  {'计算的' if distill_cfm_need_comp else '继承的'} CFM蒸馏损失缩放因子: {self.loss_scaling_factors['distill_cfm']:.6f}")
                    
                # 用新计算 或 继承的 缩放因子 重新计算 目标损失值
                target_ar_loss = original_ar_loss_val
                target_cfm_loss = original_cfm_loss_val
                target_distill_ar_loss = original_distill_ar_loss * self.loss_scaling_factors['distill_ar']
                target_distill_cfm_loss = original_distill_cfm_loss * self.loss_scaling_factors['distill_cfm']
                
                target_total_loss = target_ar_loss + target_distill_ar_loss + target_cfm_loss + target_distill_cfm_loss
                
                if self.accelerator.is_main_process:
                    print(f"目标损失值:")
                    print(f"  AR损失: {target_ar_loss:.6f}")
                    print(f"  CFM损失: {target_cfm_loss:.6f}")
                    print(f"  AR蒸馏损失: {target_distill_ar_loss:.6f}")
                    print(f"  CFM蒸馏损失: {target_distill_cfm_loss:.6f}")
                    print(f"  总损失: {target_total_loss:.6f}")
                
                # 初始化ema_loss为总原始损失值
                if self.accelerator.is_main_process:
                    if self.ema_loss is None or self.ema_loss == 0:
                        self.ema_loss = target_total_loss
                    
        except Exception as e:
            if self.accelerator.is_main_process:                
                print(f"计算初始损失缩放因子时出错: {e}")
            # 使用默认缩放因子
            # self.loss_scaling_factors = {
            #     'ar': 1.0,
            #     'cfm': 1.0,
            #     'distill_ar': 0.0,
            #     'distill_cfm': 0.0
            # }
        
        # 恢复模型训练模式
        self.model.train()
        if self.teacher_model is not None and (self.use_distill_ar or self.use_distill_cfm):
            self.teacher_model.eval()

    def _init_dataloader(self, data_dir, batch_size, num_workers, spect_params, sr):
        self.spect_params = spect_params
        self.sr = sr
        # Initialize dataloader
        self.train_dataloader = build_ft_dataloader(
            data_dir,
            spect_params,
            self.sr,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        # 保存数据集引用以便在每个epoch设置随机索引
        self.train_dataset = self.train_dataloader.dataset

    def _init_models(self, train_cfm=False, train_ar=False):
        """Initialize models and optimizers"""
        assert train_cfm or train_ar, "At least one model should be trained"
        self.train_cfm = train_cfm
        self.train_ar = train_ar
        # Initialize main model
        self._init_main_model(train_cfm=train_cfm, train_ar=train_ar)

        # Initialize optimizers
        self._init_optimizers()


    def _init_main_model(self, train_cfm=True, train_ar=False):
        """Initialize the main model"""
        with self.accelerator.main_process_first():
            cfg = DictConfig(self.config)
            self.model = hydra.utils.instantiate(cfg).to(self.device)
            for p in self.model.parameters():
                p.requires_grad = False
            if train_cfm:
                for p in self.model.cfm.parameters():
                    p.requires_grad = True
                for p in self.model.cfm_length_regulator.parameters():
                    p.requires_grad = True
            if train_ar:
                for p in self.model.ar.parameters():
                    p.requires_grad = True
                for p in self.model.ar_length_regulator.parameters():
                    p.requires_grad = True


    def _init_optimizers(self):
        """Initialize optimizers and schedulers"""
        from optimizers import build_single_optimizer
        self.optimizer, self.scheduler = build_single_optimizer(
            self.model,
            lr=self.initial_lr,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            min_lr=self.min_lr
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)
        self.scheduler = self.accelerator.prepare(self.scheduler)
        
        # 保存warmup_steps到局部变量，避免Lambda函数捕获错误的值
        warmup_steps = self.warmup_steps
        # 创建LambdaLR用于预热
        # 使用默认参数确保捕获正确的warmup_steps值
        warmup_lambda = lambda step, ws=warmup_steps: min(1.0, float(step + 1) / float(ws)) if step < ws else 1.0
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=warmup_lambda
        )
        
        # 创建余弦退火调度器
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.max_steps - self.warmup_steps,
            eta_min=self.min_lr
        )

    def _find_checkpoint(self, name_pattern, max_keep=1):
        """Find checkpoint files in the specified directory"""
        available_checkpoints = glob.glob(os.path.join(self.log_dir, name_pattern))
        if len(available_checkpoints) > max_keep - 1:
            # find the checkpoint that has the highest step number
            latest_checkpoint = max(
                available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            earliest_checkpoint = min(
                available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            # delete the earliest checkpoint
            if (
                    earliest_checkpoint != latest_checkpoint
                    and self.accelerator.is_main_process
                    and len(available_checkpoints) > max_keep
            ):
                os.remove(earliest_checkpoint)
                print(f"Removed {earliest_checkpoint}")
            return latest_checkpoint
        else:
            return None

    def _load_checkpoint(self, pretrained_cfm_ckpt_path, pretrained_ar_ckpt_path):
        """Load checkpoint if available"""
        # 只有在需要训练对应模型时才查找和加载检查点
        # 对于预训练检查点，只有在对应模型需要训练时才使用
        cfm_checkpoint_path = None
        ar_checkpoint_path = None
        
        # 首先查找本地检查点（用于断点续训）
        local_cfm_checkpoint = self._find_checkpoint("CFM_epoch_*_step_*.pth", max_keep=1) if self.train_cfm else None
        local_ar_checkpoint = self._find_checkpoint("AR_epoch_*_step_*.pth", max_keep=1) if self.train_ar else None
        
        # 确定是否为本地续训练
        is_local_resume = local_cfm_checkpoint is not None or local_ar_checkpoint is not None
        
        if is_local_resume:
            # 如果存在本地检查点，则优先使用本地检查点进行续训练
            if self.train_cfm:
                cfm_checkpoint_path = local_cfm_checkpoint
            if self.train_ar:
                ar_checkpoint_path = local_ar_checkpoint
            checkpoint_type = "local"
        else:
            # 如果不存在本地检查点，则根据参数决定是否加载预训练检查点
            if self.train_cfm:
                cfm_checkpoint_path = pretrained_cfm_ckpt_path
            if self.train_ar:
                ar_checkpoint_path = pretrained_ar_ckpt_path
            
            # 确定检查点类型
            found_checkpoint = cfm_checkpoint_path or ar_checkpoint_path
            if found_checkpoint:
                # 检查是否为教师模式（基于蒸馏参数和训练参数）
                # 只有在对应模型需要训练且启用了蒸馏时才算作教师模式
                if (self.use_distill_ar and self.train_ar) or (self.use_distill_cfm and self.train_cfm):
                    checkpoint_type = "teacher"
                else:
                    checkpoint_type = "pretrained"
            else:
                checkpoint_type = "none"
        
        # 检查是否找到了检查点文件
        found_checkpoint = cfm_checkpoint_path or ar_checkpoint_path
        checkpoint_path = cfm_checkpoint_path if cfm_checkpoint_path else ar_checkpoint_path
        
        with self.accelerator.main_process_first():
            
            # 只有在checkpoint_type为"local"的情况下才从检查点恢复状态，其他情况都视为首次训练
            is_local_checkpoint = (checkpoint_type == "local")
            should_restore_from_checkpoint = is_local_checkpoint
            
            # 如果找到了检查点文件，尝试恢复训练状态
            if found_checkpoint and should_restore_from_checkpoint:
                # 确定使用哪个检查点文件来恢复状态
                checkpoint_to_load = cfm_checkpoint_path if cfm_checkpoint_path else ar_checkpoint_path
                
                try:
                    # 加载检查点状态
                    state = torch.load(checkpoint_to_load, map_location="cpu")
                    
                    if 'train_cfm' in state and 'train_ar' in state and self.train_cfm == state['train_cfm'] and self.train_ar == state['train_ar']:
                        print("Checked train_cfm and train_ar from checkpoint state")
                    else:
                        print("Warning: train_cfm and train_ar not match in checkpoint state. will not load from checkpoint")
                        if self.train_cfm:
                            cfm_checkpoint_path = pretrained_cfm_ckpt_path
                        if self.train_ar:
                            ar_checkpoint_path = pretrained_ar_ckpt_path
                        checkpoint_type = "pretrained"
                        raise Exception("train_cfm and train_ar not match in checkpoint state")
                    
                    # 恢复训练状态
                    if 'iters' in state:
                        self.iters = state['iters']
                    if 'epoch' in state:
                        self.epoch = state['epoch']
                    
                    # 恢复验证相关状态
                    if 'best_val_loss' in state:
                        self.best_val_loss = state['best_val_loss']
                        print(f"Loaded best_val_loss: {self.best_val_loss}")
                        
                    # 恢复 loss 缩放因子
                    if 'loss_scaling_factors' in state:
                        self.loss_scaling_factors = state['loss_scaling_factors']
                        print(f"Loaded loss_scaling_factors: {self.loss_scaling_factors}")
                        
                    if 'patience_counter' in state:
                        if self.resume_lr > 0.0:
                            #强制设置 恢复学习率时， 也强制重置 早停耐心计数器。
                            print("Using resume_lr > 0.0, forcing patience_counter to 0")
                            self.patience_counter = 0
                        else:
                            self.patience_counter = state['patience_counter']
                            print(f"Loaded patience_counter: {self.patience_counter}")
                            
                    if 'ema_loss' in state:
                        self.ema_loss = state['ema_loss']
                    
                    # 恢复学习率相关状态
                    if 'best_train_loss' in state:
                        self.best_train_loss = state['best_train_loss']
                        print(f"Loaded best_train_loss: {self.best_train_loss}")
                    if 'switched_to_val_scheduler' in state:
                        self.switched_to_val_scheduler = state['switched_to_val_scheduler']
                    
                    # 恢复学习率调度器状态
                    if 'scheduler' in state:
                        try:
                            self.scheduler.load_state_dict(state['scheduler'])
                        except Exception as e:
                            print(f"Warning: Could not load scheduler state from checkpoint: {e}")
                    
                    print(f"Loaded training state from checkpoint: epoch {self.epoch}, step {self.iters}")
                    
                    # When loading from checkpoint, epoch represents the epoch index (0-based)
                    # After loading, self.epoch represents the next epoch to train
                    # According to specification, we should increment epoch here to match V1 behavior
                    if checkpoint_type == "local":
                        # self.epoch += 1
                        print(f"Loaded local checkpoint from {checkpoint_to_load} with validation states")
                        print(f"Resuming from epoch {self.epoch}, step {self.iters}")
                    
                    # 如果是断点续训练，需要将学习率调度器快进到当前step
                    if self.iters > 0:
                        print(f"Fast-forwarding learning rate scheduler to step {self.iters}...")
                        for step in range(self.iters):
                            if step < self.warmup_steps:
                                # 预热阶段
                                self.warmup_scheduler.step()
                            elif step == self.warmup_steps:
                                # 预热结束，开始余弦退火
                                self.cosine_scheduler.last_epoch = 0
                                self.cosine_scheduler.step()
                            else:
                                # 余弦退火阶段
                                self.cosine_scheduler.step()
                        print(f"Learning rate scheduler fast-forwarded to step {self.iters}")
                        
                        # 获取检查点中保存的学习率
                        if 'current_lr' in state:
                            # 如果指定了大于0的resume_lr值，则强制使用该值作为当前学习率
                            if self.resume_lr > 0.0:
                                checkpoint_lr = self.resume_lr
                                print(f"Using resume_lr ({self.resume_lr}) to override checkpoint current_lr ({state['current_lr']})")
                            else:
                                checkpoint_lr = state['current_lr']
                            # 设置优化器的学习率到检查点中保存的值
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = checkpoint_lr
                            print(f"Set optimizer learning rate to {checkpoint_lr:.2e}")
                except Exception as e:
                    print(f"Warning: Could not load training state from checkpoint: {e}")
                    print(f"Loaded teacher/pretrained checkpoint, starting from scratch")
                    # 重置状态
                    self.iters = 0
                    self.epoch = 0
                    self.best_val_loss = float('inf')
                    self.patience_counter = 0
                    self.best_train_loss = float('inf')
                    self.switched_to_val_scheduler = False
            elif found_checkpoint:
                # 对于非本地检查点（教师模型/预训练模型），初始化为新训练
                print(f"Loaded teacher/pretrained checkpoint, starting from scratch")
                self.iters = 0
                self.epoch = 0
                self.best_val_loss = float('inf')
                self.patience_counter = 0
                self.best_train_loss = float('inf')
                self.switched_to_val_scheduler = False
            
            if cfm_checkpoint_path:
                print(f"Loading CFM checkpoint from {cfm_checkpoint_path}")
            if ar_checkpoint_path:
                print(f"Loading AR checkpoint from {ar_checkpoint_path}")
            
            # 加载模型检查点，只传递需要的检查点路径，并传递train_cfm和train_ar参数
            self.model.load_checkpoints(
                cfm_checkpoint_path=cfm_checkpoint_path if self.train_cfm else None, 
                ar_checkpoint_path=ar_checkpoint_path if self.train_ar else None,
                train_cfm=self.train_cfm,
                train_ar=self.train_ar
            )
            print(f"Starting from epoch {self.epoch}, step {self.iters}")
            
        # 准备模型用于分布式训练
        self.model = self.accelerator.prepare(self.model)

    def filter_state_dict_shapes(self, params, model):
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in params.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        skipped_keys = set(params.keys()) - set(filtered_state_dict.keys())
        if skipped_keys:
            print(
                f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}"
            )
        return filtered_state_dict, skipped_keys

    def set_teacher_model(self, teacher_model_ar_path=None, teacher_model_cfm_path=None):
        """设置教师模型用于知识蒸馏"""
        # 如果没有提供路径参数，则根据是否启用蒸馏和训练来决定使用哪个模型
        if teacher_model_ar_path is None:
            # 只有在启用了AR蒸馏且需要训练AR模型时才加载教师模型
            if self.use_distill_ar and self.train_ar:
                if self.pretrained_ar_ckpt_path:
                    # 如果指定了预训练检查点路径，则使用该路径
                    teacher_model_ar_path = self.pretrained_ar_ckpt_path
                else:
                    # 如果没有指定预训练检查点，则使用默认模型
                    from hf_utils import load_custom_model_from_hf
                    teacher_model_ar_path = load_custom_model_from_hf("Plachta/Seed-VC", "v2/ar_base.pth", None)
                    print(f"使用默认AR模型作为教师模型: {teacher_model_ar_path}")
                
        if teacher_model_cfm_path is None:
            # 只有在启用了CFM蒸馏且需要训练CFM模型时才加载教师模型
            if self.use_distill_cfm and self.train_cfm:
                if self.pretrained_cfm_ckpt_path:
                    # 如果指定了预训练检查点路径，则使用该路径
                    teacher_model_cfm_path = self.pretrained_cfm_ckpt_path
                else:
                    # 如果没有指定预训练检查点，则使用默认模型
                    from hf_utils import load_custom_model_from_hf
                    teacher_model_cfm_path = load_custom_model_from_hf("Plachta/Seed-VC", "v2/cfm_small.pth", None)
                    print(f"使用默认CFM模型作为教师模型: {teacher_model_cfm_path}")
            
        # 检查是否有教师模型需要加载
        # 只有在对应模型需要训练且启用了蒸馏时才检查教师模型文件
        has_ar_teacher = (teacher_model_ar_path and os.path.exists(teacher_model_ar_path)) if (self.train_ar and self.use_distill_ar) else False
        has_cfm_teacher = (teacher_model_cfm_path and os.path.exists(teacher_model_cfm_path)) if (self.train_cfm and self.use_distill_cfm) else False
        
        # 只有在需要加载教师模型时才继续
        if (self.train_ar and self.use_distill_ar and has_ar_teacher) or (self.train_cfm and self.use_distill_cfm and has_cfm_teacher):
            print(f"正在加载教师模型:")
            if self.train_ar and self.use_distill_ar and has_ar_teacher:
                print(f"  AR教师模型: {teacher_model_ar_path}")
            if self.train_cfm and self.use_distill_cfm and has_cfm_teacher:
                print(f"  CFM教师模型: {teacher_model_cfm_path}")
                
            # 加载教师模型（不更新其参数）
            with self.accelerator.main_process_first():
                cfg = DictConfig(self.config)
                self.teacher_model = hydra.utils.instantiate(cfg).to(self.device)
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                
                # 添加额外的检查，确保教师模型参数确实被冻结
                teacher_params_frozen = all(p.requires_grad == False for p in self.teacher_model.parameters())
                print(f"教师模型参数冻结状态检查: {teacher_params_frozen}")
                if not teacher_params_frozen:
                    print("警告: 教师模型参数未完全冻结!")
                    # 强制冻结所有参数
                    for p in self.teacher_model.parameters():
                        p.requires_grad = False
                
                # 根据训练参数和蒸馏参数决定加载哪些教师模型部分
                ar_checkpoint_path = teacher_model_ar_path if self.train_ar and self.use_distill_ar and has_ar_teacher else None
                cfm_checkpoint_path = teacher_model_cfm_path if self.train_cfm and self.use_distill_cfm and has_cfm_teacher else None
                
                print(f"加载教师模型检查点:")
                if ar_checkpoint_path:
                    print(f"  加载AR检查点: {ar_checkpoint_path}")
                if cfm_checkpoint_path:
                    print(f"  加载CFM检查点: {cfm_checkpoint_path}")
                
                # 加载检查点（根据训练参数决定加载哪一部分）
                self.teacher_model.load_checkpoints(
                    cfm_checkpoint_path=cfm_checkpoint_path,
                    ar_checkpoint_path=ar_checkpoint_path,
                    train_cfm=bool(cfm_checkpoint_path) and self.train_cfm and self.use_distill_cfm,
                    train_ar=bool(ar_checkpoint_path) and self.train_ar and self.use_distill_ar
                )
                
                # 准备教师模型用于分布式训练
                self.teacher_model = self.accelerator.prepare(self.teacher_model)
                
                # 设置模型为评估模式
                self.teacher_model.eval()
                
            print(f"教师模型加载完成")
        else:
            print("未找到教师模型检查点，将不使用知识蒸馏")

    def validate_one_step(self, batch):
        """在验证集上评估一个批次"""
        with torch.no_grad():
            waves, mels, wave_lens, mel_lens = batch
            # Resample to 16kHz for ASR models
            waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
            wave_lengths_16k = (wave_lens.float() * 16000 / self.sr).long()

            # Forward pass and loss calculation
            with self.accelerator.autocast():
                loss_ar, loss_cfm = self.model(
                    waves_16k.to(self.device),
                    mels.to(self.device),
                    wave_lengths_16k.to(self.device),
                    mel_lens.to(self.device),
                    forward_ar=self.train_ar,
                    forward_cfm=self.train_cfm,
                )

                # 在验证时也应用相同的损失缩放因子
                scaled_loss_ar = loss_ar * self.loss_scaling_factors['ar'] if isinstance(loss_ar, torch.Tensor) else torch.tensor(0.0, device=self.device)
                scaled_loss_cfm = loss_cfm * self.loss_scaling_factors['cfm'] if isinstance(loss_cfm, torch.Tensor) else torch.tensor(0.0, device=self.device)
                
                loss = scaled_loss_ar + scaled_loss_cfm
                
                # 验证时不包含蒸馏损失，但返回各组件用于详细打印
                return loss.detach().item(), scaled_loss_ar.detach().item(), scaled_loss_cfm.detach().item()
    
    def validate(self):
        """在整个验证集上评估模型"""
        if self.val_dataloader is None:
            return None
            
        self.model.eval()
        # 如果有教师模型，也确保其处于评估模式
        if self.teacher_model is not None:
            self.teacher_model.eval()
            # 确保教师模型的所有模块都处于评估模式
            def check_and_set_eval(model, model_name):
                for name, module in model.named_modules():
                    if hasattr(module, 'training') and module.training:
                        print(f"警告: {model_name}中的模块 {name} 未处于评估模式，正在强制设置...")
                        module.eval()
            
            check_and_set_eval(self.teacher_model, "教师模型")
        total_loss = 0
        total_ar_loss = 0
        total_cfm_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = [b.to(self.device) for b in batch]
                loss_result = self.validate_one_step(batch)
                if isinstance(loss_result, tuple) and len(loss_result) == 3:
                    loss, ar_loss, cfm_loss = loss_result
                else:
                    # 兼容旧版本返回值
                    loss = loss_result
                    ar_loss = cfm_loss = 0
                
                total_loss += loss
                total_ar_loss += ar_loss
                total_cfm_loss += cfm_loss
                num_batches += 1
        
        self.model.train()
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_ar_loss = total_ar_loss / num_batches
            avg_cfm_loss = total_cfm_loss / num_batches
            
            # 打印详细的验证损失组件信息
            if self.accelerator.is_main_process:
                print(f"\nDetailed Validation Loss Components:")
                print(f"  Average AR Loss: {avg_ar_loss:.6f}")
                print(f"  Average CFM Loss: {avg_cfm_loss:.6f}")
                print(f"  Average Total Loss: {avg_loss:.6f}")
                # 打印各组件占总损失的比例
                if avg_loss > 0:
                    print(f"  Loss Composition:")
                    print(f"    AR: {avg_ar_loss/avg_loss*100:.1f}%")
                    print(f"    CFM: {avg_cfm_loss/avg_loss*100:.1f}%")
            
            return avg_loss
        else:
            return None
    
    def _save_best_model(self):
        """保存最佳模型"""
        print(f"Saving best model with validation loss: {self.best_val_loss}")
        training_state = {
            'iters': self.iters,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'ema_loss': self.ema_loss,
            'train_ar': self.train_ar,
            'train_cfm': self.train_cfm,
            'best_train_loss': self.best_train_loss,
            'switched_to_val_scheduler': self.switched_to_val_scheduler,
            'scheduler': self.scheduler.state_dict(),
            'current_lr': self.optimizer.param_groups[0]['lr'],  # 保存当前学习率
            'loss_scaling_factors': self.loss_scaling_factors,
        }
        
        # 根据训练参数决定保存哪些模型
        # 当同时训练AR和CFM时，分别保存两个模型以区分
        if self.train_ar:
            state = {
                'net': {
                    'ar': self.accelerator.unwrap_model(self.model).ar.state_dict(),
                    'length_regulator': self.accelerator.unwrap_model(self.model).ar_length_regulator.state_dict(),
                },
                **training_state,  # 合并训练状态
            }
            save_path = os.path.join(self.log_dir, 'ar_best.pth')
            torch.save(state, save_path)
            print(f"Best AR model saved at {save_path}")
            # 同时保存 文本内容：
            with open(os.path.join(self.log_dir, 'ar_best_log.txt'), 'w') as f:
                f.write(f'AR_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth')
        
        if self.train_cfm:
            state = {
                'net': {
                    'cfm': self.accelerator.unwrap_model(self.model).cfm.state_dict(),
                    'length_regulator': self.accelerator.unwrap_model(self.model).cfm_length_regulator.state_dict(),
                },
                **training_state,  # 合并训练状态
            }
            save_path = os.path.join(self.log_dir, 'cfm_best.pth')
            torch.save(state, save_path)
            print(f"Best CFM model saved at {save_path}")
            # 同时保存 文本内容：
            with open(os.path.join(self.log_dir, 'cfm_best_log.txt'), 'w') as f:
                f.write(f'CFM_epoch_{self.epoch:05d}_step_{self.iters:05d}.pth')

    def train(self):
        """Main training loop"""
        
        print(f"Starting training from epoch {self.epoch}, step {self.iters} At {datetime.datetime.now()}")
        
        # # 在训练开始前计算初始损失缩放因子（如果启用了蒸馏）
        # if (self.use_distill_ar or self.use_distill_cfm) and \
        #    ((self.train_ar and self.use_distill_ar) or (self.train_cfm and self.use_distill_cfm)):
        self._compute_initial_loss_scaling_factors()
        
        print(f"Start training with loss: {self.ema_loss}")
        
        for epoch in range(self.max_epochs):
            epoch_start_time = time.time()
            
            init_epoch = False
            
            firstItersIdx = self.iters % len(self.train_dataloader)
            if firstItersIdx == 0 and self.iters != 0:
                self.epoch += 1
            if self.epoch >= self.max_epochs:
                print("Reached max epochs, stopping training")
                exit() # 无需归档，直接退出。

            # 在epoch开始时就设置随机种子，确保数据加载器能正确使用
            try:
                self.train_dataloader.sampler.set_epoch(self.epoch)
            except AttributeError:
                # 如果sampler没有set_epoch方法，则使用我们的自定义方法
                self.train_dataset.set_epoch(self.epoch)

            for i, batch in enumerate(tqdm(self.train_dataloader)):
                
                stepInEpoch = self.iters % len(self.train_dataloader)
                if stepInEpoch != i:
                    continue
                self.iters += 1
                
                if self.iters > self.max_steps:
                    self.should_stop = True
                    print("\nReached max steps, stopping training")
                    exit() # 无需归档，直接退出。
                    
                if self.iters == 1:
                    # 整个训练开始前，先验证一次。只打印。
                    first_val_loss = self.validate()
                    print(f"\nFirst validation loss: 【{first_val_loss}】")
                
                if not init_epoch:
                    self.model.train()
                    # 如果有教师模型，确保其处于评估模式
                    if self.teacher_model is not None:
                        self.teacher_model.eval()
                        # 添加额外的检查，确保教师模型确实处于评估模式
                        if self.teacher_model.training:
                            print("警告: 教师模型未处于评估模式，正在强制设置...")
                            self.teacher_model.eval()
                        
                        # 确保教师模型的所有模块都处于评估模式
                        def check_and_set_eval(model, model_name):
                            for name, module in model.named_modules():
                                if hasattr(module, 'training') and module.training:
                                    print(f"警告: {model_name}中的模块 {name} 未处于评估模式，正在强制设置...")
                                    module.eval()
                        
                        check_and_set_eval(self.teacher_model, "教师模型")
                    init_epoch = True
                
                # Process batch with fp16 error handling
                try:
                    self._process_batch(self.epoch, i, batch)
                except RuntimeError as e:
                    if "FP16_LAYER_NORM_ERROR" in str(e) and self.requested_fp16:
                        print(f"\nWarning: Encountered LayerNorm error with fp16 at step {self.iters}, falling back to fp32 training...")
                        # 实现真正的fp16到fp32的自动切换
                        self._fallback_to_fp32()
                        # 重新处理当前批次
                        self._process_batch(self.epoch, i, batch)
                    else:
                        # 如果不是fp16相关的错误，重新抛出异常
                        raise e
                
                # 验证和早停机制（warmup阶段不进行验证和早停）
                if self.val_dataloader and self.iters - self.warmup_steps > 0 and (self.iters - self.warmup_steps) % self.validation_interval == 0:
                    val_loss = self.validate()
                    if val_loss is not None:
                        if self.accelerator.is_main_process:
                            print(f"\nValidation loss at step {self.iters}:【{val_loss}】/「{self.ema_loss}」loss")
                        
                        # 早停机制
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            if self.accelerator.is_main_process:
                                print(f"\nImproved validation loss: 【{val_loss}】")
                                # 保存最佳模型
                                self._save_best_model()
                        else:
                            print(f"\nBest validation loss: {self.best_val_loss}")
                            self.patience_counter += 1
                            if self.accelerator.is_main_process:
                                print(f"No improvement in validation loss. Patience: {self.patience_counter}/{self.patience}")
                            
                            if self.patience_counter >= self.patience:
                                if self.accelerator.is_main_process:
                                    print(f"Early stopping triggered at step {self.iters}")
                                self.should_stop = True
                                # 训练完成，设置标志位
                                self.training_completed = True
                                self._save_checkpoint(self.epoch)
                                # 不再在早停时强制保存最佳模型，因为在训练过程中已经保存过了
                                # self._save_best_model()
                                return
                
                if self.iters >= self.max_steps and self.accelerator.is_main_process:
                    print("\nReached max steps, stopping training")
                    self._save_checkpoint(self.epoch)
                    # 训练完成，设置标志位
                    self.training_completed = True
                    exit() # 无需归档，直接退出。只保存当前检查点
                
                
                # 检查是否应该早停
                if self.should_stop:
                    break

            # Log epoch completion
            if self.accelerator.is_main_process:
                print(f"\nEpoch {self.epoch} completed in {time.time() - epoch_start_time:.2f} seconds")
            
            # 检查是否应该早停
            if self.should_stop:
                break

    def _process_batch(self, epoch, i, batch):
        """Process a single batch"""
        # Move batch to device
        waves, mels, wave_lens, mel_lens = batch
        # Resample to 16kHz for ASR models
        waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
        wave_lengths_16k = (wave_lens.float() * 16000 / self.sr).long()

        # Forward pass and loss calculation
        try:
            # 根据当前的精度设置决定是否使用autocast
            if self.requested_fp16:
                autocast_context = self.accelerator.autocast()
            else:
                # 如果已切换到fp32，则不使用autocast
                autocast_context = nullcontext()
            
            with autocast_context:
                loss_ar, loss_cfm = self.model(
                    waves_16k.to(self.device),
                    mels.to(self.device),
                    wave_lengths_16k.to(self.device),
                    mel_lens.to(self.device),
                    forward_ar=self.train_ar,
                    forward_cfm=self.train_cfm,
                )

                # 应用损失缩放因子（如果已计算）
                scaled_loss_ar = loss_ar * self.loss_scaling_factors['ar'] if isinstance(loss_ar, torch.Tensor) else torch.tensor(0.0, device=self.device)
                scaled_loss_cfm = loss_cfm * self.loss_scaling_factors['cfm'] if isinstance(loss_cfm, torch.Tensor) else torch.tensor(0.0, device=self.device)
                
                loss = scaled_loss_ar + scaled_loss_cfm
                
                # 如果有教师模型，添加知识蒸馏损失
                distill_loss = 0
                if self.teacher_model is not None:
                    # 确保教师模型处于评估模式
                    self.teacher_model.eval()
                    # 添加额外的检查，确保教师模型确实处于评估模式
                    if self.teacher_model.training:
                        print("警告: 教师模型未处于评估模式，正在强制设置...")
                        self.teacher_model.eval()
                    
                    # 确保教师模型的所有模块都处于评估模式
                    def check_and_set_eval(model, model_name):
                        for name, module in model.named_modules():
                            if hasattr(module, 'training') and module.training:
                                print(f"警告: {model_name}中的模块 {name} 未处于评估模式，正在强制设置...")
                                module.eval()
                    
                    check_and_set_eval(self.teacher_model, "教师模型")
                    
                    with torch.no_grad():
                        # 使用教师模型生成目标输出
                        # 只计算需要蒸馏的模型部分的输出，提高效率
                        teacher_forward_ar = self.train_ar and self.use_distill_ar
                        teacher_forward_cfm = self.train_cfm and self.use_distill_cfm
                        
                        teacher_loss_ar, teacher_loss_cfm = self.teacher_model(
                            waves_16k.to(self.device),
                            mels.to(self.device),
                            wave_lengths_16k.to(self.device),
                            mel_lens.to(self.device),
                            forward_ar=teacher_forward_ar,
                            forward_cfm=teacher_forward_cfm,
                        )
                    
                    # 计算学生模型和教师模型输出之间的蒸馏损失
                    # 只有在对应模型被训练且启用了蒸馏时才计算蒸馏损失
                    if self.train_cfm and self.use_distill_cfm:
                        # CFM蒸馏损失
                        # 确保loss_cfm和teacher_loss_cfm都是张量且形状匹配
                        if isinstance(loss_cfm, torch.Tensor) and isinstance(teacher_loss_cfm, torch.Tensor):
                            if loss_cfm.size() == teacher_loss_cfm.size():
                                # 确保数据类型一致
                                if loss_cfm.dtype != teacher_loss_cfm.dtype:
                                    print(f"警告: CFM蒸馏损失数据类型不一致 - student: {loss_cfm.dtype}, teacher: {teacher_loss_cfm.dtype}")
                                    teacher_loss_cfm = teacher_loss_cfm.to(loss_cfm.dtype)
                                # 使用KL散度计算蒸馏损失，添加温度参数支持
                                kl_loss = self.compute_kl_distill_loss(loss_cfm, teacher_loss_cfm.detach(), temperature=self.distill_temperature)
                                # 应用损失缩放因子和权重
                                scaled_distill_cfm_loss = kl_loss * self.loss_scaling_factors['distill_cfm']
                                distill_loss += scaled_distill_cfm_loss
                            else:
                                print(f"Warning: Shape mismatch in CFM distillation loss - student: {loss_cfm.size()}, teacher: {teacher_loss_cfm.size()}")
                        else:
                            print(f"Warning: Type mismatch in CFM distillation loss - student: {type(loss_cfm)}, teacher: {type(teacher_loss_cfm)}")
                    if self.train_ar and self.use_distill_ar:
                        # AR蒸馏损失
                        # 确保loss_ar和teacher_loss_ar都是张量且形状匹配
                        if isinstance(loss_ar, torch.Tensor) and isinstance(teacher_loss_ar, torch.Tensor):
                            if loss_ar.size() == teacher_loss_ar.size():
                                # 确保数据类型一致
                                if loss_ar.dtype != teacher_loss_ar.dtype:
                                    print(f"警告: AR蒸馏损失数据类型不一致 - student: {loss_ar.dtype}, teacher: {teacher_loss_ar.dtype}")
                                    teacher_loss_ar = teacher_loss_ar.to(loss_ar.dtype)
                                # 使用KL散度计算蒸馏损失，添加温度参数支持
                                kl_loss = self.compute_kl_distill_loss(loss_ar, teacher_loss_ar.detach(), temperature=self.distill_temperature)
                                # 应用损失缩放因子和权重
                                scaled_distill_ar_loss = kl_loss * self.loss_scaling_factors['distill_ar']
                                distill_loss += scaled_distill_ar_loss
                            else:
                                print(f"Warning: Shape mismatch in AR distillation loss - student: {loss_ar.size()}, teacher: {teacher_loss_ar.size()}")
                        else:
                            print(f"Warning: Type mismatch in AR distillation loss - student: {type(loss_ar)}, teacher: {type(teacher_loss_ar)}")
                loss_total = loss + distill_loss
                # 使用指数移动平均计算ema_loss，与train.py保持一致
                if not hasattr(self, 'ema_loss'):
                    self.ema_loss = 0
                if not hasattr(self, 'loss_smoothing_rate'):
                    self.loss_smoothing_rate = 0.99
                self.ema_loss = (
                    self.ema_loss * self.loss_smoothing_rate + loss_total.item() * (1 - self.loss_smoothing_rate)
                    if self.iters > 0 else loss_total.item()
                )
                self.accelerator.backward(loss_total)

                # 自适应梯度裁剪 - 根据损失值动态调整裁剪阈值
                # 基础阈值为self.grad_clip_norm，但当损失较大时会降低阈值
                base_clip_norm = self.grad_clip_norm
                # 降低最低限制，允许更严格的梯度裁剪
                adaptive_clip_norm = max(0.01, min(base_clip_norm, 10.0 / (loss_total.item() + 1e-8)))
                grad_norm_g = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), adaptive_clip_norm
                )
                self.optimizer.step()
                # Increment iteration counter
                # self.iters += 1
        
        except RuntimeError as e:
            if "LayerNormKernelImpl" in str(e) and self.requested_fp16:
                print(f"Warning: Encountered LayerNorm error with fp16 at step {self.iters}, attempting to fall back to fp32 training...")
                # 抛出自定义异常，让上层处理真正的回退逻辑
                raise RuntimeError("FP16_LAYER_NORM_ERROR") from e
            else:
                # 如果不是fp16相关的错误，重新抛出异常
                raise e
        
        # 应用预热和余弦退火调度
        if self.iters < self.warmup_steps:
            # 预热阶段
            self.warmup_scheduler.step()
        elif self.iters == self.warmup_steps:
            # 预热结束，开始余弦退火
            self.cosine_scheduler.last_epoch = 0
            self.cosine_scheduler.step()
        else:
            # 余弦退火阶段
            self.cosine_scheduler.step()
        
        self.optimizer.zero_grad()

        # 每隔一定step打印学习率信息
        if self.iters % self.lr_adjust_interval == 0 and self.accelerator.is_main_process:
            # 获取当前使用的学习率调度器的学习率
            if self.iters < self.warmup_steps:
                # 预热阶段使用预热调度器的学习率
                cur_lr = self.warmup_scheduler.get_last_lr()[0]
            else:
                # 余弦退火阶段使用余弦调度器的学习率
                cur_lr = self.cosine_scheduler.get_last_lr()[0]
            print(f"Learning rate at step {self.iters}: 《{cur_lr:.2e}》")
            
            # 在预热阶段结束后，根据验证损失情况决定是否手动调整学习率
            if self.iters >= self.warmup_steps and self.val_dataloader:
                # 当patience_counter达到一定阈值时，手动降低学习率
                # 每当patience_counter增加时，按0.5的比例降低学习率
                switch_patience = max(1, self.patience // 4)  # 使用早停耐心值的四分之一作为切换耐心值
                if self.patience_counter >= switch_patience and self.patience_counter < self.patience and self.patience_counter % max(2, switch_patience) == 0:
                    # 获取当前学习率并降低它
                    current_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = max(current_lr * 0.5, self.min_lr)
                    
                    # 手动设置新的学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    print(f"Learning rate manually adjusted from {current_lr:.2e} to 《{new_lr:.2e}》 based on validation loss plateau")

        # Log training progress
        self._log_training_progress(epoch, i, loss_total, scaled_loss_ar, scaled_loss_cfm, grad_norm_g, distill_loss)

        # Save checkpoint
        if self.iters % self.save_interval == 0 and self.accelerator.is_main_process:
            self._save_checkpoint(epoch)

    def _fallback_to_fp32(self):
        """Fallback from fp16 to fp32 training"""
        print("Initiating fallback from fp16 to fp32 training...")
        
        # 更新标记
        self.requested_fp16 = False
        self.mixed_precision = 'no'
        
        # 重新初始化Accelerator以使用fp32
        # 注意：这里我们不能简单地重新创建Accelerator，因为这会破坏分布式训练设置
        # 我们需要告诉accelerator后续操作不再使用autocast
        print("Switched to fp32 training mode. Continuing training...")

    def _log_training_progress(self, epoch, i, loss, loss_ar, loss_cfm, grad_norm_g, distill_loss=0):
        """Log training progress to tensorboard and wandb"""
        if self.iters % self.log_interval == 0 and self.accelerator.is_main_process:
            with torch.no_grad():
                # 获取当前使用的学习率调度器的学习率
                if self.iters < self.warmup_steps:
                    # 预热阶段使用预热调度器的学习率
                    cur_lr = self.warmup_scheduler.get_last_lr()[0] if i != 0 else 0
                else:
                    # 余弦退火阶段使用余弦调度器的学习率
                    cur_lr = self.cosine_scheduler.get_last_lr()[0] if i != 0 else 0

                # 打印详细的损失组件信息
                print(f"\nDetailed Loss Components at epoch {epoch}, step {self.iters}:")
                print(f"  AR Loss: {loss_ar.item():.6f}")
                print(f"  CFM Loss: {loss_cfm.item():.6f}")
                print(f"  Total Main Loss: {loss.item():.6f}")
                if self.teacher_model is not None and (self.use_distill_ar or self.use_distill_cfm):
                    print(f"  Distill Loss: {distill_loss:.6f}")
                # 计算总训练损失（包含蒸馏损失）
                total_training_loss = loss.item() + (distill_loss if self.teacher_model is not None else 0)
                print(f"  Total Training Loss: {total_training_loss:.6f}")
                # 同时打印各组件占总损失的比例
                if total_training_loss > 0:
                    print(f"  Loss Composition:")
                    print(f"    AR: {loss_ar.item()/total_training_loss*100:.1f}%")
                    print(f"    CFM: {loss_cfm.item()/total_training_loss*100:.1f}%")
                    if self.teacher_model is not None and (self.use_distill_ar or self.use_distill_cfm):
                        print(f"    Distill: {distill_loss/total_training_loss*100:.1f}%")

                # Log to console
                print("Epoch %d, Step %d, Iteration %d, Loss: 「%.4f」, Loss AR: %.4f, Loss CFM: %.4f, Grad Norm: %.4f, LR: %.6f"
                      % (epoch, self.iters, i, total_training_loss, loss_ar.item(), loss_cfm.item(), grad_norm_g, cur_lr))
                
                # 如果有验证集，也打印验证相关信息
                if self.val_dataloader:
                    print(f"  Best val loss: {self.best_val_loss:.4f}, Patience: {self.patience_counter}/{self.patience}")

    def _save_checkpoint(self, epoch):
        """Save model checkpoint"""
        print('Saving checkpoint...')
        
        # 准备要保存的训练状态
        # Save with the epoch index (0-based indexing)
        # According to the specification, we save the completed epoch number
        # The epoch parameter passed in is already the completed epoch number
        # save_epoch = epoch - 1 if epoch > 0 else 0
        save_epoch = epoch
        training_state = {
            'iters': self.iters,
            'epoch': save_epoch,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'ema_loss': self.ema_loss,
            'train_ar': self.train_ar,
            'train_cfm': self.train_cfm,
            'best_train_loss': self.best_train_loss,
            'switched_to_val_scheduler': self.switched_to_val_scheduler,
            'scheduler': self.scheduler.state_dict(),
            'current_lr': self.optimizer.param_groups[0]['lr'],  # 保存当前学习率
            'loss_scaling_factors': self.loss_scaling_factors,
        }
        
        if self.train_ar:
            # Find all checkpoints and remove old ones
            self._remove_old_checkpoints("AR_epoch_*_step_*.pth", max_keep=1)
            state = {
                'net': {
                    'ar': self.accelerator.unwrap_model(self.model).ar.state_dict(),
                    'length_regulator': self.accelerator.unwrap_model(self.model).ar_length_regulator.state_dict(),
                },
                **training_state,  # 合并训练状态
            }
            save_path = os.path.join(self.log_dir, 'AR_epoch_%05d_step_%05d.pth' % (save_epoch, self.iters))
            torch.save(state, save_path)
            print(f"Saved AR checkpoint to {save_path}")
        if self.train_cfm:
             # Find all checkpoints and remove old ones
            self._remove_old_checkpoints("CFM_epoch_*_step_*.pth", max_keep=1)
            state = {
                'net': {
                    'cfm': self.accelerator.unwrap_model(self.model).cfm.state_dict(),
                    'length_regulator': self.accelerator.unwrap_model(self.model).cfm_length_regulator.state_dict(),
                },
                **training_state,  # 合并训练状态
            }
            save_path = os.path.join(self.log_dir, 'CFM_epoch_%05d_step_%05d.pth' % (save_epoch, self.iters))
            torch.save(state, save_path)
            print(f"Saved CFM checkpoint to {save_path}")
           
            
    def _remove_old_checkpoints(self, name_pattern, max_keep=1):
        """Remove old checkpoints"""
        checkpoints = glob.glob(os.path.join(self.log_dir, name_pattern))
        if len(checkpoints) > max_keep:
            # Sort by step
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            # Remove all except last 1
            for cp in checkpoints[:-max_keep]:
                os.remove(cp)


def copy_final_models(log_dir, run_name, train_cfm=False, train_ar=False, config_path=None, training_completed=False):
    """共用的文件拷贝函数，用于将最终模型和配置文件拷贝到基础运行目录"""
    # 只有在训练完成时才拷贝最终模型文件
    if not training_completed:
        return
    # 保持与V1版本一致的归档逻辑
    base_log_dir = os.path.join(os.path.dirname(log_dir), run_name)
    os.makedirs(base_log_dir, exist_ok=True)
    
    # 拷贝CFM最终模型（如果训练了CFM模型）
    # 优先使用最佳模型文件，如果没有则使用最新的检查点
    if train_cfm:
        # 检查是否存在最佳CFM模型
        best_cfm_path = os.path.join(log_dir, 'cfm_best.pth')
        if os.path.exists(best_cfm_path):
            shutil.copy2(best_cfm_path, os.path.join(base_log_dir, 'final_cfm_model.pth'))
            print(f"已将最佳CFM模型拷贝到: {os.path.join(base_log_dir, 'final_cfm_model.pth')}")
        else:
            cfm_checkpoints = glob.glob(os.path.join(log_dir, 'CFM_epoch_*_step_*.pth'))
            if cfm_checkpoints:
                latest_cfm_checkpoint = max(cfm_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                shutil.copy2(latest_cfm_checkpoint, os.path.join(base_log_dir, 'final_cfm_model.pth'))
                print(f"已将最终CFM模型拷贝到: {os.path.join(base_log_dir, 'final_cfm_model.pth')}")
    
    # 拷贝AR最终模型（如果训练了AR模型）
    # 优先使用最佳模型文件，如果没有则使用最新的检查点
    if train_ar:
        # 检查是否存在最佳AR模型
        best_ar_path = os.path.join(log_dir, 'ar_best.pth')
        if os.path.exists(best_ar_path):
            shutil.copy2(best_ar_path, os.path.join(base_log_dir, 'final_ar_model.pth'))
            print(f"已将最佳AR模型拷贝到: {os.path.join(base_log_dir, 'final_ar_model.pth')}")
        else:
            ar_checkpoints = glob.glob(os.path.join(log_dir, 'AR_epoch_*_step_*.pth'))
            if ar_checkpoints:
                latest_ar_checkpoint = max(ar_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                shutil.copy2(latest_ar_checkpoint, os.path.join(base_log_dir, 'final_ar_model.pth'))
                print(f"已将最终AR模型拷贝到: {os.path.join(base_log_dir, 'final_ar_model.pth')}")
    
    # 拷贝配置文件
    if config_path:
        config_file = os.path.join(log_dir, os.path.basename(config_path))
        if os.path.exists(config_file):
            shutil.copy2(config_file, os.path.join(base_log_dir, os.path.basename(config_path)))
            print(f"已将配置文件拷贝到: {os.path.join(base_log_dir, os.path.basename(config_path))}")


def main(args):
    # 如果config参数为空，则使用默认值
    if not args.config:
        args.config = './configs/v2/vc_wrapper.yaml'
        
    # 统一使用Trainer进行训练，无论是否启用知识蒸馏
    # 使用数据集目录名作为后缀，保持一致的输出目录结构
    dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
    version_run_name = f"{args.run_name}_{dataset_name}"
    
    trainer = Trainer(
        config_path=args.config,
        pretrained_cfm_ckpt_path=args.pretrained_cfm_ckpt,
        pretrained_ar_ckpt_path=args.pretrained_ar_ckpt,
        data_dir=args.dataset_dir,
        run_name=version_run_name,
        batch_size=args.batch_size,
        steps=args.max_steps,
        max_epochs=args.max_epochs,
        save_interval=args.save_every,
        num_workers=args.num_workers,
        train_cfm=args.train_cfm,
        train_ar=args.train_ar,
        fp16=args.fp16,
        val_dataset_dir=args.val_dataset_dir,
        patience=args.patience,
        validation_interval=args.validation_interval,
        min_lr=args.min_lr,
        lr_adjust_interval=args.lr_adjust_interval,
        initial_lr=args.initial_lr,
        warmup_steps=args.warmup_steps,
        resume_lr=args.resume_lr,
        language=args.language,
        # 传递新的知识蒸馏参数
        distill_ar=args.distill_ar,
        distill_cfm=args.distill_cfm,
        grad_clip_norm=args.grad_clip_norm,
        distill_temperature=args.distill_temperature,
    )
    # 添加fp16错误处理
    try:
        trainer.train()
    except RuntimeError as e:
        if "LayerNormKernelImpl" in str(e) and args.fp16:
            print(f"Error: Encountered LayerNorm error with fp16 in training. The system should automatically switch to fp32.")
            # 实际的自动切换在Trainer内部处理
        else:
            raise e
    
    # 获取日志目录和训练完成状态
    log_dir = trainer.log_dir
    training_completed = getattr(trainer, 'training_completed', False)
    
    # 训练完成后，将最终模型和配置文件拷贝到基础运行目录
    copy_final_models(log_dir, args.run_name, args.train_cfm, args.train_ar, args.config, training_completed)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/v2/vc_wrapper.yaml')
    parser.add_argument('--pretrained-cfm-ckpt', type=str, default=None)
    parser.add_argument('--pretrained-ar-ckpt', type=str, default=None)
    parser.add_argument('--dataset-dir', type=str, default='/path/to/dataset')
    parser.add_argument('--run-name', type=str, default='my_run')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--train-cfm', action='store_true', help='Train CFM model')
    parser.add_argument('--train-ar', action='store_true', help='Train AR model')
    parser.add_argument('--fp16', action='store_true', help='Use fp16 precision')
    
    # 验证集相关参数
    parser.add_argument('--val-dataset-dir', type=str, default=None, help='Validation dataset directory')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--validation-interval', type=int, default=50, help='Validation interval in steps')
    
    # 学习率调度参数
    parser.add_argument('--min-lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--lr-adjust-interval', type=int, default=50, help='Interval (in steps) for learning rate adjustment print logs')
    parser.add_argument('--initial-lr', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='Number of warmup steps')
    parser.add_argument('--resume-lr', type=float, default=0.0, help='Resume learning rate for resuming training from checkpoint')
    
    # 知识蒸馏参数
    parser.add_argument('--distill-ar', type=float, default=0.0, help='Enable knowledge distillation for AR model with specified weight (0.0 means no distillation)')
    parser.add_argument('--distill-cfm', type=float, default=0.0, help='Enable knowledge distillation for CFM model with specified weight (0.0 means no distillation)')
    parser.add_argument('--grad-clip-norm', type=float, default=1.0,
                       help='Gradient clipping norm value')
    parser.add_argument('--distill-temperature', type=float, default=1.0,
                       help='Temperature parameter for knowledge distillation')
    
    # 语言参数
    parser.add_argument('--language', type=str, default=None, help='Language for Whisper model')
    
    args = parser.parse_args()
    main(args)