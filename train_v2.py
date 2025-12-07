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


class ManualProgressiveTrainer:
    def __init__(self, config_path, run_name, **kwargs):
        self.config_path = config_path
        self.run_name = run_name
        # 添加新的知识蒸馏参数
        self.distill_ar = kwargs.pop('distill_ar', False)
        self.distill_cfm = kwargs.pop('distill_cfm', False)
        # 添加预训练检查点路径参数
        self.pretrained_cfm_ckpt_path = kwargs.pop('pretrained_cfm_ckpt_path', None)
        self.pretrained_ar_ckpt_path = kwargs.pop('pretrained_ar_ckpt_path', None)
        self.kwargs = kwargs
        self.current_trainer = None
    
    def train_single_dataset(self, dataset_dir, run_name_suffix=None):
        """训练单个数据集，不进行自动检查点管理"""
        # 直接使用传入的run_name，避免重复添加数据集名后缀
        # Trainer类内部已经会处理数据集名后缀
        version_run_name = self.run_name
        
        # 保存数据目录用于可能的重新初始化
        self.dataset_dir = dataset_dir
        
        # 初始化训练器
        self.current_trainer = Trainer(
            config_path=self.config_path,
            pretrained_cfm_ckpt_path=self.pretrained_cfm_ckpt_path,
            pretrained_ar_ckpt_path=self.pretrained_ar_ckpt_path,
            data_dir=dataset_dir,
            run_name=version_run_name,
            **self.kwargs
        )
        
        # 开始训练，添加fp16错误处理
        try:
            self.current_trainer.train()
        except RuntimeError as e:
            original_fp16 = self.kwargs.get('fp16', False)
            if "LayerNormKernelImpl" in str(e) and original_fp16:
                print(f"Error: Encountered LayerNorm error with fp16 in progressive training. Attempting automatic fallback to fp32...")
                # 尝试使用fp32重新训练
                self.kwargs['fp16'] = False
                print("Reinitializing trainer with fp32 precision...")
                # 重新初始化训练器
                self.current_trainer = Trainer(
                    config_path=self.config_path,
                    pretrained_cfm_ckpt_path=self.pretrained_cfm_ckpt_path,
                    pretrained_ar_ckpt_path=self.pretrained_ar_ckpt_path,
                    data_dir=self.dataset_dir,  # 使用原始数据目录
                    run_name=self.run_name,  # 使用原始的run_name，避免重复添加数据集名后缀
                    **self.kwargs
                )
                # 重新开始训练
                self.current_trainer.train()
            else:
                raise e
        
        # 获取训练完成状态
        training_completed = getattr(self.current_trainer, 'training_completed', False)
        print(f"数据集训练完成。训练完成状态: {training_completed}")
        return self.current_trainer.log_dir, training_completed


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
            distill_ar=False,
            distill_cfm=False,
        ):
        self.config_path = config_path
        self.mixed_precision = mixed_precision
        self.requested_fp16 = fp16  # 保存用户请求的fp16设置
        # 保存新的知识蒸馏参数
        self.distill_ar = distill_ar
        self.distill_cfm = distill_cfm
        # 保存预训练检查点路径（同时用于预训练和蒸馏）
        self.pretrained_ar_ckpt_path = pretrained_ar_ckpt_path
        self.pretrained_cfm_ckpt_path = pretrained_cfm_ckpt_path
        
        # Check FORCE_CPU environment variable
        force_cpu = os.environ.get('FORCE_CPU', '0') == '1'

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
        
        # 学习率调度相关参数
        self.initial_lr = initial_lr  # 初始学习率
        self.min_lr = min_lr      # 最小学习率
        self.lr_adjust_interval = lr_adjust_interval  # 学习率调整间隔
        self.warmup_steps = warmup_steps  # 学习率预热步数
        self.best_train_loss = float('inf')  # 用于学习率调度的最佳训练损失
        self.switched_to_val_scheduler = False  # 学习率调度器切换状态

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
        if (self.distill_ar and self.train_ar) or (self.distill_cfm and self.train_cfm):
            self.set_teacher_model()

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
                if (self.distill_ar and self.train_ar) or (self.distill_cfm and self.train_cfm):
                    checkpoint_type = "teacher"
                else:
                    checkpoint_type = "pretrained"
            else:
                checkpoint_type = "none"
        
        # 检查是否找到了检查点文件
        found_checkpoint = cfm_checkpoint_path or ar_checkpoint_path
        checkpoint_path = cfm_checkpoint_path if cfm_checkpoint_path else ar_checkpoint_path
        
        with self.accelerator.main_process_first():
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
                    
                    # 恢复训练状态
                    if 'iters' in state:
                        self.iters = state['iters']
                    if 'epoch' in state:
                        self.epoch = state['epoch']
                    
                    # 恢复验证相关状态
                    if 'best_val_loss' in state:
                        self.best_val_loss = state['best_val_loss']
                    if 'patience_counter' in state:
                        self.patience_counter = state['patience_counter']
                    
                    # 恢复学习率相关状态
                    if 'best_train_loss' in state:
                        self.best_train_loss = state['best_train_loss']
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
                        
                        # 恢复优化器的学习率
                        # 获取检查点中保存的学习率
                        if 'current_lr' in state:
                            checkpoint_lr = state['current_lr']
                            # 设置优化器的学习率到检查点中保存的值
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = checkpoint_lr
                            print(f"Restored optimizer learning rate to {checkpoint_lr:.2e}")
                except Exception as e:
                    print(f"Warning: Could not load training state from checkpoint: {e}")
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
                print(f"Teacher/pretrained model loaded, starting from epoch {self.epoch}, step {self.iters}")
        
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
        # 如果没有提供路径参数，则使用预训练检查点路径
        if teacher_model_ar_path is None and self.pretrained_ar_ckpt_path:
            teacher_model_ar_path = self.pretrained_ar_ckpt_path
        if teacher_model_cfm_path is None and self.pretrained_cfm_ckpt_path:
            teacher_model_cfm_path = self.pretrained_cfm_ckpt_path
            
        # 检查是否有教师模型需要加载
        # 只有在对应模型需要训练且启用了蒸馏时才检查教师模型文件
        has_ar_teacher = (teacher_model_ar_path and os.path.exists(teacher_model_ar_path)) if (self.train_ar and self.distill_ar) else False
        has_cfm_teacher = (teacher_model_cfm_path and os.path.exists(teacher_model_cfm_path)) if (self.train_cfm and self.distill_cfm) else False
        
        # 只有在需要加载教师模型时才继续
        if (self.train_ar and self.distill_ar and has_ar_teacher) or (self.train_cfm and self.distill_cfm and has_cfm_teacher):
            print(f"正在加载教师模型:")
            if self.train_ar and self.distill_ar and has_ar_teacher:
                print(f"  AR教师模型: {teacher_model_ar_path}")
            if self.train_cfm and self.distill_cfm and has_cfm_teacher:
                print(f"  CFM教师模型: {teacher_model_cfm_path}")
                
            # 加载教师模型（不更新其参数）
            with self.accelerator.main_process_first():
                cfg = DictConfig(self.config)
                self.teacher_model = hydra.utils.instantiate(cfg).to(self.device)
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                
                # 根据训练参数和蒸馏参数决定加载哪些教师模型部分
                ar_checkpoint_path = teacher_model_ar_path if self.train_ar and self.distill_ar and has_ar_teacher else None
                cfm_checkpoint_path = teacher_model_cfm_path if self.train_cfm and self.distill_cfm and has_cfm_teacher else None
                
                print(f"加载教师模型检查点:")
                if ar_checkpoint_path:
                    print(f"  加载AR检查点: {ar_checkpoint_path}")
                if cfm_checkpoint_path:
                    print(f"  加载CFM检查点: {cfm_checkpoint_path}")
                
                # 加载检查点（根据训练参数决定加载哪一部分）
                # 在教师模型加载时，不自动加载默认模型，只在明确指定了检查点路径时才加载
                self.teacher_model.load_checkpoints(
                    cfm_checkpoint_path=cfm_checkpoint_path,
                    ar_checkpoint_path=ar_checkpoint_path,
                    train_cfm=bool(cfm_checkpoint_path) and self.train_cfm and self.distill_cfm,
                    train_ar=bool(ar_checkpoint_path) and self.train_ar and self.distill_ar
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

                loss = loss_ar + loss_cfm
                
                return loss.detach().item()
    
    def validate(self):
        """在整个验证集上评估模型"""
        if self.val_dataloader is None:
            return None
            
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = [b.to(self.device) for b in batch]
                loss = self.validate_one_step(batch)
                total_loss += loss
                num_batches += 1
        
        self.model.train()
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            return avg_loss
        else:
            return None

    def train(self):
        """Main training loop"""
        
        for epoch in range(self.max_epochs):
            epoch_start_time = time.time()
            
            init_epoch = False

            for i, batch in enumerate(tqdm(self.train_dataloader)):
                stepInEpoch = self.iters % len(self.train_dataloader)
                if stepInEpoch == 0 and self.iters != 0:
                    self.epoch += 1
                    # epoch += 1
                if stepInEpoch != i:
                    continue
                self.iters += 1
                
                if self.iters > self.max_steps or self.epoch >= self.max_epochs:
                    self.should_stop = True
                    print("Reached max epochs (or max steps), stopping training")
                    exit() # 无需归档，直接退出。
                
                if not init_epoch:
                    try:
                        self.train_dataloader.sampler.set_epoch(self.epoch)
                    except AttributeError:
                        pass
                    self.model.train()
                    init_epoch = True
                
                # Process batch with fp16 error handling
                try:
                    self._process_batch(self.epoch, i, batch)
                except RuntimeError as e:
                    if "FP16_LAYER_NORM_ERROR" in str(e) and self.requested_fp16:
                        print(f"Warning: Encountered LayerNorm error with fp16 at step {self.iters}, falling back to fp32 training...")
                        # 实现真正的fp16到fp32的自动切换
                        self._fallback_to_fp32()
                        # 重新处理当前批次
                        self._process_batch(self.epoch, i, batch)
                    else:
                        # 如果不是fp16相关的错误，重新抛出异常
                        raise e
                
                # 验证和早停机制
                if self.val_dataloader and self.iters % self.validation_interval == 0:
                    val_loss = self.validate()
                    if val_loss is not None:
                        if self.accelerator.is_main_process:
                            print(f"Validation loss at step {self.iters}:【{val_loss}】")
                        
                        # 早停机制
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.patience_counter = 0
                            if self.accelerator.is_main_process:
                                print(f"Improved validation loss: 【{val_loss}】")
                        else:
                            self.patience_counter += 1
                            if self.accelerator.is_main_process:
                                print(f"No improvement in validation loss. Patience: {self.patience_counter}/{self.patience}")
                            
                            if self.patience_counter >= self.patience:
                                if self.accelerator.is_main_process:
                                    print(f"Early stopping triggered at step {self.iters}")
                                self.should_stop = True
                                # 训练完成，设置标志位
                                self.training_completed = True
                                return
                
                if self.iters >= self.max_steps and self.accelerator.is_main_process:
                    print("Reached max steps, stopping training")
                    self._save_checkpoint(self.epoch)
                    # 训练完成，设置标志位
                    self.training_completed = True
                    exit() # 无需归档，直接退出。只保存当前检查点
                
                
                # 检查是否应该早停
                if self.should_stop:
                    break

            # Log epoch completion
            if self.accelerator.is_main_process:
                print(f"Epoch {self.epoch} completed in {time.time() - epoch_start_time:.2f} seconds")
            
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

                loss = loss_ar + loss_cfm
                
                # 如果有教师模型，添加知识蒸馏损失
                distill_loss = 0
                if self.teacher_model is not None:
                    with torch.no_grad():
                        # 使用教师模型生成目标输出
                        # 只计算需要蒸馏的模型部分的输出，提高效率
                        teacher_forward_ar = self.train_ar and self.distill_ar
                        teacher_forward_cfm = self.train_cfm and self.distill_cfm
                        
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
                    if self.train_cfm and self.distill_cfm:
                        # CFM蒸馏损失
                        distill_loss += F.mse_loss(loss_cfm, teacher_loss_cfm.detach()) * 0.5  # 知识蒸馏损失权重
                    if self.train_ar and self.distill_ar:
                        # AR蒸馏损失
                        distill_loss += F.mse_loss(loss_ar, teacher_loss_ar.detach()) * 0.3  # AR知识蒸馏损失权重
                loss_total = loss + distill_loss

                self.accelerator.backward(loss_total)

                grad_norm_g = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1000.0
                )
                self.optimizer.step()
        except RuntimeError as e:
            if "LayerNormKernelImpl" in str(e) and self.requested_fp16:
                print(f"Warning: Encountered LayerNorm error with fp16 at step {self.iters}, attempting to fall back to fp32 training...")
                # 抛出自定义异常，让上层处理真正的回退逻辑
                raise RuntimeError("FP16_LAYER_NORM_ERROR") from e
            else:
                # 如果不是fp16相关的错误，重新抛出异常
                raise e
        # Increment iteration counter
        # self.iters += 1
        
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
                if self.patience_counter >= switch_patience and self.patience_counter % max(2, switch_patience // 2) == 0:
                    # 获取当前学习率并降低它
                    current_lr = self.optimizer.param_groups[0]['lr']
                    new_lr = max(current_lr * 0.5, self.min_lr)
                    
                    # 手动设置新的学习率
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_lr
                    
                    print(f"Learning rate manually adjusted from {current_lr:.2e} to 《{new_lr:.2e}》 based on validation loss plateau")

        # Log training progress
        self._log_training_progress(epoch, i, loss_total, loss_ar, loss_cfm, grad_norm_g)

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

    def _log_training_progress(self, epoch, i, loss, loss_ar, loss_cfm, grad_norm_g):
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

                # Log to console
                print("Epoch %d, Step %d, Iteration %d, Loss: %.4f, Loss AR: %.4f, Loss CFM: %.4f, Grad Norm: %.4f, LR: %.6f"
                      % (epoch, self.iters, i, loss.item(), loss_ar.item(), loss_cfm.item(), grad_norm_g, cur_lr))
                
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
            'best_train_loss': self.best_train_loss,
            'switched_to_val_scheduler': self.switched_to_val_scheduler,
            'scheduler': self.scheduler.state_dict(),
            'current_lr': self.optimizer.param_groups[0]['lr'],  # 保存当前学习率
        }
        
        if self.train_ar:
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

            # Find all checkpoints and remove old ones
            self._remove_old_checkpoints("AR_epoch_*_step_*.pth", max_keep=1)
        if self.train_cfm:
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

            # Find all checkpoints and remove old ones
            self._remove_old_checkpoints("CFM_epoch_*_step_*.pth", max_keep=1)
            
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
    if train_cfm:
        cfm_checkpoints = glob.glob(os.path.join(log_dir, 'CFM_epoch_*_step_*.pth'))
        if cfm_checkpoints:
            latest_cfm_checkpoint = max(cfm_checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            shutil.copy2(latest_cfm_checkpoint, os.path.join(base_log_dir, 'final_cfm_model.pth'))
            print(f"已将最终CFM模型拷贝到: {os.path.join(base_log_dir, 'final_cfm_model.pth')}")
    
    # 拷贝AR最终模型（如果训练了AR模型）
    if train_ar:
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
            print(f"已将配置文件拷贝到: {os.path.join(base_log_dir, 'final_ar_model.pth')}")

def main(args):
    # 如果config参数为空，则使用默认值
    if not args.config:
        args.config = './configs/v2/vc_wrapper.yaml'
        
    # 检查是否启用知识蒸馏
    # 在V2版本中，渐进式训练模式不支持知识蒸馏
    # 只有在没有启用任何蒸馏时才使用渐进式训练模式
    distill_enabled = args.distill_ar or args.distill_cfm
    if not distill_enabled:
        # 渐进式训练模式（无知识蒸馏）
        print("启用渐进式训练模式（无知识蒸馏）")
        manual_trainer = ManualProgressiveTrainer(
            config_path=args.config,
            run_name=args.run_name,
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
            # 传递新的知识蒸馏参数
            distill_ar=args.distill_ar,
            distill_cfm=args.distill_cfm,
            pretrained_cfm_ckpt_path=args.pretrained_cfm_ckpt,
            pretrained_ar_ckpt_path=args.pretrained_ar_ckpt,
        )
        
        # 训练单个数据集，添加fp16错误处理
        try:
            log_dir, training_completed = manual_trainer.train_single_dataset(args.dataset_dir)
        except RuntimeError as e:
            if "LayerNormKernelImpl" in str(e) and args.fp16:
                print(f"Error: Encountered LayerNorm error with fp16 in progressive training. Consider disabling fp16 or let the system auto-switch to fp32.")
                # 可以选择重新尝试使用fp32训练
                # 这里我们只是提醒用户，实际的自动切换在Trainer内部处理
            else:
                raise e
    else:
        # 普通训练模式（启用知识蒸馏）
        print("启用普通训练模式（启用知识蒸馏）")
        print("  启用知识蒸馏:")
        if args.distill_ar:
            print("    AR模型蒸馏: 是")
        if args.distill_cfm:
            print("    CFM模型蒸馏: 是")
        trainer = Trainer(
            config_path=args.config,
            pretrained_cfm_ckpt_path=args.pretrained_cfm_ckpt,
            pretrained_ar_ckpt_path=args.pretrained_ar_ckpt,
            data_dir=args.dataset_dir,
            run_name=args.run_name,
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
            # 传递新的知识蒸馏参数
            distill_ar=args.distill_ar,
            distill_cfm=args.distill_cfm,
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
    
    # 知识蒸馏参数
    parser.add_argument('--distill-ar', action='store_true', help='Enable knowledge distillation for AR model')
    parser.add_argument('--distill-cfm', action='store_true', help='Enable knowledge distillation for CFM model')
    
    args = parser.parse_args()
    main(args)