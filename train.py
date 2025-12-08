import os
import sys
os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'
# 设置 MPS 回退到 CPU 的环境变量
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
import random
import numpy as np
import librosa
import yaml
import argparse
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import glob
from tqdm import tqdm
import shutil

from modules.commons import recursive_munch, build_model, load_checkpoint
from optimizers import build_optimizer
from data.ft_dataset import build_ft_dataloader
from hf_utils import load_custom_model_from_hf


class ManualProgressiveTrainer:
    def __init__(self, config_path, run_name, teacher_model_path=None, **kwargs):
        self.config_path = config_path
        self.run_name = run_name
        self.teacher_model_path = teacher_model_path
        # 提取resume_lr参数（如果存在）
        self.resume_lr = kwargs.pop('resume_lr', 0.0)
        self.kwargs = kwargs
        self.current_trainer = None
    
    def train_single_dataset(self, dataset_dir, run_name_suffix=None):
        """训练单个数据集，不进行自动检查点管理"""
        # 构建运行名称
        if run_name_suffix:
            version_run_name = f"{self.run_name}_{run_name_suffix}"
        else:
            # 使用数据集目录名作为后缀
            dataset_name = os.path.basename(os.path.normpath(dataset_dir))
            version_run_name = f"{self.run_name}_{dataset_name}"
        
        # 使用指定的教师模型作为预训练检查点
        pretrained_ckpt = self.teacher_model_path
        
        # 初始化训练器
        self.current_trainer = Trainer(
            config_path=self.config_path,
            pretrained_ckpt_path=pretrained_ckpt,
            data_dir=dataset_dir,
            run_name=version_run_name,
            resume_lr=self.resume_lr,
            **self.kwargs
        )
        
        # 如果指定了教师模型，设置为教师模型用于知识蒸馏
        if self.teacher_model_path:
            self.current_trainer.set_teacher_model(self.teacher_model_path)        
        # 开始训练
        self.current_trainer.train()
        
        # 训练完成后，将ft_model.pth和配置文件拷贝到基础运行目录
        # 使用与V2版本一致的归档逻辑
        copy_final_models(self.current_trainer.log_dir, self.run_name, self.config_path, self.current_trainer.should_copy)
                
        # 返回最终模型路径
        final_model_path = os.path.join(os.path.dirname(self.current_trainer.log_dir), self.run_name, 'ft_model.pth')
        print(f"数据集训练完成。最终模型: {final_model_path}")
        return final_model_path


class Trainer:
    def __init__(self,
                 config_path,
                 pretrained_ckpt_path,
                 data_dir,
                 run_name,
                 batch_size=0,
                 num_workers=0,
                 steps=1000,
                 save_interval=500,
                 max_epochs=1000,
                 device="cuda:0",
                 fp16=True,
                 val_dataset_dir=None,
                 patience=20,
                 validation_interval=50,
                 min_lr=1e-7,
                 lr_adjust_interval=50,
                 initial_lr=1e-5,
                 warmup_steps=1000,
                 teacher_model_path=None,  # 添加教师模型路径参数
                 resume_lr=0.0,  # 添加resume_lr参数，默认值为0.0
                 ):
        self.device = torch.device(device)
        self.fp16 = fp16
        config = yaml.safe_load(open(config_path))
        # 使用传入的run_name，不再添加数据集名称后缀，因为在调用处已经处理过了
        self.log_dir = os.path.join(config['log_dir'], run_name)
        os.makedirs(self.log_dir, exist_ok=True)
        # copy config file to log dir
        shutil.copyfile(config_path, os.path.join(self.log_dir, os.path.basename(config_path)))
        batch_size = config.get('batch_size', 10) if batch_size == 0 else batch_size
        self.batch_size = batch_size  # 保存batch_size为实例属性
        self.max_steps = steps

        self.n_epochs = max_epochs
        self.log_interval = config.get('log_interval', 10)
        self.save_interval = save_interval

        self.sr = config['preprocess_params'].get('sr', 22050)
        self.hop_length = config['preprocess_params']['spect_params'].get('hop_length', 256)
        self.win_length = config['preprocess_params']['spect_params'].get('win_length', 1024)
        self.n_fft = config['preprocess_params']['spect_params'].get('n_fft', 1024)
        preprocess_params = config['preprocess_params']
        
        # 早停机制参数
        self.patience = patience
        self.validation_interval = validation_interval
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
        
        self.should_copy = False # 是否应该归档训练模型
        
        # 学习率调度相关参数
        self.initial_lr = initial_lr  # 初始学习率
        self.min_lr = min_lr      # 最小学习率
        self.lr_adjust_interval = lr_adjust_interval  # 学习率调整间隔
        self.warmup_steps = warmup_steps  # 学习率预热步数
        self.resume_lr = resume_lr  # resume_lr参数
        self.best_train_loss = float('inf')  # 用于学习率调度的最佳训练损失
        # 注意：不要在这里初始化 switched_to_val_scheduler，因为它会在检查点恢复时被设置
        # 但是在 fresh training 的情况下需要初始化
        # 使用 hasattr 检查是否已经设置过，避免覆盖检查点中恢复的值
        if not hasattr(self, 'switched_to_val_scheduler'):
            self.switched_to_val_scheduler = False  # 默认初始化为False
        
        # 按文件名顺序加载数据以确保训练的一致性
        shuffle_data = False  # 默认情况下不打乱数据，按文件名顺序加载
        
        self.train_dataloader = build_ft_dataloader(
            data_dir,
            preprocess_params['spect_params'],
            self.sr,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle_data,
        )
        
        # 初始化验证集数据加载器（如果提供了验证集路径）
        self.val_dataloader = None
        if val_dataset_dir:
            self.val_dataloader = build_ft_dataloader(
                val_dataset_dir,
                preprocess_params['spect_params'],
                self.sr,
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=False,  # 验证集不打乱
            )
        
        self.f0_condition = config['model_params']['DiT'].get('f0_condition', False)
        self.build_sv_model(self.device, config)
        self.build_semantic_fn(self.device, config, fp16)
        if self.f0_condition:
            self.build_f0_fn(self.device, config)
        self.build_converter(self.device, config)
        self.build_vocoder(self.device, config)

        scheduler_params = {
            "warmup_steps": 0,
            "base_lr": self.initial_lr,
        }

        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, stage='DiT')

        _ = [self.model[key].to(self.device) for key in self.model]
        self.model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)

        # initialize optimizers after preparing models for compatibility with FSDP
        self.optimizer = build_optimizer({key: self.model[key] for key in self.model},
                                         lr=float(scheduler_params['base_lr']))

        # Find latest local checkpoint first
        available_checkpoints = glob.glob(os.path.join(self.log_dir, "DiT_epoch_*_step_*.pth"))
        latest_local_checkpoint = ""
        if len(available_checkpoints) > 0:
            latest_local_checkpoint = max(
                available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            earliest_checkpoint = min(
                available_checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )
            # delete the earliest checkpoint if we have more than 2
            if (
                earliest_checkpoint != latest_local_checkpoint
                and len(available_checkpoints) > 2
            ):
                os.remove(earliest_checkpoint)
                print(f"Removed {earliest_checkpoint}")
        
        # Determine which checkpoint to use
        if latest_local_checkpoint:
            # Use local checkpoint if available
            latest_checkpoint = latest_local_checkpoint
            self.checkpoint_type = "local"
            print(f"Found local checkpoint: {latest_checkpoint}")
        elif pretrained_ckpt_path is not None:
            # Use pretrained checkpoint if no local checkpoint
            assert os.path.exists(pretrained_ckpt_path), f"Pretrained checkpoint {pretrained_ckpt_path} not found"
            latest_checkpoint = pretrained_ckpt_path
            # 根据是否有教师模型路径来判断检查点类型
            if teacher_model_path is not None:
                self.checkpoint_type = "teacher"
            else:
                self.checkpoint_type = "pretrained"
        elif config.get('pretrained_model', ''):
            latest_checkpoint = load_custom_model_from_hf("Plachta/Seed-VC", config['pretrained_model'], None)
            self.checkpoint_type = "pretrained"
        else:
            latest_checkpoint = ""
            self.checkpoint_type = "none"
        if os.path.exists(latest_checkpoint):
            # Determine if we should load only parameters (for teacher models) or full checkpoint (for local/resume checkpoints)
            is_teacher_checkpoint = (self.checkpoint_type == "teacher")
            
            # 只有在checkpoint_type为"local"的情况下才从检查点恢复状态，其他情况都视为首次训练
            is_local_checkpoint = (self.checkpoint_type == "local")
            should_restore_from_checkpoint = is_local_checkpoint
            
            self.model, self.optimizer, self.epoch, self.iters = load_checkpoint(
                self.model, self.optimizer, latest_checkpoint,
                load_only_params=not should_restore_from_checkpoint,  # Only load params for non-local checkpoints
                ignore_modules=[],
                is_distributed=False
            )
            
            # When loading from checkpoint, epoch represents the epoch index (0-based)
            # After loading, self.epoch represents the next epoch to train
            # Only increment epoch for DiT_epoch_ checkpoints, not for teacher models or other checkpoints
            if os.path.basename(latest_checkpoint).startswith("DiT_epoch_"):
                # self.epoch += 1
                print(f"Loaded DiT checkpoint from {latest_checkpoint} with validation states")
                print(f"Resuming from epoch {self.epoch}, step {self.iters}")
            else:
                print(f"Loaded teacher/pretrained checkpoint from {latest_checkpoint}")
                if is_teacher_checkpoint:
                    # For teacher checkpoints, we start from scratch (epoch 0, step 0)
                    self.epoch, self.iters = 0, 0
                    print(f"Teacher model loaded, starting from epoch {self.epoch}, step {self.iters}")
                else:
                    print(f"Starting from epoch {self.epoch}, step {self.iters}")
            
            # Load validation related states if they exist in the checkpoint
            # Only for local checkpoints (resume training), others are treated as fresh training
            if should_restore_from_checkpoint:
                try:
                    state = torch.load(latest_checkpoint, map_location="cpu")
                    if 'best_val_loss' in state:
                        self.best_val_loss = state['best_val_loss']
                    if 'patience_counter' in state:
                        if self.resume_lr > 0.0:
                            #强制设置 恢复学习率时， 也强制重置 早停耐心计数器。
                            print("Using resume_lr > 0.0, forcing patience_counter to 0")
                            self.patience_counter = 0
                        else:
                            self.patience_counter = state['patience_counter']
                    if 'ema_loss' in state:
                        self.ema_loss = state['ema_loss']
                    else:
                        # For checkpoints without ema_loss (older versions), initialize it
                        self.ema_loss = 0
                                    
                    # 恢复学习率调度器切换状态（如果存在）
                    if 'switched_to_val_scheduler' in state:
                        self.switched_to_val_scheduler = state['switched_to_val_scheduler']
                        print(f"Restored switched_to_val_scheduler state: {self.switched_to_val_scheduler}")
                    else:
                        self.switched_to_val_scheduler = False
                                
                    # 恢复学习率相关状态
                    if 'best_train_loss' in state:
                        self.best_train_loss = state['best_train_loss']
                    # 区分新训练和断点续训练的情况
                    if os.path.basename(latest_checkpoint).startswith("DiT_epoch_"):
                        # warmup_steps始终使用命令行参数
                        self.warmup_steps = warmup_steps
                        
                        # 不再从检查点中读取已预热步数，而是动态计算
                        # 计算当前已预热的步数
                        current_passed_warmup_steps = min(self.iters, warmup_steps)
                        print(f"Current passed warmup steps: {current_passed_warmup_steps}")
                        
                        # 计算剩余预热步数
                        remaining_warmup_steps = max(0, warmup_steps - current_passed_warmup_steps)
                        print(f"Calculated remaining warmup steps: {remaining_warmup_steps} (command line warmup_steps: {warmup_steps})")
                        
                        # 如果已经预热的步数大于等于命令行的warmup_steps，表示预热已完成
                        if current_passed_warmup_steps >= warmup_steps:
                            print(f"Warmup already completed. Passed steps: {current_passed_warmup_steps}, command line warmup_steps: {warmup_steps}")

                        # min_lr始终使用命令行参数
                        self.min_lr = min_lr
                        
                        # 处理当前学习率
                        if 'current_lr' in state:
                            # 如果指定了大于0的resume_lr值，则强制使用该值作为当前学习率
                            if self.resume_lr > 0.0:
                                checkpoint_current_lr = self.resume_lr
                                print(f"Using resume_lr ({self.resume_lr}) to override checkpoint current_lr ({state['current_lr']})")
                            else:
                                checkpoint_current_lr = state['current_lr']
                                            
                            self.need_adjust_current_lr = True
                            # 学习率调度器切换状态已在上面恢复，这里不需要重复恢复
                            
                            # 首先检查是否需要根据最小学习率调整
                            # 如果检查点中的当前学习率比命令行的最小学习率低，需要调整
                            if checkpoint_current_lr < min_lr:
                                print(f"Warning: Checkpoint current_lr ({checkpoint_current_lr}) is lower than command line min_lr ({min_lr}). Will adjust to min_lr.")
                                # 保存需要调整的信息
                                self.adjusted_current_lr = min_lr
                                # 使用调整后的学习率
                                adjusted_lr = self.adjusted_current_lr
                                self.target_initial_lr = adjusted_lr
                                # self.checkpoint_current_lr = adjusted_lr
                                print(f"Adjusted current_lr from {checkpoint_current_lr} to {adjusted_lr} due to min_lr constraint.")
                            else:
                                # 不需要调整，正常处理initial_lr
                                # self.need_adjust_current_lr = True
                                self.adjusted_current_lr = checkpoint_current_lr
                                
                                # target_initial_lr永远使用命令行的设定
                                self.target_initial_lr = initial_lr
                                
                                # checkpoint_current_lr需要根据具体情况调整
                                # 如果命令行指定的initial_lr比检查点中的当前学习率低，则使用命令行的initial_lr
                                # 否则，应恢复中断时的学习率
                                if initial_lr < checkpoint_current_lr:
                                    # self.checkpoint_current_lr = initial_lr
                                    # self.need_adjust_current_lr = True
                                    self.adjusted_current_lr = initial_lr
                                    print(f"Warning: Command line initial_lr ({initial_lr}) is lower than min_lr ({checkpoint_current_lr}).current_lr Will adjust to initial_lr.")
                                # else:
                                #     # 恢复中断时的学习率
                                #     self.checkpoint_current_lr = checkpoint_current_lr
                                #     # 在这种情况下，保持原来的need_adjust_current_lr和adjusted_current_lr状态
                        else:
                            # 检查点中没有current_lr，使用命令行参数
                            self.target_initial_lr = initial_lr
                            # self.checkpoint_current_lr = None
                            self.need_adjust_current_lr = False
                            self.adjusted_current_lr = None
                        # initial_lr始终使用命令行参数
                        self.initial_lr = initial_lr
                    else:
                        # 新训练或教师模型：始终使用命令行参数
                        self.warmup_steps = warmup_steps
                        self.initial_lr = initial_lr
                        self.min_lr = min_lr
                        self.target_initial_lr = initial_lr  # 使用传入的initial_lr参数作为目标学习率
                        # self.checkpoint_current_lr = None  # 新训练没有检查点当前学习率
                        self.need_adjust_current_lr = False  # 新训练不需要调整当前学习率
                        self.adjusted_current_lr = None  # 新训练没有调整后的学习率
                        
                    # 恢复优化器的学习率状态
                    if 'scheduler' in state and hasattr(self.optimizer, 'load_scheduler_state_dict'):
                        self.optimizer.load_scheduler_state_dict(state['scheduler'])
                    print(f"Loaded checkpoint from {latest_checkpoint} with validation states")
                    if os.path.basename(latest_checkpoint).startswith("DiT_epoch_"):
                        print(f"Resuming from epoch {self.epoch}, step {self.iters}")
                    else:
                        print(f"Starting from epoch {self.epoch}, step {self.iters}")
                except Exception as e:
                    print(f"Warning: Could not load validation states from checkpoint: {e}")
                    # Reset validation states if they couldn't be loaded
                    self.best_val_loss = float('inf')
                    self.patience_counter = 0
                    self.ema_loss = 0
                    self.switched_to_val_scheduler = False  # 初始化为False
            else:
                # For non-local checkpoints (teacher/pretrained/none), initialize as fresh training
                self.best_val_loss = float('inf')
                self.patience_counter = 0
                self.ema_loss = 0
                self.best_train_loss = float('inf')
                self.warmup_steps = warmup_steps  # 使用传入的warmup_steps参数
                self.initial_lr = initial_lr  # 使用传入的initial_lr参数
                self.min_lr = min_lr  # 使用传入的min_lr参数
                self.target_initial_lr = initial_lr  # 使用传入的initial_lr参数作为目标学习率
                # self.checkpoint_current_lr = None  # 新训练没有检查点当前学习率
                self.need_adjust_current_lr = False  # 新训练不需要调整当前学习率
                self.adjusted_current_lr = None  # 新训练没有调整后的学习率
                self.switched_to_val_scheduler = False  # 新训练初始化为False
            
            # 重新初始化优化器和学习率调度器以确保正确的学习率设置
            # 先重建优化器以使用正确的初始学习率
            self.optimizer = build_optimizer({key: self.model[key] for key in self.model},
                                             lr=float(self.target_initial_lr))
            # 再初始化学习率调度器
            self._init_lr_scheduler()
            
            # # 如果有检查点当前学习率且不为None，需要调整优化器的学习率
            # # if self.checkpoint_current_lr is not None and self.target_initial_lr != self.checkpoint_current_lr:
            # if self.target_initial_lr is not None:
            #     # 设置优化器的当前学习率为目标学习率
            #     for param_group in self.optimizer.optimizers['cfm'].param_groups:
            #         param_group['lr'] = self.target_initial_lr
            #     # print(f"Adjusted optimizer learning rate from {self.checkpoint_current_lr} to {self.target_initial_lr}")
            #     print(f"Adjusted optimizer learning rate to initial_lr: {self.target_initial_lr}")
            
            # 如果是断点续训练，需要将学习率调度器的状态设置到当前step
            if should_restore_from_checkpoint:
                # 将学习率调度器快进到当前step
                for step in range(self.iters):
                    if step < self.warmup_steps:
                        # 预热阶段
                        for key in self.lr_schedulers.keys():
                            self.lr_schedulers[key].step()
                    elif step == self.warmup_steps:
                        # 预热结束，开始余弦退火
                        self.cosine_scheduler.last_epoch = 0
                        self.cosine_scheduler.step()
                    else:
                        # 余弦退火阶段
                        self.cosine_scheduler.step()
                        
             # 如果需要根据最小学习率调整，也要更新优化器的学习率
            if self.need_adjust_current_lr and self.adjusted_current_lr is not None:
                for param_group in self.optimizer.optimizers['cfm'].param_groups:
                    param_group['lr'] = self.adjusted_current_lr
                print(f"Adjusted optimizer learning rate to saved lr: {self.adjusted_current_lr}")
            
            # Ensure deterministic behavior by setting seeds based on current state after loading checkpoint
            seed = 1234 + self.iters
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Also set seed for built-in hash randomization
            os.environ['PYTHONHASHSEED'] = str(seed)
        else:
            self.epoch, self.iters = 0, 0
            self.ema_loss = 0  # Initialize ema_loss for fresh training
            print("Failed to load any checkpoint, training from scratch.")
            print(f"Starting from epoch {self.epoch}, step {self.iters}")
            
            # Ensure deterministic behavior by setting seeds for fresh training
            seed = 1234 + self.iters
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Also set seed for built-in hash randomization
            os.environ['PYTHONHASHSEED'] = str(seed)

        # 初始化教师模型（用于知识蒸馏）
        self.teacher_model = None

    def _init_lr_scheduler(self):
        """初始化学习率调度器"""
        # 为每个优化器创建预热和余弦退火调度器
        self.lr_schedulers = {}
        # 保存warmup_steps到局部变量，避免Lambda函数捕获错误的值
        warmup_steps = self.warmup_steps
        for key in self.optimizer.optimizers.keys():
            # 创建LambdaLR用于预热
            # 使用默认参数确保捕获正确的warmup_steps值
            warmup_lambda = lambda step, ws=warmup_steps: min(1.0, float(step + 1) / float(ws)) if step < ws else 1.0
            self.lr_schedulers[key] = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer.optimizers[key],
                lr_lambda=warmup_lambda
            )
        
        # 创建余弦退火调度器
        self.cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer.optimizers['cfm'],
            T_max=self.max_steps - self.warmup_steps,
            eta_min=self.min_lr
        )
        
        # 移除了switched_to_val_scheduler标志，因为我们现在直接修改余弦退火的学习率
    
    def _adjust_learning_rate(self, loss):
        """根据训练步骤调整学习率（预热+余弦退火+验证损失动态调整）"""
        # 获取当前学习率
        current_lr = self.optimizer.optimizers['cfm'].param_groups[0]['lr']
        
        # 应用预热和余弦退火调度
        if self.iters < self.warmup_steps:
            # 预热阶段
            for key in self.lr_schedulers.keys():
                self.lr_schedulers[key].step()
        elif self.iters == self.warmup_steps:
            # 预热结束，开始余弦退火
            self.cosine_scheduler.last_epoch = 0
            self.cosine_scheduler.step()
        else:
            # 余弦退火阶段
            # 直接应用余弦退火调度
            self.cosine_scheduler.step()
            
        # 获取新的学习率
        new_lr = self.optimizer.optimizers['cfm'].param_groups[0]['lr']
        
        # 每隔一定step才打印一次学习率信息，避免过于频繁的打印
        if self.iters % self.lr_adjust_interval == 0:
            # 使用更高的精度显示学习率，避免因精度问题导致的误判
            lr_diff = abs(new_lr - current_lr)
            # 只有当学习率变化足够大时才认为发生了调整（避免浮点数精度问题）
            if lr_diff > 1e-12:
                print(f"Learning rate adjusted from {current_lr:.2e} to {new_lr:.2e}")
            else:
                # 在断点续训练时，仍然打印当前学习率以提供反馈
                if self.iters >= self.warmup_steps:
                    print(f"Learning rate remains at {new_lr:.2e} (cosine annealing phase)")
                else:
                    print(f"Learning rate remains at {new_lr:.2e} (warmup phase)")
                # 调试信息：显示实际的学习率差异
                if lr_diff > 0:
                    print(f"  Debug: Actual learning rate difference: {lr_diff:.2e}")

    def build_sv_model(self, device, config):
        from modules.campplus.DTDNN import CAMPPlus
        self.campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
        campplus_sd_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
        campplus_sd = torch.load(campplus_sd_path, map_location='cpu')
        self.campplus_model.load_state_dict(campplus_sd)
        self.campplus_model.eval()
        self.campplus_model.to(self.device)
        self.sv_fn = self.campplus_model

    def build_f0_fn(self, device, config):
        from modules.rmvpe import RMVPE
        model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        self.rmvpe = RMVPE(model_path, is_half=False, device=self.device)
        self.f0_fn = self.rmvpe

    def build_converter(self, device, config):
        from modules.openvoice.api import ToneColorConverter
        ckpt_converter, config_converter = load_custom_model_from_hf("myshell-ai/OpenVoiceV2", "converter/checkpoint.pth", "converter/config.json")
        self.tone_color_converter = ToneColorConverter(config_converter, device=self.device)
        self.tone_color_converter.load_ckpt(ckpt_converter)
        self.tone_color_converter.model.eval()
        se_db_path = load_custom_model_from_hf("Plachta/Seed-VC", "se_db.pt", None)
        self.se_db = torch.load(se_db_path, map_location='cpu')

    def build_vocoder(self, device, config):
        vocoder_type = config['model_params']['vocoder']['type']
        vocoder_name = config['model_params']['vocoder'].get('name', None)
        if vocoder_type == 'bigvgan':
            from modules.bigvgan import bigvgan
            self.bigvgan_model = bigvgan.BigVGAN.from_pretrained(vocoder_name, use_cuda_kernel=False)
            self.bigvgan_model.remove_weight_norm()
            self.bigvgan_model = self.bigvgan_model.eval().to(self.device)
            vocoder_fn = self.bigvgan_model
        elif vocoder_type == 'hifigan':
            from modules.hifigan.generator import HiFTGenerator
            from modules.hifigan.f0_predictor import ConvRNNF0Predictor
            hift_config = yaml.safe_load(open('configs/hifigan.yml', 'r'))
            hift_path = load_custom_model_from_hf("FunAudioLLM/CosyVoice-300M", 'hift.pt', None)
            self.hift_gen = HiFTGenerator(**hift_config['hift'],
                                          f0_predictor=ConvRNNF0Predictor(**hift_config['f0_predictor']))
            self.hift_gen.load_state_dict(torch.load(hift_path, map_location='cpu'))
            self.hift_gen.eval()
            self.hift_gen.to(self.device)
            vocoder_fn = self.hift_gen
        else:
            raise ValueError(f"Unsupported vocoder type: {vocoder_type}")
        self.vocoder_fn = vocoder_fn

    def set_teacher_model(self, teacher_checkpoint):
        """设置教师模型用于知识蒸馏"""
        if teacher_checkpoint and os.path.exists(teacher_checkpoint):
            print(f"正在加载教师模型: {teacher_checkpoint}")
            # 加载教师模型（不更新其参数）
            self.teacher_model = build_model(self.model_params, stage='DiT')
            _ = [self.teacher_model[key].to(self.device) for key in self.teacher_model]
            # 初始化教师模型的缓存
            self.teacher_model.cfm.estimator.setup_caches(max_batch_size=self.batch_size, max_seq_length=8192)
            self.teacher_model, _, _, _ = load_checkpoint(
                self.teacher_model, None, teacher_checkpoint,
                load_only_params=True,  # 只加载模型参数
                ignore_modules=[],
                is_distributed=False
            )
            _ = [self.teacher_model[key].eval() for key in self.teacher_model]
            print(f"教师模型加载完成: {teacher_checkpoint}")
        else:
            print("未找到教师模型检查点，将不使用知识蒸馏")

    def build_semantic_fn(self, device, config, fp16=True):
        speech_tokenizer_type = config['model_params']['speech_tokenizer'].get('type', 'cosyvoice')
        if speech_tokenizer_type == 'whisper':
            from transformers import AutoFeatureExtractor, WhisperModel
            whisper_model_name = config['model_params']['speech_tokenizer']['name']
            # 让WhisperModel自动选择合适的数据类型，根据设备支持情况
            if fp16:
                # 如果启用fp16，尝试使用float16精度加载模型
                print(f"正在尝试使用fp16精度加载Whisper模型到{self.device}设备...")
                try:
                    self.whisper_model = WhisperModel.from_pretrained(whisper_model_name, torch_dtype=torch.float16).to(self.device)
                    model_dtype = self.whisper_model.encoder.dtype
                    print(f"Whisper模型已加载到{self.device}设备，使用模型默认数据类型: {model_dtype}")
                except Exception as e:
                    print(f"警告: 在{self.device}设备上无法使用fp16精度加载Whisper模型: {e}")
                    print(f"正在回退到float32精度加载Whisper模型...")
                    self.whisper_model = WhisperModel.from_pretrained(whisper_model_name, torch_dtype=torch.float32).to(self.device)
                    model_dtype = self.whisper_model.encoder.dtype
                    self.fp16 = False
                    print(f"信息: 已将内部fp16标志设置为False，以保持一致性")
                    print(f"Whisper模型已加载到{self.device}设备，使用float32数据类型")
                # 检查模型实际加载的数据类型是否与用户指定的fp16值一致
                if model_dtype != (torch.float16 if self.fp16 else torch.float32):
                    print(f"信息: Whisper模型已根据设备特性自动切换数据类型，从{'float16' if self.fp16 else 'float32'}切换到{model_dtype}")
                    # 当模型自动切换数据类型时，将内部fp16标志设置为False，以避免后续处理中的不一致
                    if model_dtype == torch.float32:
                        self.fp16 = False
                        print(f"信息: 已将内部fp16标志设置为False，以保持一致性")
                else:
                    print(f"信息: Whisper模型已按用户指定的fp16设置加载，数据类型为{model_dtype}")
            else:
                # 如果不启用fp16，强制使用float32
                print(f"正在使用float32精度加载Whisper模型到{self.device}设备...")
                self.whisper_model = WhisperModel.from_pretrained(whisper_model_name, torch_dtype=torch.float32).to(self.device)
                print(f"Whisper模型已加载到{self.device}设备，使用float32数据类型")
            self.whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(whisper_model_name)
            # remove decoder to save memory
            del self.whisper_model.decoder

            def reload_whisper_model():
                """重新加载Whisper模型为float32精度"""
                self.fp16 = False
                print(f"信息: 已将内部fp16标志设置为False，以保持一致性")
                # 重新加载模型为float32精度
                whisper_model_name = config['model_params']['speech_tokenizer']['name']
                self.whisper_model = WhisperModel.from_pretrained(whisper_model_name, torch_dtype=torch.float32).to(self.device)
                print(f"信息: 已成功回退到float32精度并重新加载模型")
            
            def semantic_fn(waves_16k):
                ori_inputs = self.whisper_feature_extractor(
                    [w16k.cpu().numpy() for w16k in waves_16k],
                    return_tensors="pt",
                    return_attention_mask=True,
                    sampling_rate=16000,
                )
                ori_input_features = self.whisper_model._mask_input_features(
                    ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
                ).to(self.device)
                with torch.no_grad():
                    # 确保输入数据类型与模型兼容，避免CPU/MPS设备上的LayerNorm错误
                    if self.device.type == "cpu":
                        # CPU设备可以尝试使用模型的数据类型，但如果出现问题则回退到float32
                        try:
                            encoder_input = ori_input_features.to(self.whisper_model.encoder.dtype)
                        except:
                            print(f"警告: 在CPU设备上无法使用模型的原生数据类型，已自动转换为float32")
                            encoder_input = ori_input_features.to(torch.float32)
                    elif self.device.type == "mps":
                        # MPS设备可以尝试使用模型的数据类型，但如果出现问题则回退到float32
                        try:
                            encoder_input = ori_input_features.to(self.whisper_model.encoder.dtype)
                        except:
                            print(f"警告: 在MPS设备上无法使用模型的原生数据类型，已自动转换为float32")
                            encoder_input = ori_input_features.to(torch.float32)
                    else:
                        encoder_input = ori_input_features.to(self.whisper_model.encoder.dtype)
                    
                    # 执行模型推理，处理可能的LayerNorm错误
                    try:
                        ori_outputs = self.whisper_model.encoder(
                            encoder_input,
                            head_mask=None,
                            output_attentions=False,
                            output_hidden_states=False,
                            return_dict=True,
                        )
                    except RuntimeError as e:
                        if "LayerNormKernelImpl" in str(e) and self.device.type == "cpu":
                            print(f"警告: 在CPU设备上使用fp16时遇到LayerNorm错误，正在回退到float32精度...")
                            # 回退到float32精度重新加载模型
                            reload_whisper_model()
                            # 重新处理输入特征
                            encoder_input = ori_input_features.to(torch.float32)
                            ori_outputs = self.whisper_model.encoder(
                                encoder_input,
                                head_mask=None,
                                output_attentions=False,
                                output_hidden_states=False,
                                return_dict=True,
                            )
                            print(f"信息: 已成功回退到float32精度并重新执行推理")
                        else:
                            # 如果不是预期的LayerNorm错误，则重新抛出异常
                            raise e
                S_ori = ori_outputs.last_hidden_state.to(torch.float32)
                S_ori = S_ori[:, :waves_16k.size(-1) // 320 + 1]
                return S_ori

        elif speech_tokenizer_type == 'xlsr':
            from transformers import (
                Wav2Vec2FeatureExtractor,
                Wav2Vec2Model,
            )
            model_name = config['model_params']['speech_tokenizer']['name']
            output_layer = config['model_params']['speech_tokenizer']['output_layer']
            self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
            self.wav2vec_model = Wav2Vec2Model.from_pretrained(model_name)
            self.wav2vec_model.encoder.layers = self.wav2vec_model.encoder.layers[:output_layer]
            self.wav2vec_model = self.wav2vec_model.to(self.device)
            self.wav2vec_model = self.wav2vec_model.eval()
            # 根据fp16参数决定是否使用half精度
            if fp16:
                print(f"正在尝试使用fp16精度加载Wav2Vec2模型到{self.device}设备...")
                self.wav2vec_model = self.wav2vec_model.half()
                model_dtype = next(self.wav2vec_model.parameters()).dtype
                print(f"Wav2Vec2模型已转换为fp16精度，当前数据类型: {model_dtype}")
                if self.device.type == "cpu" and model_dtype != torch.float32:
                    print(f"警告: 在CPU设备上使用非float32类型({model_dtype})，可能存在兼容性问题")
                elif self.device.type == "cpu":
                    # 当模型在CPU上运行且使用float32时，将内部fp16标志设置为False
                    if model_dtype == torch.float32:
                        self.fp16 = False
                        print(f"信息: 已将内部fp16标志设置为False，以保持一致性")
            else:
                print(f"使用默认精度加载Wav2Vec2模型到{self.device}设备...")
                model_dtype = next(self.wav2vec_model.parameters()).dtype
                print(f"Wav2Vec2模型已加载，当前数据类型: {model_dtype}")
                if self.device.type == "cpu":
                    print(f"信息: Wav2Vec2模型已根据CPU设备使用推荐的数据类型")

            def semantic_fn(waves_16k):
                ori_waves_16k_input_list = [waves_16k[bib].cpu().numpy() for bib in range(len(waves_16k))]
                ori_inputs = self.wav2vec_feature_extractor(
                    ori_waves_16k_input_list,
                    return_tensors="pt",
                    return_attention_mask=True,
                    padding=True,
                    sampling_rate=16000
                ).to(self.device)
                with torch.no_grad():
                    # 根据fp16参数决定是否使用half精度
                    if fp16:
                        print(f"正在将输入数据转换为fp16精度...")
                        input_values = ori_inputs.input_values.half()
                    else:
                        input_values = ori_inputs.input_values
                    ori_outputs = self.wav2vec_model(
                        input_values,
                    )
                S_ori = ori_outputs.last_hidden_state.float()
                return S_ori
        else:
            raise ValueError(f"Unsupported speech tokenizer type: {speech_tokenizer_type}")
        self.semantic_fn = semantic_fn

    def train_one_step(self, batch):
        waves, mels, wave_lengths, mel_input_length = batch

        B = waves.size(0)
        target_size = mels.size(2)
        target = mels
        target_lengths = mel_input_length

        # get speaker embedding
        if self.sr != 22050:
            waves_22k = torchaudio.functional.resample(waves, self.sr, 22050)
            wave_lengths_22k = (wave_lengths.float() * 22050 / self.sr).long()
        else:
            waves_22k = waves
            wave_lengths_22k = wave_lengths
        se_batch = self.tone_color_converter.extract_se(waves_22k, wave_lengths_22k)

        # Use deterministic selection of reference speaker embedding
        # Generate a deterministic index based on current iteration and batch size
        ref_se_indices = [(self.iters * B + i) % len(self.se_db) for i in range(B)]
        ref_se_idx = torch.tensor(ref_se_indices)
        ref_se = self.se_db[ref_se_idx].to(self.device)

        # convert
        converted_waves_22k = self.tone_color_converter.convert(
            waves_22k, wave_lengths_22k, se_batch, ref_se
        ).squeeze(1)

        if self.sr != 22050:
            converted_waves = torchaudio.functional.resample(converted_waves_22k, 22050, self.sr)
        else:
            converted_waves = converted_waves_22k

        waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
        wave_lengths_16k = (wave_lengths.float() * 16000 / self.sr).long()
        converted_waves_16k = torchaudio.functional.resample(converted_waves, self.sr, 16000)

        # extract S_alt (perturbed speech tokens)
        S_ori = self.semantic_fn(waves_16k)
        S_alt = self.semantic_fn(converted_waves_16k)

        if self.f0_condition:
            F0_ori = self.rmvpe.infer_from_audio_batch(waves_16k)
        else:
            F0_ori = None

        # interpolate speech token to match acoustic feature length
        alt_cond, _, alt_codes, alt_commitment_loss, alt_codebook_loss = (
            self.model.length_regulator(S_alt, ylens=target_lengths, f0=F0_ori)
        )
        ori_cond, _, ori_codes, ori_commitment_loss, ori_codebook_loss = (
            self.model.length_regulator(S_ori, ylens=target_lengths, f0=F0_ori)
        )
        if alt_commitment_loss is None:
            alt_commitment_loss = 0
            alt_codebook_loss = 0
            ori_commitment_loss = 0
            ori_codebook_loss = 0

        # deterministically set a length as prompt
        # Generate deterministic random-like values based on current iteration
        torch.manual_seed(self.iters)
        prompt_len_max = target_lengths - 1
        prompt_len = (torch.rand([B], device=alt_cond.device) * prompt_len_max).floor().long()
        prompt_len[torch.rand([B], device=alt_cond.device) < 0.1] = 0

        # for prompt cond token, use ori_cond instead of alt_cond
        cond = alt_cond.clone()
        for bib in range(B):
            cond[bib, :prompt_len[bib]] = ori_cond[bib, :prompt_len[bib]]

        # diffusion target
        common_min_len = min(target_size, cond.size(1))
        target = target[:, :, :common_min_len]
        cond = cond[:, :common_min_len]
        target_lengths = torch.clamp(target_lengths, max=common_min_len)
        x = target

        # style vectors are extracted from the prompt only
        feat_list = []
        for bib in range(B):
            # Check if we're using MPS device and handle accordingly
            if self.device.type == "mps":
                # MPS doesn't support ComplexFloat type, compute on CPU and move back
                wave_cpu = waves_16k[bib:bib + 1, :wave_lengths_16k[bib]].cpu()
                feat = kaldi.fbank(
                    wave_cpu,
                    num_mel_bins=80,
                    dither=0,
                    sample_frequency=16000
                )
                # Move the result back to MPS device
                feat = feat.to(self.device)
            else:
                feat = kaldi.fbank(
                    waves_16k[bib:bib + 1, :wave_lengths_16k[bib]],
                    num_mel_bins=80,
                    dither=0,
                    sample_frequency=16000
                )
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        y_list = []
        with torch.no_grad():
            for feat in feat_list:
                y = self.sv_fn(feat.unsqueeze(0))
                y_list.append(y)
        y = torch.cat(y_list, dim=0)

        loss, _ = self.model.cfm(x, target_lengths, prompt_len, cond, y)

        # 如果有教师模型，添加知识蒸馏损失
        distill_loss = 0
        if self.teacher_model is not None:
            with torch.no_grad():
                # 使用教师模型生成目标输出
                teacher_loss, teacher_output = self.teacher_model.cfm(x, target_lengths, prompt_len, cond, y)
            # 计算学生模型和教师模型输出之间的蒸馏损失
            distill_loss = F.mse_loss(_, teacher_output.detach())

        loss_total = (
            loss +
            (alt_commitment_loss + ori_commitment_loss) * 0.05 +
            (ori_codebook_loss + alt_codebook_loss) * 0.15 +
            distill_loss * 0.5  # 知识蒸馏损失权重
        )

        self.optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.cfm.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(self.model.length_regulator.parameters(), 10.0)
        self.optimizer.step('cfm')
        self.optimizer.step('length_regulator')
        
        # 使用新的学习率调整机制
        self._adjust_learning_rate(loss_total.item())

        return loss.detach().item()

    def validate_one_step(self, batch):
        """在验证集上评估一个批次"""
        # Ensure deterministic behavior by setting seeds based on current state
        seed = 1234 + self.iters
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Also set seed for built-in hash randomization
        os.environ['PYTHONHASHSEED'] = str(seed)
            
        with torch.no_grad():
            waves, mels, wave_lengths, mel_input_length = batch
            
            B = waves.size(0)
            target_size = mels.size(2)
            target = mels
            target_lengths = mel_input_length
            
            # get speaker embedding
            if self.sr != 22050:
                waves_22k = torchaudio.functional.resample(waves, self.sr, 22050)
                wave_lengths_22k = (wave_lengths.float() * 22050 / self.sr).long()
            else:
                waves_22k = waves
                wave_lengths_22k = wave_lengths
            se_batch = self.tone_color_converter.extract_se(waves_22k, wave_lengths_22k)
            
            # Use deterministic selection of reference speaker embedding
            # Generate a deterministic index based on current iteration and batch size
            ref_se_indices = [(self.iters * B + i) % len(self.se_db) for i in range(B)]
            ref_se_idx = torch.tensor(ref_se_indices)
            ref_se = self.se_db[ref_se_idx].to(self.device)
            
            # convert
            converted_waves_22k = self.tone_color_converter.convert(
                waves_22k, wave_lengths_22k, se_batch, ref_se
            ).squeeze(1)
            
            if self.sr != 22050:
                converted_waves = torchaudio.functional.resample(converted_waves_22k, 22050, self.sr)
            else:
                converted_waves = converted_waves_22k
            
            waves_16k = torchaudio.functional.resample(waves, self.sr, 16000)
            wave_lengths_16k = (wave_lengths.float() * 16000 / self.sr).long()
            converted_waves_16k = torchaudio.functional.resample(converted_waves, self.sr, 16000)
            
            # extract S_alt (perturbed speech tokens)
            S_ori = self.semantic_fn(waves_16k)
            S_alt = self.semantic_fn(converted_waves_16k)
            
            if self.f0_condition:
                F0_ori = self.rmvpe.infer_from_audio_batch(waves_16k)
            else:
                F0_ori = None
            
            # interpolate speech token to match acoustic feature length
            alt_cond, _, alt_codes, alt_commitment_loss, alt_codebook_loss = (
                self.model.length_regulator(S_alt, ylens=target_lengths, f0=F0_ori)
            )
            ori_cond, _, ori_codes, ori_commitment_loss, ori_codebook_loss = (
                self.model.length_regulator(S_ori, ylens=target_lengths, f0=F0_ori)
            )
            if alt_commitment_loss is None:
                alt_commitment_loss = 0
                alt_codebook_loss = 0
                ori_commitment_loss = 0
                ori_codebook_loss = 0
            
            # deterministically set a length as prompt
            # Generate deterministic random-like values based on current iteration
            torch.manual_seed(self.iters)
            prompt_len_max = target_lengths - 1
            prompt_len = (torch.rand([B], device=alt_cond.device) * prompt_len_max).floor().long()
            prompt_len[torch.rand([B], device=alt_cond.device) < 0.1] = 0
            
            # for prompt cond token, use ori_cond instead of alt_cond
            cond = alt_cond.clone()
            for bib in range(B):
                cond[bib, :prompt_len[bib]] = ori_cond[bib, :prompt_len[bib]]
            
            # diffusion target
            common_min_len = min(target_size, cond.size(1))
            target = target[:, :, :common_min_len]
            cond = cond[:, :common_min_len]
            target_lengths = torch.clamp(target_lengths, max=common_min_len)
            x = target
            
            # style vectors are extracted from the prompt only
            feat_list = []
            for bib in range(B):
                # Check if we're using MPS device and handle accordingly
                if self.device.type == "mps":
                    # MPS doesn't support ComplexFloat type, compute on CPU and move back
                    wave_cpu = waves_16k[bib:bib + 1, :wave_lengths_16k[bib]].cpu()
                    feat = kaldi.fbank(
                        wave_cpu,
                        num_mel_bins=80,
                        dither=0,
                        sample_frequency=16000
                    )
                    # Move the result back to MPS device
                    feat = feat.to(self.device)
                else:
                    feat = kaldi.fbank(
                        waves_16k[bib:bib + 1, :wave_lengths_16k[bib]],
                        num_mel_bins=80,
                        dither=0,
                        sample_frequency=16000
                    )
                feat = feat - feat.mean(dim=0, keepdim=True)
                feat_list.append(feat)
            y_list = []
            with torch.no_grad():
                for feat in feat_list:
                    y = self.sv_fn(feat.unsqueeze(0))
                    y_list.append(y)
            y = torch.cat(y_list, dim=0)
            
            loss, _ = self.model.cfm(x, target_lengths, prompt_len, cond, y)
            
            loss_total = (
                loss +
                (alt_commitment_loss + ori_commitment_loss) * 0.05 +
                (ori_codebook_loss + alt_codebook_loss) * 0.15
            )
            
            return loss_total.detach().item()
    
    def validate(self):
        """在整个验证集上评估模型"""
        if self.val_dataloader is None:
            return None
            
        _ = [self.model[key].eval() for key in self.model]
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_dataloader:
                batch = [b.to(self.device) for b in batch]
                loss = self.validate_one_step(batch)
                total_loss += loss
                num_batches += 1
        
        _ = [self.model[key].train() for key in self.model]
        
        if num_batches > 0:
            avg_loss = total_loss / num_batches
            return avg_loss
        else:
            return None
    
    def _save_best_model(self):
        """保存最佳模型"""
        print(f"Saving best model with validation loss: {self.best_val_loss}")
        state = {
            'net': {key: self.model[key].state_dict() for key in self.model},
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.optimizer.scheduler_state_dict(),
            'iters': self.iters,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'ema_loss': self.ema_loss,
            'best_train_loss': self.best_train_loss,
            'current_lr': self.optimizer.optimizers['cfm'].param_groups[0]['lr'],  # 保存当前学习率
        }
        os.makedirs(self.log_dir, exist_ok=True)
        save_path = os.path.join(self.log_dir, 'best_model.pth')
        torch.save(state, save_path)
        print(f"Best model saved at {save_path}")

    def _save_checkpoint(self):
        print('Saving..')
        # Save with the epoch index (0-based indexing)
        # According to the specification, we save first, then increment self.epoch
        save_epoch = self.epoch
        # After saving, increment epoch for next iteration
        # Note: This increment is moved here to comply with the specification
        # that says to save first, then increment self.epoch
        
        state = {
            'net': {key: self.model[key].state_dict() for key in self.model},
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.optimizer.scheduler_state_dict(),
            'iters': self.iters,
            'epoch': save_epoch,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'ema_loss': self.ema_loss,
            'best_train_loss': self.best_train_loss,
            'current_lr': self.optimizer.optimizers['cfm'].param_groups[0]['lr'],  # 保存当前学习率
        }
        save_path = os.path.join(
            self.log_dir,
            f'DiT_epoch_{save_epoch:05d}_step_{self.iters:05d}.pth'
        )
        torch.save(state, save_path)

        # find all checkpoints and remove old ones
        checkpoints = glob.glob(os.path.join(self.log_dir, 'DiT_epoch_*.pth'))
        if len(checkpoints) > 2:
            checkpoints.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            for cp in checkpoints[:-2]:
                os.remove(cp)

    def train_one_epoch(self):
        # self.epoch represents the epoch index (0-based)
        # For logging, we use the same value
        # print_epoch = self.epoch
            
        _ = [self.model[key].train() for key in self.model]
        for i, batch in enumerate(tqdm(self.train_dataloader)):
            
            stepInEpoch = self.iters % len(self.train_dataloader)
            if stepInEpoch == 0 and self.iters != 0:
                self.epoch += 1
            if stepInEpoch != i:
                continue
            self.iters += 1
            
            if self.iters > self.max_steps or self.epoch >= self.n_epochs:
                self.should_stop = True
                self.should_copy = False
                print("Reached max epochs (or max steps), stopping training")
                return
            
            # Ensure deterministic behavior by setting seeds based on current state
            seed = 1234 + self.iters
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            
            # Also set seed for built-in hash randomization
            os.environ['PYTHONHASHSEED'] = str(seed)
            
            batch = [b.to(self.device) for b in batch]
            loss = self.train_one_step(batch)
            self.ema_loss = (
                self.ema_loss * self.loss_smoothing_rate + loss * (1 - self.loss_smoothing_rate)
                if self.iters > 0 else loss
            )
            if self.iters % self.log_interval == 0:
                print(f"epoch {self.epoch}, step {self.iters}, loss: {self.ema_loss}")

            # 验证和早停机制
            if self.val_dataloader and self.iters - self.warmup_steps > 0 and (self.iters - self.warmup_steps) % self.validation_interval == 0 :
                val_loss = self.validate()
                if val_loss is not None:
                    print(f"Validation loss at step {self.iters}:【{val_loss}】")
                    
                    # 早停机制
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        print(f"Improved validation loss: 【{val_loss}】")
                        # 保存最佳模型
                        self._save_best_model()
                    else:
                        self.patience_counter += 1
                        print(f"No improvement in validation loss. Patience: {self.patience_counter}/{self.patience}")
                        
                        if self.patience_counter >= self.patience:
                            print(f"Early stopping triggered at step {self.iters}")
                            self.should_stop = True
                            self.should_copy = True
                            self._save_checkpoint()
                            return
                    
                    # 在预热阶段结束后，根据验证损失情况决定是否手动调整学习率
                    if self.iters >= self.warmup_steps:
                        # 获取当前学习率
                        old_lr = self.optimizer.optimizers['cfm'].param_groups[0]['lr']
                        
                        # 当patience_counter达到一定阈值时，手动降低学习率
                        # 每当patience_counter增加时，按0.5的比例降低学习率
                        switch_patience = max(1, self.patience // 4)  # 使用早停耐心值的四分之一作为切换耐心值
                        if self.patience_counter >= switch_patience and self.patience_counter % max(2, switch_patience // 2) == 0:
                            # 获取当前学习率并降低它
                            current_lr = self.optimizer.optimizers['cfm'].param_groups[0]['lr']
                            new_lr = max(current_lr * 0.5, self.min_lr)
                            
                            # 手动设置新的学习率
                            for param_group in self.optimizer.optimizers['cfm'].param_groups:
                                param_group['lr'] = new_lr
                            
                            print(f"Learning rate manually adjusted from {current_lr:.2e} to 《{new_lr:.2e}》 based on validation loss plateau")
                        else:
                            new_lr = self.optimizer.optimizers['cfm'].param_groups[0]['lr']
                            print(f"Learning rate remains at 《{new_lr:.2e}》 (cosine annealing phase - validation loss monitoring)")

            if self.iters >= self.max_steps:
                self.should_copy = False
                self.should_stop = True
                print("Reached max steps, stopping training")
                self._save_checkpoint()
                return # 不归档，只保存检查点。
                
            if self.iters % self.save_interval == 0 or self.iters >= self.max_steps:
                self._save_checkpoint()
            
            # 检查是否应该早停
            if self.should_stop:
                break

    def train(self):
        # Ensure ema_loss is initialized
        if not hasattr(self, 'ema_loss'):
            self.ema_loss = 0
        self.loss_smoothing_rate = 0.99
        # self.epoch represents the epoch index (0-based)
        # For logging, we use the same value
        start_epoch = self.epoch
        print(f"Starting training from epoch {start_epoch}, step {self.iters}")
        
        # Ensure deterministic behavior by setting seeds based on current state
        seed = 1234 + self.iters
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        
        # Also set seed for built-in hash randomization
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        for epoch in range(self.n_epochs):
            # Train for one epoch
            self.train_one_epoch()
            # According to the specification, we save first, then increment self.epoch
            # The increment was previously handled inside train_one_epoch during checkpoint saving
            # But we still need to increment at the end of each epoch if no checkpoint was saved
            # self.epoch += 1
            if self.should_stop:
                break


def copy_final_models(log_dir, run_name, config_path=None, training_completed=False):
    """共用的文件拷贝函数，用于将最终模型和配置文件拷贝到基础运行目录"""
    # 只有在训练完成时才拷贝最终模型文件
    if not training_completed:
        return
    # 保持与V2版本一致的归档逻辑
    base_log_dir = os.path.join(os.path.dirname(log_dir), run_name)
    os.makedirs(base_log_dir, exist_ok=True)
    
    # 拷贝最终模型
    # 优先使用最佳模型文件，如果没有则使用最新的检查点
    best_model_path = os.path.join(log_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        shutil.copy2(best_model_path, os.path.join(base_log_dir, 'ft_model.pth'))
        print(f"已将最佳模型拷贝到: {os.path.join(base_log_dir, 'ft_model.pth')}")
    else:
        # 查找最新的检查点文件
        checkpoints = glob.glob(os.path.join(log_dir, 'DiT_epoch_*_step_*.pth'))
        if checkpoints:
            latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            shutil.copy2(latest_checkpoint, os.path.join(base_log_dir, 'ft_model.pth'))
            print(f"已将最终模型拷贝到: {os.path.join(base_log_dir, 'ft_model.pth')}")
    
    # 拷贝配置文件
    if config_path:
        config_file = os.path.join(log_dir, os.path.basename(config_path))
        if os.path.exists(config_file):
            shutil.copy2(config_file, os.path.join(base_log_dir, os.path.basename(config_path)))
            print(f"已将配置文件拷贝到: {os.path.join(base_log_dir, os.path.basename(config_path))}")


def main(args):
    # 设置随机种子以确保训练的可重现性
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Also set seed for built-in hash randomization
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Check FORCE_CPU environment variable
    force_cpu = os.environ.get('FORCE_CPU', '0') == '1'
    
    if force_cpu:
        device_str = "cpu"
    elif hasattr(torch, 'xpu') and torch.xpu.is_available():
        device_str = "xpu"
    elif torch.backends.mps.is_available():
        device_str = "mps"
    else:
        device_str = f"cuda:{args.gpu}" if args.gpu else "cuda:0"
        
    print(f"Using device: {device_str}")
    print(f"FP16 precision: {'Enabled' if args.fp16 else 'Disabled'}")
    if device_str == "cpu" and args.fp16:
        print("注意: 在CPU设备上启用FP16精度时，模型会根据设备特性自动切换到兼容的数据类型")
        print("注意: 在CPU上使用FP16精度可能不会带来性能提升，且可能导致兼容性问题")
        print("注意: 如果遇到问题，建议关闭FP16精度(--fp16 False)")
    
    # 如果config参数为空，则使用默认值
    if not args.config:
        args.config = './configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml'
    
    # 检查是否启用知识蒸馏
    if args.distill:
        # 渐进式训练模式（启用知识蒸馏）
        print("启用渐进式训练模式（启用知识蒸馏）")
        manual_trainer = ManualProgressiveTrainer(
            config_path=args.config,
            run_name=args.run_name,
            teacher_model_path=args.pretrained_ckpt,  # 使用pretrained_ckpt作为教师模型路径
            batch_size=args.batch_size,
            steps=args.max_steps,
            max_epochs=args.max_epochs,
            save_interval=args.save_every,
            num_workers=args.num_workers,
            device=device_str,
            fp16=args.fp16,
            val_dataset_dir=args.val_dataset_dir,
            patience=args.patience,
            validation_interval=args.validation_interval,
            min_lr=args.min_lr,
            lr_adjust_interval=args.lr_adjust_interval,
            initial_lr=args.initial_lr,
            warmup_steps=args.warmup_steps,
            resume_lr=args.resume_lr,
        )
        
        # 训练数据集
        manual_trainer.train_single_dataset(args.dataset_dir, None)
    else:
        # 普通训练模式（不启用知识蒸馏）
        # 使用数据集目录名作为后缀，保持与渐进式训练一致的输出目录结构
        dataset_name = os.path.basename(os.path.normpath(args.dataset_dir))
        version_run_name = f"{args.run_name}_{dataset_name}"
        
        trainer = Trainer(
            config_path=args.config,
            pretrained_ckpt_path=args.pretrained_ckpt,
            data_dir=args.dataset_dir,
            run_name=version_run_name,
            batch_size=args.batch_size,
            steps=args.max_steps,
            max_epochs=args.max_epochs,
            save_interval=args.save_every,
            num_workers=args.num_workers,
            device=device_str,
            fp16=args.fp16,
            val_dataset_dir=args.val_dataset_dir,
            patience=args.patience,
            validation_interval=args.validation_interval,
            min_lr=args.min_lr,
            lr_adjust_interval=args.lr_adjust_interval,
            initial_lr=args.initial_lr,
            warmup_steps=args.warmup_steps,
            resume_lr=args.resume_lr,
            teacher_model_path=args.pretrained_ckpt if args.distill else None,  # 根据distill参数决定是否传递教师模型路径
        )
        trainer.train()
        
        # 训练完成后，将ft_model.pth和配置文件拷贝到基础运行目录
        # 使用与V2版本一致的归档逻辑
        copy_final_models(trainer.log_dir, args.run_name, args.config, trainer.should_copy)    
if __name__ == '__main__':
    # Set multiprocessing start method to avoid 'Too many open files' error on macOS
    if sys.platform == 'darwin':  # macOS
        try:
            mp.set_start_method('fork', force=True)
        except RuntimeError:
            print("Failed to set multiprocessing start method to 'fork', using default.")
    elif sys.platform == 'win32':
        mp.freeze_support()
        mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml')
    parser.add_argument('--pretrained-ckpt', type=str, default=None)
    parser.add_argument('--dataset-dir', type=str, default='/path/to/dataset')
    parser.add_argument('--run-name', type=str, default='my_run')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--max-epochs', type=int, default=1000)
    parser.add_argument('--save-every', type=int, default=500)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--fp16', action='store_true', help='Use fp16 precision')
            
    # 早停机制参数
    parser.add_argument('--val-dataset-dir', type=str, default=None, help='Validation dataset directory')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--validation-interval', type=int, default=50, help='Validation interval in steps')
    
    # 学习率调度参数
    parser.add_argument('--min-lr', type=float, default=1e-7, help='Minimum learning rate')
    parser.add_argument('--lr-adjust-interval', type=int, default=50, help='Interval (in steps) for learning rate adjustment print logs')
    parser.add_argument('--initial-lr', type=float, default=1e-5, help='Initial learning rate')
    parser.add_argument('--warmup-steps', type=int, default=1000, help='Number of warmup steps')
    parser.add_argument('--resume-lr', type=float, default=0.0, help='Resume learning rate for resuming training from checkpoint')

    parser.add_argument("--gpu", type=int, help="Which GPU id to use", default=0)
    

    parser.add_argument('--distill', action='store_true',
                       help='Enable knowledge distillation')    
    args = parser.parse_args()
    main(args)