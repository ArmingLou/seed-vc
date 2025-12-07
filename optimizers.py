#coding:utf-8
import os, sys
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from functools import reduce
from torch.optim import AdamW

class MultiOptimizer:
    def __init__(self, optimizers={}, schedulers={}):
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.keys = list(optimizers.keys())
        self.param_groups = reduce(lambda x,y: x+y, [v.param_groups for v in self.optimizers.values()])

    def state_dict(self):
        state_dicts = [(key, self.optimizers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def scheduler_state_dict(self):
        state_dicts = [(key, self.schedulers[key].state_dict())\
                       for key in self.keys]
        return state_dicts

    def load_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.optimizers[key].load_state_dict(val)
            except:
                print("Unloaded %s" % key)

    def load_scheduler_state_dict(self, state_dict):
        for key, val in state_dict:
            try:
                self.schedulers[key].load_state_dict(val)
            except:
                print("Unloaded %s" % key)

    def step(self, key=None, scaler=None):
        keys = [key] if key is not None else self.keys
        _ = [self._step(key, scaler) for key in keys]

    def _step(self, key, scaler=None):
        if scaler is not None:
            scaler.step(self.optimizers[key])
            scaler.update()
        else:
            self.optimizers[key].step()

    def zero_grad(self, key=None):
        if key is not None:
            self.optimizers[key].zero_grad()
        else:
            _ = [self.optimizers[key].zero_grad() for key in self.keys]

    def scheduler(self, *args, key=None):
        if key is not None:
            self.schedulers[key].step(*args)
        else:
            _ = [self.schedulers[key].step_batch(*args) for key in self.keys]

def define_scheduler(optimizer, params):
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=params['gamma'])

    return scheduler

def build_optimizer(model_dict, lr, type='AdamW'):
    optim = {}
    for key, model in model_dict.items():
        model_parameters = model.parameters()
        parameters_names = []
        parameters_names.append(
            [
                name_param_pair[0]
                for name_param_pair in model.named_parameters()
            ]
        )
        if type == 'AdamW':
            optim[key] = AdamW(
                model_parameters,
                lr=lr,
                betas=(0.9, 0.98),
                eps=1e-6,
                weight_decay=0.01,
            )
        else:
            raise ValueError('Unknown optimizer type: %s' % type)

    schedulers = dict([(key, torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.999996))
                       for key, opt in optim.items()])

    multi_optim = MultiOptimizer(optim, schedulers)
    return multi_optim

class MinLRExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self, optimizer, gamma, min_lr=1e-5):
        self.min_lr = min_lr
        super().__init__(optimizer, gamma)

    def get_lr(self):
        lrs = super().get_lr()
        return [max(lr, self.min_lr) for lr in lrs]

def build_single_optimizer(model, lr, warmup_steps=1000, max_steps=10000, min_lr=1e-7):
    """构建优化器和学习率调度器（预热+余弦退火）"""
    model_parameters = model.parameters()
    parameters_require_grad = filter(lambda p: p.requires_grad, model_parameters)
    optim = AdamW(
        parameters_require_grad,
        lr=lr,
        betas=(0.9, 0.98),
        eps=1e-6,
        weight_decay=0.01,
    )

    # 创建预热调度器
    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lr_lambda=lambda step: min(1.0, float(step + 1) / float(warmup_steps)) if step < warmup_steps else 1.0
    )
    
    # 创建余弦退火调度器
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim,
        T_max=max_steps - warmup_steps,
        eta_min=min_lr
    )
    
    # 使用SequentialLR组合预热和余弦退火
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    return optim, scheduler