# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import torch
from torch.optim import lr_scheduler
from .WarmupMultiStepLR import WarmupMultiStepLR


def make_scheduler(cfg,optimizer):
    
    if cfg.SOLVER.WARMUP:
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
    else:
        scheduler = lr_scheduler.StepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA)
    return scheduler
