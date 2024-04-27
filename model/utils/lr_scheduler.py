##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

from torch.optim.lr_scheduler import MultiStepLR
import torch

class WarmupMultiStepLR(MultiStepLR):
    def __init__(self, cfg, optimizer, milestones, gamma=0.1, warmup_factor=1.0 / 3,
                warmup_iters=500, last_epoch=-1):
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.init_lr = [cfg.SOLVER.LR]
        super().__init__(optimizer, milestones, gamma, last_epoch)

    def get_lr(self):
        # lr = super().get_lr()
        lr = self.init_lr
        if self.last_epoch < self.warmup_iters:
            alpha = self.last_epoch / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha
            return [l * warmup_factor for l in lr]
        return lr

class UpDownScheduler:
    def __init__(self, pretrain_iter, resume_iter, scheduler_flag):
        self.pretrain_iter = pretrain_iter
        self.resume_iter = resume_iter
        self.scheduler_flag = scheduler_flag

    def __call__(self, _iter):
        _iter_main = _iter - (self.pretrain_iter - 1) + self.resume_iter
        # print('main iter:', _iter_main)
        if 70000 < _iter_main < 95000 and self.scheduler_flag:
            return 10
        else:
            return 1