##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import numpy as np
import argparse
import os
import random
import shutil
import datetime
import socket

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler
from torch.optim.lr_scheduler import LambdaLR

from model.utils.sync_batchnorm import convert_model
from model.config import cfg
from model.engine.trainer import do_train, do_pretrain_sr
import torchvision.transforms as transforms
from model.data.transforms.data_preprocess import TrainTransforms, TestTransforms
from model.data.transforms.transforms import FactorResize
from model.modeling.build_model import JointModelWithLoss, JointInvModelWithLoss, SRModelWithLoss
from model.data.crack_dataset import CrackDataSet, SRPretrainDataSet
from model.data.retinal_dataset import RetinalDataSet
from model.utils.misc import str2bool, fix_model_state_dict
from model.data import samplers
from model.utils.lr_scheduler import WarmupMultiStepLR, UpDownScheduler
from torch.multiprocessing import Pool, Process, set_start_method

def train(args, cfg):
    device = torch.device(cfg.DEVICE)

    print('Loading Datasets...')
    train_transforms = TrainTransforms(cfg)
    sr_transforms = FactorResize(cfg.MODEL.SCALE_FACTOR, cfg.SOLVER.DOWNSCALE_INTERPOLATION)
    if cfg.DATASET.ONLY_IMAGES:
        trainval_dataset = SRPretrainDataSet(cfg, cfg.DATASET.TRAIN_IMAGE_DIR, transforms=train_transforms, sr_transforms=sr_transforms)
    elif "RetinalSeg" in cfg.DATASET.TRAIN_IMAGE_DIR:
        trainval_dataset = RetinalDataSet(cfg, cfg.DATASET.TRAIN_IMAGE_DIR, cfg.DATASET.TRAIN_MASK_DIR, transforms=train_transforms, sr_transforms=sr_transforms)
    else:
        trainval_dataset = CrackDataSet(cfg, cfg.DATASET.TRAIN_IMAGE_DIR, cfg.DATASET.TRAIN_MASK_DIR, transforms=train_transforms, sr_transforms=sr_transforms)

    n_samples = len(trainval_dataset) 
    train_size = int(len(trainval_dataset) * cfg.SOLVER.TRAIN_DATASET_RATIO) 
    val_size = n_samples - train_size
    if "RetinalSeg" in cfg.DATASET.TRAIN_IMAGE_DIR:
        train_size, val_size = 12, 3
    print(f"Train dataset size: {train_size}, Validation dataset size: {val_size}")
    train_dataset, val_dataset = torch.utils.data.random_split(trainval_dataset, [train_size, val_size])
    # print(val_dataset.__dict__)

    sampler = torch.utils.data.RandomSampler(train_dataset)
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler=sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=False) #True)
    batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iterations=cfg.SOLVER.MAX_ITER)
    train_loader = DataLoader(train_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler, pin_memory=True)

    eval_sampler = SequentialSampler(val_dataset)
    eval_batch_sampler = BatchSampler(sampler=eval_sampler, batch_size=cfg.SOLVER.BATCH_SIZE, drop_last=False) # True)
    eval_loader = DataLoader(val_dataset, num_workers=1, batch_sampler=eval_batch_sampler, pin_memory=False) # num_workers = 1 for memory retrenchment

    print('Building model...')
    if cfg.DATASET.ONLY_IMAGES and cfg.MODEL.SR != "DSRL":
        # print(SRModelWithLoss.__mro__)
        model = SRModelWithLoss(cfg, sr_transforms).to(device)
        # print(f'------------Model Architecture-------------\n\n<Network SR>\n{model.sr_model}\n')
        scheduler_flag = False # for KBPN paper
    else:
        scheduler_flag = cfg.SOLVER.SCHEDULER
        if cfg.MODEL.SR_SEG_INV:
            model = JonitInvModelWithLoss(cfg, num_train_ds=train_size, resume_iter=args.resume_iter, sr_transforms=sr_transforms).to(device)
            # print(f'------------Model Architecture-------------\n\n<Network SS>\n{model.segmentation_model}\n\n<Network SR>\n{model.sr_model}')
        else:
            model = JointModelWithLoss(cfg, num_train_ds=train_size, resume_iter=args.resume_iter, sr_transforms=sr_transforms).to(device)
            pass
            if cfg.MODEL.SR == "DSRL":
                # print(f'------------Model Architecture-------------\n{model}')
                pass
            else:
                # print(f'------------Model Architecture-------------\n\n<Network SR>\n{model.sr_model}\n\n<Network SS>\n{model.segmentation_model}')
                pass

    if cfg.MODEL.OPTIMIZER == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, betas=(0.9, 0.999), eps=1e-8) # betas and eps is referenced from KBPN.
    elif cfg.MODEL.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(filter(lambda p:p.requires_grad, model.parameters()), lr=cfg.SOLVER.LR, momentum=0.9, weight_decay=5e-4)

    sch_func = UpDownScheduler(cfg.SOLVER.SR_PRETRAIN_ITER[1], args.resume_iter, scheduler_flag)
    scheduler = LambdaLR(optimizer, lr_lambda = sch_func)



    if args.resume_iter > 0:
        print('Resume from {}'.format(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter))))
        state_dict = fix_model_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(args.resume_iter))))
        model.load_state_dict(state_dict , strict=False)

    if cfg.SOLVER.SYNC_BATCHNORM:
        model = convert_model(model).to(device)
    
    if args.num_gpus > 1:
        device_ids = list(range(args.num_gpus))
        # device_ids.insert(0, device_ids.pop(cfg.DEVICE_NUM))
        print("device_ids:",device_ids)
        model = torch.nn.DataParallel(model, device_ids=device_ids)  # primaly gpu is last device.
    
    if cfg.DATASET.ONLY_IMAGES:
        do_pretrain_sr(args, cfg, model, optimizer, scheduler, train_loader, eval_loader)
    else:
        do_train(args, cfg, model, optimizer, scheduler, train_loader, eval_loader)

def main():
    parser = argparse.ArgumentParser(description='Crack Segmentation with Blind Super Resolution(CSBSR)')
    parser.add_argument('--config_file', type=str, default='./config/configs_train.yaml', metavar='FILE', help='path to config file')
    parser.add_argument('--output_dirname', type=str, default='', help='')
    parser.add_argument('--num_workers', type=int, default=2, help='')
    parser.add_argument('--log_step', type=int, default=50, help='')
    parser.add_argument('--save_step', type=int, default=2000)
    parser.add_argument('--eval_step', type=int, default=2000)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--mixed_precision', type=str2bool, default=False)
    parser.add_argument('--wandb_flag', type=str2bool, default=True)
    parser.add_argument('--resume_iter', type=int, default=0)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--wandb_prj_name', type=str, default="CSBSR") # CSBSR_main_ECCV2022, KBPN_verkondo_iso

    # only local project
    parser.add_argument('--local', type=bool, default=False)

    args = parser.parse_args()

    torch.manual_seed(cfg.SEED)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(cfg.SEED)

    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)
    
    if "_ds_" in cfg.DATASET.TRAIN_IMAGE_DIR:
        cfg.INPUT.IMAGE_SIZE = int(cfg.INPUT.IMAGE_SIZE / cfg.MODEL.SCALE_FACTOR )

    if args.local:
        train_img_dir_list = (cfg.DATASET.TRAIN_IMAGE_DIR).split('/')
        train_img_dir_list[1] =  train_img_dir_list[1] + f'_{socket.gethostname()}'
        cfg.DATASET.TRAIN_IMAGE_DIR = '/'.join(train_img_dir_list)
        train_msk_dir_list = (cfg.DATASET.TRAIN_MASK_DIR).split('/')
        train_msk_dir_list[1] =  train_msk_dir_list[1] + f'_{socket.gethostname()}'
        cfg.DATASET.TRAIN_MASK_DIR = '/'.join(train_msk_dir_list)

    cfg.freeze()

    if not args.debug and args.resume_iter == 0:
        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
        shutil.copy2(args.config_file, os.path.join(cfg.OUTPUT_DIR, 'config.yaml'))

    train(args, cfg)

if __name__ == '__main__':
    # os.environ['WANDB_MODE'] = 'offline'
    set_start_method('spawn')
    main()


