##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import argparse
import datetime
import os
import re

import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler

from model.config import cfg
from model.modeling.build_model import JointModel, JointInvModel
from model.data.transforms.data_preprocess import TestTransforms
from model.data.crack_dataset import CrackDataSetTest, TTICrackDataSetTest
from model.data.retinal_dataset import RetinalDataSetTest
from model.engine.inference import inference_for_ss, inference_tti_building
from model.utils.misc import fix_model_state_dict
from model.data.transforms.transforms import FactorResize
from torch.multiprocessing import Pool, Process, set_start_method

def test(args, cfg):
    device = torch.device(cfg.DEVICE)
    # model = Model(cfg).to(device)
    if cfg.MODEL.SR_SEG_INV:
        model = JointInvModel(cfg).to(device)
        # print(f'------------Model Architecture-------------\n\n<Network SS>\n{model.segmentation_model}\n\n<Network SR>\n{model.sr_model}')
    else:
        model = JointModel(cfg).to(device)
        if model.sr_model == "DSRL":
            # print(f'------------Model Architecture-------------\n\n<Network parallel>\n{model.parallel_model}')
            pass
        else:
            # print(f'------------Model Architecture-------------\n\n<Network SR>\n{model.sr_model}\n\n<Network SS>\n{model.segmentation_model}')
            pass

    model.load_state_dict(fix_model_state_dict(torch.load(args.trained_model, map_location=lambda storage, loc:storage)))
    if 'indOptim' in cfg.OUTPUT_DIR:
        if cfg.MODEL.SR == 'KBPN':
            path = 'weights/KBPN_30000iter.pth'
        elif cfg.MODEL.SR == 'DBPN':
            path = 'weights/DBPN_30000iter.pth'
        print(f'load {path}')
        m_key, u_key = model.load_state_dict(fix_model_state_dict(torch.load(path)), strict=False)
        assert len(u_key) == 0, (f'unexpected_keys are exist.\n {u_key}')
    model.eval()

    print('Loading Datasets...')
    test_transforms = TestTransforms(cfg)
    sr_transforms = FactorResize(cfg.MODEL.SCALE_FACTOR, cfg.SOLVER.DOWNSCALE_INTERPOLATION)
    if args.tti_crack_dataset:
        test_dataset = TTICrackDataSetTest(cfg, cfg.DATASET.TEST_IMAGE_DIR, args.batch_size, transforms=test_transforms, )
    elif 'RetinalSeg' in cfg.DATASET.TEST_IMAGE_DIR:
        test_dataset = RetinalDataSetTest(cfg, cfg.DATASET.TEST_IMAGE_DIR, cfg.DATASET.TEST_MASK_DIR, 
                                        cfg.DATASET.TEST_BLURED_DIR, cfg.DATASET.TEST_BLURED_NAME, args.batch_size,
                                        transforms=test_transforms, sr_transforms=sr_transforms)
    else:
        test_dataset = CrackDataSetTest(cfg, cfg.DATASET.TEST_IMAGE_DIR, cfg.DATASET.TEST_MASK_DIR, 
                                        cfg.DATASET.TEST_BLURED_DIR, cfg.DATASET.TEST_BLURED_NAME, args.batch_size,
                                        transforms=test_transforms, sr_transforms=sr_transforms)
    sampler = SequentialSampler(test_dataset)
    batch_sampler = BatchSampler(sampler=sampler, batch_size=args.batch_size, drop_last=False)
    test_loader = DataLoader(test_dataset, num_workers=args.num_workers, batch_sampler=batch_sampler)

    if args.num_gpus > 1:
        device_ids = list(range(args.num_gpus))
        print("device_ids:",device_ids)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    with torch.no_grad(): 
        if args.tti_crack_dataset:
            inference_tti_building(args, cfg, model, test_loader)
        else:
            inference_for_ss(args, cfg, model, test_loader)

def main():
    parser = argparse.ArgumentParser(description='Crack Segmentation with Blind Super Resolution(CSBSR)')
    parser.add_argument('test_dir', type=str, default=None)
    parser.add_argument('iter_or_weight_name', type=str, default=None)

    parser.add_argument('--output_dirname', type=str, default=None)
    parser.add_argument('--config_file', type=str, default=None, metavar='FILE')
    parser.add_argument('--test_blured_name', type=str, default=None)    
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--test_aiu', type=bool, default=True)
    parser.add_argument('--test_surface_distance', action="store_true")
    parser.add_argument('--test_classification_metrics', action="store_true")
    parser.add_argument('--sf_save_image', action="store_false", help="If you do not want the output images to be saved, you should turn off this flag.")
    parser.add_argument('--origin_img_size', type=bool, default=True)
    parser.add_argument('--tti_crack_dataset', type=bool, default=False)
    parser.add_argument('--trained_model', type=str, default=None)
    parser.add_argument('--wandb_flag', type=bool, default=True)
    parser.add_argument('--wandb_prj_name', type=str, default="CSBSR_test")
    args = parser.parse_args()

    if bool(re.search(r"[^0-9]", args.iter_or_weight_name)):
        # not iteration number
        _out_dir = args.iter_or_weight_name
        model_fname = args.iter_or_weight_name
    else:
        # iteration number
        _out_dir = f"iter_{args.iter_or_weight_name}"
        model_fname = f"iteration_{args.iter_or_weight_name}"    

    check_args = [('config_file', f'{args.test_dir}config.yaml'), 
     ('output_dirname', f'{args.test_dir}eval_AIU/{_out_dir}'),
     ('trained_model', f'{args.test_dir}model/{model_fname}.pth'), 
    ]

    if args.origin_img_size:
        img_size = cfg.INPUT.IMAGE_SIZE # keep default


    for check_arg in check_args:
        arg_name = f'args.{check_arg[0]}'
        if exec(arg_name) == None:
            exec(f'{arg_name} = "{check_arg[1]}"')

    cuda = torch.cuda.is_available()
    if cuda:
        torch.backends.cudnn.benchmark = True

    if len(args.config_file) > 0:
        print('Configration file is loaded from {}'.format(args.config_file))
        cfg.merge_from_file(args.config_file)
    
    if args.test_blured_name != None:
        cfg.DATASET.TEST_BLURED_NAME = args.test_blured_name
        args.output_dirname = f'{args.test_dir}/eval_AIU/compe_blur/{_out_dir}_{args.test_blured_name}'

    if args.tti_crack_dataset:
        args.output_dirname = f'{args.test_dir}/eval_AIU/tti_bulinding/{_out_dir}_size64'
        cfg.DATASET.TEST_IMAGE_DIR = 'datasets/tti_crack/blured_image/'
        img_size = [64, 64] #128

    if 'RetinalSeg' in cfg.DATASET.TEST_IMAGE_DIR and args.origin_img_size:
        img_size = [560, 560]

    cfg.OUTPUT_DIR = args.output_dirname
    if args.origin_img_size:
        print(f'Size of input image is {img_size}.')
        cfg.INPUT.IMAGE_SIZE = img_size

    cfg.freeze()

    print('Running with config:\n{}'.format(cfg))

    test(args, cfg)


if __name__ == '__main__':
    set_start_method('spawn')
    main()