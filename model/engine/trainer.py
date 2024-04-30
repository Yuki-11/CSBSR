##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Toyota Technological Institute
## Author: Yuki Kondo
## Copyright (c) 2024
## yuki.kondo.ab@gmail.com
##
## This source code is licensed under the Apache License license found in the
## LICENSE file in the root directory of this source tree 
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import time
import os
import datetime
from copy import copy
import numpy as np
from tqdm import tqdm

import torch

from model.utils.estimate_metrics import PSNR, SSIM, IoU 
from model.utils.save_output import save_img
import torchvision.transforms as transforms

import wandb

def do_train(args, cfg, model, optimizer, scheduler, train_loader, eval_loader):
    max_iter = len(train_loader) + args.resume_iter
    trained_time = 0
    tic = time.time()
    end = time.time()

    psnr = PSNR()
    ssim = SSIM()
    iou = IoU()

    logging_sr_loss = 0
    logging_segment_loss = 0
    logging_fa_loss = 0

    free_1st_stage_model_params = True

    ## --- wandb setting https://docs.wandb.ai/integrations/pytorch --- ##
    if args.wandb_flag:
        if "RetinalSeg" in cfg.DATASET.TRAIN_IMAGE_DIR:
            wandb.init(config=cfg, project=args.wandb_prj_name+'_Retinal')
        elif "RetinalSeg" in cfg.DATASET.TRAIN_IMAGE_DIR:
            wandb.init(config=cfg, project=args.wandb_prj_name+'_Retinal')
        else:
            wandb.init(config=cfg, project=args.wandb_prj_name)
        wandb.config.update(args)
        # Magic
        wandb.watch(model, log='all')        
        wandb.run.name = cfg.OUTPUT_DIR.replace("output/", "")

    print('Start training!!')
    fix_2nd_stage_model_params(cfg, args, model)
    for iteration, (imgs, sr_targets, segment_targets, kernel_targets) in enumerate(train_loader, args.resume_iter+1):
        free_1st_stage_model_params = fix_1st_stage_model_params(cfg, args, model, free_1st_stage_model_params, iteration)

        model.train()
        optimizer.zero_grad()

        if cfg.MODEL.SR == "DSRL":
            segment_loss, sr_loss, segment_preds, sr_preds, kernel_preds, fa_loss = model(iteration, imgs, sr_targets=sr_targets, segment_targets=segment_targets, kernel_targets=kernel_targets)
            loss, logging_segment_loss, logging_sr_loss, logging_fa_loss = calc_loss4DSRL(segment_loss, logging_segment_loss, sr_loss, logging_sr_loss, fa_loss, logging_fa_loss, iteration, cfg, args)
        else:
            segment_loss, sr_loss, segment_preds, sr_preds, kernel_preds = model(iteration, imgs, sr_targets=sr_targets, segment_targets=segment_targets, kernel_targets=kernel_targets)
            loss, logging_segment_loss, logging_sr_loss = calc_loss(segment_loss, logging_segment_loss, sr_loss, logging_sr_loss, iteration, cfg, args)

        loss.backward()
        optimizer.step()
        scheduler.step()
        
        trained_time += time.time() - end
        end = time.time()
        del loss, segment_loss, sr_loss, imgs, sr_targets, segment_targets
        if cfg.MODEL.SR == "DSRL":
            del fa_loss

        if iteration % args.log_step == 0:
            logging_segment_loss /= args.log_step
            logging_sr_loss /= args.log_step
            logging_fa_loss /= args.log_step
            logging_tot_loss = logging_sr_loss + cfg.SOLVER.TASK_LOSS_WEIGHT * logging_segment_loss
            if cfg.MODEL.SR == "DSRL":
                logging_tot_loss = 0.5 * logging_sr_loss + 0.5 * logging_fa_loss + logging_segment_loss

            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            print('===> Iter: {:07d}, LR: {:.5f}, Cost: {:.2f}s, Eta: {}, Segment_Loss({}): {:.6f}, SR_Loss({}): {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), cfg.SOLVER.SEG_LOSS_FUNC, logging_segment_loss, cfg.SOLVER.SR_LOSS_FUNC, logging_sr_loss))

            alpha = None
            if "Boundary" in cfg.SOLVER.SEG_LOSS_FUNC:
                if args.num_gpus > 1:
                    alpha = model.module.ss_loss_fn.alpha
                else:
                    alpha = model.ss_loss_fn.alpha

            if args.wandb_flag:
                # wandb
                wandb.log({"loss": logging_tot_loss, 
                        f"segment_loss({cfg.SOLVER.SEG_LOSS_FUNC})":logging_segment_loss, 
                        f"sr_loss({cfg.SOLVER.SR_LOSS_FUNC})": logging_sr_loss,
                        'lr': optimizer.param_groups[0]['lr'],
                        'Iteration': iteration,
                        'boundary_alpha': alpha, 
                        })
                if cfg.MODEL.SR == "DSRL":
                    wandb.log({"fa_loss":logging_fa_loss, 
                            'Iteration': iteration,
                            })
                    logging_fa_loss = 0

            logging_segment_loss = logging_sr_loss = logging_tot_loss = 0

            tic = time.time()

        if iteration % args.save_step == 0 and not args.debug:
            model_path = os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(iteration))
            optimizer_path = os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(iteration))

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)

            if args.num_gpus > 1:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

            torch.save(optimizer.state_dict(), optimizer_path)

            print('=====> Save Checkpoint to {}'.format(model_path))

        if iteration % args.eval_step == 0:

            if args.num_gpus > 1:
                model.module.iter_cnt = False
            else:
                model.iter_cnt = False
            
            with torch.no_grad():
                model.eval() # add
                eval_sr_loss = 0
                eval_segment_loss = 0
                data_len = len(eval_loader)

                psnr_scores = np.array([])
                kernel_psnr_scores = np.array([])
                ssim_scores = np.array([])
                iou_scores = np.array([])
                first_set = True

                for imgs, sr_targets, segment_targets, kernel_targets in tqdm(eval_loader):
                    # print(sr_targets.shape, imgs.shape)
                    if cfg.MODEL.SR == "DSRL":
                        segment_loss, sr_loss, segment_preds, sr_preds, kernel_preds, fa_loss = model(iteration, imgs, sr_targets=sr_targets, segment_targets=segment_targets, kernel_targets=kernel_targets)
                        # loss, logging_segment_loss, logging_sr_loss, logging_fa_loss = calc_loss4DSRL(segment_loss, logging_segment_loss, sr_loss, logging_sr_loss, fa_loss, logging_fa_loss, iteration, cfg, args)
                    else:
                        segment_loss, sr_loss, segment_preds, sr_preds, kernel_preds = model(iteration, imgs, sr_targets=sr_targets, segment_targets=segment_targets, kernel_targets=kernel_targets)

                    if sr_loss is None:
                        # sr_loss = model.sr_loss_fn(sr_images, sr_targets).mean(dim=(1,2,3))
                        loss = segment_loss.mean()
                        eval_segment_loss += loss.item()
                    else:
                        segment_loss, sr_loss = segment_loss.mean(), sr_loss.mean()
                        eval_segment_loss += segment_loss.item()
                        eval_sr_loss += sr_loss.item()

                    if not cfg.MODEL.SR_SEG_INV and cfg.MODEL.SCALE_FACTOR != 1:
                        sr_preds[sr_preds>1] = 1 # clipping
                        sr_preds[sr_preds<0] = 0 # clipping
                        kernel_preds[kernel_preds>1] = 1 # clipping
                        kernel_preds[kernel_preds<0] = 0 # clipping
                        psnr_scores = np.append(psnr_scores, psnr(sr_preds, sr_targets.to("cuda")))
                        kernel_psnr_scores = np.append(kernel_psnr_scores, psnr(kernel_preds, kernel_targets.to("cuda")))
                        ssim_scores = np.append(ssim_scores, ssim(sr_preds, sr_targets.to("cuda")))
                    else:
                        psnr_scores = np.append(psnr_scores, 0)
                        kernel_psnr_scores = np.append(kernel_psnr_scores, 0)
                        ssim_scores = np.append(ssim_scores, 0)

                    segment_preds = (segment_preds.to("cuda") >=  torch.Tensor([0.5]).to("cuda")).float()
                    iou_scores = np.append(iou_scores, iou(segment_preds, segment_targets.to("cuda")))

                    # ============================== debug ===================================
                    if first_set:
                        fname = [f'lr{iteration}_{i}.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        save_img(cfg.OUTPUT_DIR+"/pred/", imgs, fname) # debug
                        fname = [f'sr{iteration}_{i}.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        save_img(cfg.OUTPUT_DIR+"/pred/", sr_preds, fname) # debug
                        fname = [f'hr{iteration}_{i}.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        save_img(cfg.OUTPUT_DIR+"/pred/", sr_targets, fname) # debug

                        # print(kernel_preds.shape)
                        # print(torch.amax(kernel_preds, dim=(1, 2, 3), keepdims=True))
                        # kernel_preds = kernel_preds / torch.amax(kernel_targets, dim=(1, 2, 3), keepdims=True)
                        # kernel_preds = torch.clamp(kernel_preds * 80, 0, 1)
                        # print(kernel_targets.size())
                        # kernel_targets = kernel_targets / torch.amax(kernel_targets, dim=(1, 2, 3), keepdims=True)                
                        fname_t = [f'kernel{iteration}_{i}_target.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        fname_p = [f'kernel{iteration}_{i}_pred.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        os.makedirs(os.path.dirname(cfg.OUTPUT_DIR+f"/pred/kernels/gt/"), exist_ok=True)
                        os.makedirs(os.path.dirname(cfg.OUTPUT_DIR+f"/pred/kernels/pred/"), exist_ok=True)
                        for batch_num in range(len(kernel_targets)):
                            # print(kernel_targets.size())
                            kernel = transforms.ToPILImage(mode='L')(kernel_targets[batch_num])
                            fpath = os.path.join(cfg.OUTPUT_DIR+"/pred/kernels/gt/", f"{fname_t[batch_num]}")
                            kernel.save(fpath)

                            kernel = transforms.ToPILImage(mode='L')(kernel_preds[batch_num])
                            fpath = os.path.join(cfg.OUTPUT_DIR+"/pred/kernels/pred/", f"{fname_p[batch_num]}")
                            kernel.save(fpath)

                        fname_t = [f'segment{iteration}_{i}_target.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        fname_p = [f'segment{iteration}_{i}_pred.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        os.makedirs(os.path.dirname(cfg.OUTPUT_DIR+f"/pred/segment/gt/"), exist_ok=True)
                        os.makedirs(os.path.dirname(cfg.OUTPUT_DIR+f"/pred/segment/pred/"), exist_ok=True)
                        for batch_num in range(len(segment_targets)):
                            segment = transforms.ToPILImage(mode='L')(segment_targets[batch_num])
                            fpath = os.path.join(cfg.OUTPUT_DIR+"/pred/segment/gt/", f"{fname_t[batch_num]}")
                            segment.save(fpath)

                            segment = transforms.ToPILImage(mode='L')(segment_preds[batch_num])
                            fpath = os.path.join(cfg.OUTPUT_DIR+"/pred/segment/pred/", f"{fname_p[batch_num]}")
                            segment.save(fpath)

                        first_set = False
                    # # =======================================================================

                eval_segment_loss /= data_len
                eval_sr_loss /= data_len

                print(f"\nestimation result (iter={iteration}):")
                print(f'=====> Segment_Loss({cfg.SOLVER.SEG_LOSS_FUNC}): {eval_segment_loss:.6f}, SR_Loss({cfg.SOLVER.SR_LOSS_FUNC}): {eval_sr_loss:.6f} PSNR:{sum(psnr_scores)/len(psnr_scores):.4f} SSIM:{sum(ssim_scores)/len(ssim_scores):.4f}  PSNR(Kernel):{sum(kernel_psnr_scores)/len(kernel_psnr_scores):.4f} IoU:{sum(iou_scores)/len(iou_scores):.4f}')
                # print(f"PSNR:{sum(psnr_scores)/len(psnr_scores):.4f}  SSIM:{sum(ssim_scores)/len(ssim_scores):.4f}  IoU:{sum(iou_scores)/len(iou_scores):.4f}\n")

                if args.wandb_flag:
                    wandb.log({f"segment_loss_eval({cfg.SOLVER.SEG_LOSS_FUNC})":eval_segment_loss,
                            f"sr_loss_eval({cfg.SOLVER.SR_LOSS_FUNC})": eval_sr_loss,
                            'Iteration': iteration,
                            "PSNR_eval":sum(psnr_scores)/len(psnr_scores),
                            "PSNR(Kernel)_eval":sum(kernel_psnr_scores)/len(kernel_psnr_scores),
                            "SSIM_eval":sum(ssim_scores)/len(ssim_scores),
                            "IoU_eval":sum(iou_scores)/len(iou_scores)
                            })

            if args.num_gpus > 1:
                model.module.iter_cnt = True
            else:
                model.iter_cnt = True

def do_pretrain_sr(args, cfg, model, optimizer, scheduler, train_loader, eval_loader):

    print(len(eval_loader))
    
    max_iter = len(train_loader) + args.resume_iter
    trained_time = 0
    tic = time.time()
    end = time.time()

    psnr = PSNR()
    ssim = SSIM()

    logging_sr_loss = 0

    ## --- wandb setting https://docs.wandb.ai/integrations/pytorch --- ##
    if args.wandb_flag:
        wandb.init(config=cfg, project=args.wandb_prj_name)
        wandb.config.update(args)
        # Magic
        wandb.watch(model, log='all')        
        wandb.run.name = cfg.OUTPUT_DIR.replace("output/", "")

    print('Training Starts')
    for iteration, (imgs, sr_targets, kernel_targets) in enumerate(train_loader, args.resume_iter+1):
        model.train()
        optimizer.zero_grad()

        if cfg.MODEL.SR == "DSRL":
            _, sr_loss, _, sr_preds, kernel_preds, _ = model(iteration, imgs, sr_targets=sr_targets, kernel_targets=kernel_targets.detach())
        else:
            sr_loss, sr_preds, kernel_preds = model(iteration, imgs, sr_targets=sr_targets, kernel_targets=kernel_targets.detach())
        # print(kernel_preds.shape)
        sr_loss = sr_loss.mean()
        logging_sr_loss += sr_loss.item()

        # print(sr_loss.sum(), sr_loss.sum().size())
        sr_loss.backward()
        optimizer.step()   
        scheduler.step()
        
        trained_time += time.time() - end
        end = time.time()

        del sr_loss, imgs, sr_targets, kernel_targets

        if iteration % args.log_step == 0:
            logging_sr_loss /= args.log_step
            eta_seconds = int((trained_time / iteration) * (max_iter - iteration))
            print('===> Iter: {:07d}, LR: {:.5f}, Cost: {:.2f}s, Eta: {}, SR_Loss({}): {:.6f}'.format(iteration, optimizer.param_groups[0]['lr'], time.time() - tic, str(datetime.timedelta(seconds=eta_seconds)), cfg.SOLVER.SR_LOSS_FUNC, logging_sr_loss))

            if args.wandb_flag:
                # wandb
                wandb.log({f"sr_loss({cfg.SOLVER.SR_LOSS_FUNC})": logging_sr_loss,
                        'lr': optimizer.param_groups[0]['lr'],
                        'Iteration': iteration,
                        })

            logging_sr_loss = 0
            tic = time.time()

        if iteration % args.save_step == 0 and not args.debug:
            model_path = os.path.join(cfg.OUTPUT_DIR, 'model', 'iteration_{}.pth'.format(iteration))
            optimizer_path = os.path.join(cfg.OUTPUT_DIR, 'optimizer', 'iteration_{}.pth'.format(iteration))

            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            os.makedirs(os.path.dirname(optimizer_path), exist_ok=True)

            if args.num_gpus > 1:
                torch.save(model.module.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), model_path)

            torch.save(optimizer.state_dict(), optimizer_path)

            print('=====> Save Checkpoint to {}'.format(model_path))

        if iteration % args.eval_step == 0:
            with torch.no_grad():
                model.eval()
                eval_sr_loss = 0
                data_len = len(eval_loader)

                psnr_scores = np.array([])
                kernel_psnr_scores = np.array([])
                ssim_scores = np.array([])
                first_set = True

                for imgs, sr_targets, kernel_targets in  eval_loader:#tqdm(eval_loader):
                    if cfg.MODEL.SR == "DSRL":
                        _, sr_loss, _, sr_preds, kernel_preds, _ = model(iteration, imgs, sr_targets=sr_targets, kernel_targets=kernel_targets)
                    else:
                        sr_loss, sr_preds, kernel_preds = model(iteration, imgs, sr_targets=sr_targets, kernel_targets=kernel_targets)
                    # print(torch.amax(kernel_preds, dim=(1, 2, 3), keepdims=True))

                    sr_preds[sr_preds>1] = 1 # clipping
                    sr_preds[sr_preds<0] = 0 # clipping
                    kernel_preds[kernel_preds>1] = 1 # clipping
                    kernel_preds[kernel_preds<0] = 0 # clipping
                    psnr_scores = np.append(psnr_scores, psnr(sr_preds, sr_targets.to("cuda")))
                    ssim_scores = np.append(ssim_scores, ssim(sr_preds, sr_targets.to("cuda")))
                    kernel_psnr_scores = np.append(kernel_psnr_scores, psnr(kernel_preds, kernel_targets.to("cuda")))
                    eval_sr_loss += sr_loss.mean().item()

                    # ============================== debug ===================================
                    if first_set:
                        fname = [f'lr{iteration}_{i}.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        
                        save_img(cfg.OUTPUT_DIR+"/pred/", imgs, fname) # debug

                        fname = [f'sr{iteration}_{i}.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        save_img(cfg.OUTPUT_DIR+"/pred/", sr_preds, fname) # debug

                        fname = [f'hr{iteration}_{i}.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        save_img(cfg.OUTPUT_DIR+"/pred/", sr_targets, fname) # debug

                        # print(kernel_preds.shape)
                        # print(torch.amax(kernel_preds, dim=(1, 2, 3), keepdims=True))
                        kernel_preds = kernel_preds / torch.amax(kernel_preds, dim=(1, 2, 3), keepdims=True)
                        # print(kernel_targets.size())
                        kernel_targets = kernel_targets / torch.amax(kernel_targets, dim=(1, 2, 3), keepdims=True)
                        
                        fname_t = [f'kernel{iteration}_{i}_target.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        fname_p = [f'kernel{iteration}_{i}_pred.png' for i in range(cfg.SOLVER.BATCH_SIZE)]
                        os.makedirs(os.path.dirname(cfg.OUTPUT_DIR+f"/pred/kernels/gt/"), exist_ok=True)
                        os.makedirs(os.path.dirname(cfg.OUTPUT_DIR+f"/pred/kernels/pred/"), exist_ok=True)
                        for batch_num in range(cfg.SOLVER.BATCH_SIZE):
                            # print(kernel_targets.size())
                            kernel = transforms.ToPILImage(mode='L')(kernel_targets[batch_num])
                            fpath = os.path.join(cfg.OUTPUT_DIR+"/pred/kernels/gt/", f"{fname_t[batch_num]}")
                            kernel.save(fpath)

                            kernel = transforms.ToPILImage(mode='L')(kernel_preds[batch_num])
                            fpath = os.path.join(cfg.OUTPUT_DIR+"/pred/kernels/pred/", f"{fname_p[batch_num]}")
                            kernel.save(fpath)
                    # =======================================================================
                    del sr_loss, imgs, sr_targets, kernel_targets, sr_preds, kernel_preds
                    first_set = False

                eval_sr_loss /= data_len

                print(f"\nestimation result (iter={iteration}):")
                print(f'=====> SR_Loss({cfg.SOLVER.SR_LOSS_FUNC}): {eval_sr_loss:.6f} PSNR:{sum(psnr_scores)/len(psnr_scores):.4f} SSIM:{sum(ssim_scores)/len(ssim_scores):.4f} PSNR(Kernel):{sum(kernel_psnr_scores)/len(kernel_psnr_scores):.4f}')
                # print(f"PSNR:{sum(psnr_scores)/len(psnr_scores):.4f}  SSIM:{sum(ssim_scores)/len(ssim_scores):.4f}  IoU:{sum(iou_scores)/len(iou_scores):.4f}\n")

                if args.wandb_flag:
                    wandb.log({f"sr_loss_eval({cfg.SOLVER.SR_LOSS_FUNC})": eval_sr_loss,
                            'Iteration': iteration,
                            "PSNR_eval":sum(psnr_scores)/len(psnr_scores),
                            "SSIM_eval":sum(ssim_scores)/len(ssim_scores),
                            "PSNR(Kernel)_eval":sum(kernel_psnr_scores)/len(kernel_psnr_scores),
                            })



def calc_loss(segment_loss, logging_segment_loss, sr_loss, logging_sr_loss, iteration, cfg, args):
    segment_loss = segment_loss.mean()
    logging_segment_loss += segment_loss.item()

    if sr_loss != None:
        sr_loss = sr_loss.mean()
        logging_sr_loss += sr_loss.item()

    if cfg.MODEL.SCALE_FACTOR == 1 or cfg.MODEL.SR == "bicubic":
        loss = segment_loss
    elif cfg.MODEL.JOINT_LEARNING:
        if cfg.SOLVER.TASK_LOSS_WEIGHT == -1:
            w_task = increase_w_task(cfg, args, iteration)
            loss = (1 - w_task) * sr_loss + w_task * segment_loss
        else:
            loss = (1 - cfg.SOLVER.TASK_LOSS_WEIGHT) * sr_loss + cfg.SOLVER.TASK_LOSS_WEIGHT * segment_loss
        loss = calc_pretrain_loss(loss, segment_loss, sr_loss, iteration, cfg)
    elif not cfg.MODEL.JOINT_LEARNING:
        if not cfg.MODEL.SR_SEG_INV:
            loss = segment_loss
        else:
            loss = sr_loss
        loss = calc_pretrain_loss(loss, segment_loss, sr_loss, iteration, cfg)

    return loss, logging_segment_loss, logging_sr_loss

def calc_pretrain_loss(loss, segment_loss, sr_loss, iteration, cfg):
    if cfg.SOLVER.SR_PRETRAIN_ITER[0] <= iteration < cfg.SOLVER.SR_PRETRAIN_ITER[1]:
        loss = sr_loss
    if cfg.SOLVER.SEG_PRETRAIN_ITER[0] <= iteration < cfg.SOLVER.SEG_PRETRAIN_ITER[1]:
        loss = segment_loss

    return loss

def calc_loss4DSRL(segment_loss, logging_segment_loss, sr_loss, logging_sr_loss, fa_loss, logging_fa_loss, iteration, cfg, args):
    segment_loss = segment_loss.mean()
    logging_segment_loss += segment_loss.item()
    sr_loss = sr_loss.mean()
    logging_sr_loss += sr_loss.item()
    fa_loss = fa_loss.mean()
    logging_fa_loss += fa_loss.item()

    # This weight follows the official code implementation (although it is w_fa = 1, w_sr = 0.1, w_ss=1 in the original paper).
    loss = cfg.SOLVER.DSRL_SR_WEIGHT * sr_loss + cfg.SOLVER.DSRL_FA_WEIGHT * fa_loss + cfg.SOLVER.DSRL_SEG_WEIGHT * segment_loss
    loss = calc_pretrain_loss(loss, segment_loss, sr_loss, iteration, cfg)

    return loss, logging_segment_loss, logging_sr_loss, logging_fa_loss


def increase_w_task(cfg, args, iteration):
    w_task = (1- 0) / (cfg.SOLVER.INCRESE_TASK_W_ITER[1] - cfg.SOLVER.INCRESE_TASK_W_ITER[0]) * (iteration - cfg.SOLVER.INCRESE_TASK_W_ITER[0])
    if w_task > 1:
        w_task = 1
    if args.wandb_flag:
        wandb.log({'Iteration': iteration,
                "w_task":w_task,
                })

    return w_task

    

def fix_1st_stage_model_params(cfg, args, model, free_1st_stage_model_params, iteration):
    if not cfg.MODEL.JOINT_LEARNING and free_1st_stage_model_params and cfg.MODEL.SR != "bicubic" and cfg.MODEL.SCALE_FACTOR != 1:
        if not cfg.MODEL.SR_SEG_INV and iteration >= cfg.SOLVER.SR_PRETRAIN_ITER[1]:
            print('+++++++ Fix parameters of SR model(1st stage model). +++++++') 
            print('+++++++ Update parameters of segmentation model(2nd stage model). +++++++') 
            if args.num_gpus > 1:
                for param in model.module.sr_model.parameters():
                    param.requires_grad = False
                for param in model.module.segmentation_model.parameters():
                    param.requires_grad = True
                free_1st_stage_model_params = False
            else:
                for param in model.sr_model.parameters():
                    param.requires_grad = False
                for param in model.segmentation_model.parameters():
                    param.requires_grad = True
                free_1st_stage_model_params = False                
            free_1st_stage_model_params = False

        # elif cfg.MODEL.SR_SEG_INV and iteration >= cfg.SOLVER.SEG_PRETRAIN_ITER[1]:
        #     print('+++++++ Fix parameters of segmentation model(1st stage model). +++++++') 
        #     print('+++++++ Update parameters of SR model(2nd stage model). +++++++') 
        #     for param in model.module.segmentation_model.parameters():
        #         param.requires_grad = False
        #     for param in model.module.sr_model.parameters():
        #         param.requires_grad = True

    pretrain_flag = cfg.SOLVER.SR_PRETRAIN_ITER[0] <= iteration < cfg.SOLVER.SR_PRETRAIN_ITER[1]
    # print(iteration)
    if "Boundary" in cfg.SOLVER.SEG_LOSS_FUNC and pretrain_flag:
        if args.num_gpus > 1:
            model.module.ss_loss_fn.fix_alpha, model.module.ss_loss_fn.iter = True, 1
        else:
            model.ss_loss_fn.fix_alpha, model.ss_loss_fn.iter = True, 1
    if "Boundary" in cfg.SOLVER.SEG_LOSS_FUNC and not pretrain_flag:
        if args.num_gpus > 1:
            model.module.ss_loss_fn.fix_alpha = False
            model.module.ss_loss_fn.update_alpha()
        else:
            model.ss_loss_fn.fix_alpha = False
            model.ss_loss_fn.update_alpha()

    return free_1st_stage_model_params

def fix_2nd_stage_model_params(cfg, args, model):
    if not cfg.MODEL.JOINT_LEARNING and cfg.MODEL.SR != "bicubic" and cfg.MODEL.SCALE_FACTOR != 1:
        if not cfg.MODEL.SR_SEG_INV:
            print('+++++++ Fix parameters of segmentation model(2nd stage model). +++++++') 
            print('+++++++ Update parameters of SR model(1st stage model). +++++++') 
            if args.num_gpus > 1:
                for param in model.module.segmentation_model.parameters():
                    param.requires_grad = False
            else:
                for param in model.segmentation_model.parameters():
                    param.requires_grad = False



        # else:
        #     print('+++++++ Fix parameters of SR model(2nd stage model). +++++++') 
        #     print('+++++++ Update parameters of segmentation model(1st stage model). +++++++') 
        #     for param in model.module.sr_model.parameters():
        #         param.requires_grad = False



