import os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import wandb
from model.utils.estimate_metrics import PSNR, SSIM, IoU
from model.utils.metrics.retinal_metrics import get_retinal_seg_metrics
from model.utils.metrics.surface_distance.metrics import surface_distance
from model.data.samplers.patch_sampler import JointPatch
from model.utils.save_output import *


def inference_for_ss(args, cfg, model, test_loader):
    """
    aiu_scoures : test_case(=len(test_loader)) x threshold_case(=99)
    """
    num_hd_outliner = 0
    num_msd_outliner = 0

    fnames = []
    max_iter = len(test_loader)
    assert max_iter != 0, "Dataset size is 0!!"

    img_psnr_scores = np.array([])
    kernel_psnr_scores = np.array([])
    ssim_scores = np.array([])

    psnr = PSNR()
    ssim = SSIM()
    iou = IoU()

    joint_patch = JointPatch()
    
    os.makedirs(os.path.dirname(os.path.join(args.output_dirname, "images")), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(args.output_dirname, "masks")), exist_ok=True)
    
    if args.test_aiu:
        thresholds = [i*0.01 for i in range(1, 100)]
        threshold_map = torch.Tensor(thresholds).view(len(thresholds), 1, 1).to("cuda")
        zero = torch.Tensor([0]).to("cuda")
        save_thresholds_idx = [0] + [9 + i * 10 for i in range(9)] + [98]
        iou_mode = "AIU"
        
    else:
        thresholds = [0.5]
        iou_mode = "IoU"

    if args.wandb_flag:
        # --- wandb setting https://docs.wandb.ai/integrations/pytorch --- #
        prj = args.wandb_prj_name
        if 'RetinalSeg' in cfg.DATASET.TEST_IMAGE_DIR:
            prj += '_Retinal'
        if args.test_surface_distance:
            prj += "_withDistMet"
        wandb.init(config=cfg, project= prj)
        
        wandb.config.update(args)
        wandb.run.name = cfg.OUTPUT_DIR.replace("output/", "")
        # Magic
        wandb.watch(model, log='all')

    print('===== Start Evaluation =====')
    # print(vars(test_loader))
    # print(f'Number of test dataset : {len(test_loader.dataset.fnames)}')
    # print(args.test_surface_distance)

    model.eval()
    for iteration, (imgs, sr_targets, masks, kernel_targets, fname, img_unfold_shape, seg_unfold_shape) in enumerate(test_loader, 1):
        fnames += list(fname)
        imgs = imgs.view(-1, *imgs.shape[2:])
        kernel_targets = kernel_targets.view(-1, 1, *kernel_targets.shape[2:])
        num_batch = len(sr_targets)
        num_patch = img_unfold_shape[0][2] * img_unfold_shape[0][3]
        damy_kernel = torch.zeros((num_batch * num_patch, 1, cfg.BLUR.KERNEL_SIZE, cfg.BLUR.KERNEL_SIZE))

        sr_preds, segment_preds, kernel_preds = model(imgs, damy_kernel, sr_targets=sr_targets)
        sr_preds = joint_patch(sr_preds, img_unfold_shape[0])
        segment_preds = joint_patch(segment_preds, seg_unfold_shape[0])

        # SR evaluation
        if not cfg.MODEL.SR_SEG_INV and cfg.MODEL.SCALE_FACTOR != 1:
            sr_preds[sr_preds>1] = 1 # clipping
            sr_preds[sr_preds<0] = 0 # clipping        
            img_psnr_scores = np.append(img_psnr_scores, psnr(sr_preds, sr_targets.to("cuda"))) 
            ssim_scores = np.append(ssim_scores, ssim(sr_preds, sr_targets.to("cuda"))) 
            kernel_preds[kernel_preds>1] = 1 # clipping
            kernel_preds[kernel_preds<0] = 0 # clipping   
            kernel_psnr_scores = np.append(kernel_psnr_scores, psnr(kernel_preds, kernel_targets.to("cuda"))) 
            if args.sf_save_image:
                save_img(args.output_dirname, sr_preds, fname)
                if cfg.MODEL.SR == 'KBPN':
                    save_kernel(args, kernel_preds, fname, num_batch)
        else:
            img_psnr_scores = np.append(img_psnr_scores, 0)
            ssim_scores = np.append(ssim_scores, 0)
            kernel_psnr_scores = np.append(kernel_psnr_scores, 0)
        
        # Segmentation evaluation
        segment_preds_bi = (segment_preds - threshold_map > zero).float()
        # print(segment_preds_bi.shape)
        for idx in save_thresholds_idx:
            if args.sf_save_image:
                save_mask(args, segment_preds_bi[:, idx], fname, thresholds[idx])
        
        if args.sf_save_image:
            save_mask(args, segment_preds, fname, -1)
        iou_scores = np.reshape(iou(segment_preds_bi, masks.to("cuda")), (len(masks), -1))
        if args.test_surface_distance:
            hd_scores, msd_scores, num_hd_outliner, num_msd_outliner = calc_distance_metrics(segment_preds_bi, masks, num_hd_outliner, num_msd_outliner)
        if args.test_classification_metrics:
            retinal_seg_metrics_scores = get_retinal_seg_metrics(segment_preds_bi[:, 49], masks)
            acc, sens, spec = retinal_seg_metrics_scores['acc'], retinal_seg_metrics_scores['sens'], retinal_seg_metrics_scores['spec']

        if 'aiu_scores' in locals():
            aiu_scores = np.append(aiu_scores, iou_scores, axis=0)
        else:
            aiu_scores = np.copy(iou_scores)
        
        if args.test_classification_metrics:
            if 'ahd_scores' in locals():
                acc_scores = np.append(acc_scores, acc, axis=0)
                sens_scores = np.append(sens_scores, sens, axis=0)
                spec_scores = np.append(spec_scores, spec, axis=0)
            else:
                acc_scores = np.copy(acc)
                sens_scores = np.copy(sens)
                spec_scores = np.copy(spec)
        if args.test_surface_distance:
            if 'ahd_scores' in locals():
                ahd_scores = np.append(ahd_scores, hd_scores, axis=0)
                amsd_scores = np.append(amsd_scores, msd_scores, axis=0)
            else:
                ahd_scores = np.copy(hd_scores)
                amsd_scores = np.copy(msd_scores)
        
        if args.wandb_flag:
        # wandb
            wandb_log = {"PSNR_score": img_psnr_scores[-1], 
                        "SSIM_score":ssim_scores[-1],
                        "PSNR(Kernel)_score": kernel_psnr_scores[-1], 
                        f"{iou_mode}_scores": np.mean(iou_scores), 
                        }
            if args.test_surface_distance:
                wandb_log.update({"HD95_scores": np.mean(hd_scores), 
                                "MSD_scores":np.mean(msd_scores), 
                                })

            wandb.log(wandb_log)

        del iou_scores
        if iteration % 10 == 0:
            print(f"estimation {iteration/max_iter*100:.4f} % finish!")
            if args.test_surface_distance:
                print(f"PSNR_mean:{np.mean(img_psnr_scores):.4f}  SSIM_mean:{np.mean(ssim_scores):.4f} PSNR(Kernel)_mean:{np.mean(kernel_psnr_scores):.4f} {iou_mode}_mean:{np.mean(aiu_scores):.4f} HD95_mean:{np.mean(ahd_scores):.4f} MSD_mean:{np.mean(amsd_scores):.4f}")
            else:
                print(f"PSNR_mean:{np.mean(img_psnr_scores):.4f}  SSIM_mean:{np.mean(ssim_scores):.4f} PSNR(Kernel)_mean:{np.mean(kernel_psnr_scores):.4f} {iou_mode}_mean:{np.mean(aiu_scores):.4f}")

    print(f"estimation finish!!")
    print(f"PSNR_mean:{np.mean(img_psnr_scores):.4f}  SSIM_mean:{np.mean(ssim_scores):.4f} PSNR(Kernel)_mean:{np.mean(kernel_psnr_scores):.4f} {iou_mode}_mean:{np.mean(aiu_scores):.4f}")
    if args.test_surface_distance:
        print(f"HD95_mean:{np.mean(ahd_scores):.4f} MSD_mean:{np.mean(amsd_scores):.4f}")
        print(f'num_hd_outliner:{num_hd_outliner} ,  num_msd_outliner:{num_msd_outliner}')
    if args.test_classification_metrics:
        print(f"Accuracy (th=0.50):{np.mean(acc_scores):.4f} Sensitivity (th=0.50):{np.mean(sens_scores):.4f} Specificity (th=0.50):{np.mean(spec_scores):.4f}")

    if args.wandb_flag:
        wandb.log({"PSNR_score_mean": np.mean(img_psnr_scores), 
            "SSIM_score_mean":np.mean(ssim_scores), 
            "PSNR(Kernel)_score_mean": np.mean(kernel_psnr_scores), 
            f"{iou_mode}_scores_mean": np.mean(aiu_scores),
            })
       
        if args.test_surface_distance:
            wandb.log({"HD95_score_mean": np.mean(ahd_scores), 
                "MSD_score_mean":np.mean(amsd_scores), 
                "HD95_score_median": np.median(ahd_scores), 
                "MSD_score_median": np.median(amsd_scores),
                })

        if args.test_classification_metrics:
            wandb.log({"Accuracy (th=0.50)": np.mean(acc_scores), 
                "Sensitivity (th=0.50)": np.mean(sens_scores), 
                "Specificity (th=0.50)": np.mean(spec_scores), 
                })

        if args.test_aiu:
            plot_metrics_th(aiu_scores, thresholds, "IoU")

        if args.test_surface_distance:
            plot_metrics_th(ahd_scores, thresholds, "HD95")
            plot_metrics_th(amsd_scores, thresholds, "MSD")
            plot_metrics_th(ahd_scores, thresholds, "HD95", med=True)
            plot_metrics_th(amsd_scores, thresholds, "MSD", med=True)

    save_iou_log(aiu_scores, thresholds, fnames, args.output_dirname) # Output IoU scores as csv file.


def inference_tti_building(args, cfg, model, test_loader):
    """
    aiu_scoures : test_case(=len(test_loader)) x threshold_case(=99)
    """
    fnames = []
    max_iter = len(test_loader)

    thresholds = [i*0.01 for i in range(1, 100)]
    threshold_map = torch.Tensor(thresholds).view(len(thresholds), 1, 1).to("cuda")
    zero = torch.Tensor([0]).to("cuda")
    save_thresholds_idx = [0] + [9 + i * 10 for i in range(9)] + [98]

    joint_patch = JointPatch()
    
    os.makedirs(os.path.dirname(os.path.join(args.output_dirname, "images")), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.join(args.output_dirname, "masks")), exist_ok=True)
    
    print('Evaluation Starts')
    print(f'Number of test dataset : {len(test_loader) * args.batch_size}')

    model.eval()
    for iteration, (_imgs, fname, img_unfold_shape, seg_unfold_shape) in enumerate(test_loader, 1):
        _imgs = _imgs.view(-1, *_imgs.shape[2:])
        fnames += list(fname)
        num_batch = len(_imgs)
        num_roop = round(len(_imgs) / 6)
        _imgs = torch.tensor_split(_imgs, num_roop, dim=0)
        for imgs in _imgs:
            num_patch = img_unfold_shape[0][2] * img_unfold_shape[0][3]
            damy_kernel = torch.zeros((num_batch * num_patch, 1, cfg.BLUR.KERNEL_SIZE, cfg.BLUR.KERNEL_SIZE))
            sr_pred, segment_pred, kernel_pred = model(imgs, damy_kernel)
            if 'sr_preds' in locals():
                sr_preds = torch.cat((sr_preds, sr_pred), 0)
                segment_preds = torch.cat((segment_preds, segment_pred), 0)
                kernel_preds = torch.cat((kernel_preds, kernel_pred), 0)
            else:
                sr_preds = sr_pred.clone()
                segment_preds = segment_pred.clone()
                kernel_preds = kernel_pred.clone()

        sr_preds = joint_patch(sr_preds, img_unfold_shape[0], batch_size=len(img_unfold_shape))
        segment_preds = joint_patch(segment_preds, seg_unfold_shape[0], batch_size=len(seg_unfold_shape))

        # SR evaluation
        if not cfg.MODEL.SR_SEG_INV and cfg.MODEL.SCALE_FACTOR != 1:
            sr_preds[sr_preds>1] = 1 # clipping
            sr_preds[sr_preds<0] = 0 # clipping
            if args.sf_save_image:
                save_img(args.output_dirname, sr_preds, fname)
            if cfg.MODEL.SR == 'KBPN':
                save_kernel(args, kernel_preds, fname, len(fname))
        
        # Segmentation evaluation
        segment_preds_bi = (segment_preds - threshold_map > zero).float()
        for idx in save_thresholds_idx:
            if args.sf_save_image:
                save_mask(args, segment_preds_bi[:, idx], fname, thresholds[idx])

        if iteration % 10 == 0:
            print(f"estimation {iteration/max_iter*100:.4f} % finish!")

        del sr_preds, segment_preds, kernel_preds

    print(f"estimation finish!!")

def plot_metrics_th(metrics_scores, thresholds, metrics, med=False):
    if med:
        metrics_scores = np.median(metrics_scores, axis=0)
        metrics += '_median'
    else:
        metrics_scores = np.mean(metrics_scores, axis=0)
    # print(metrics_scores)
    for iou, th in zip(metrics_scores, thresholds):
        wandb.log({f"{metrics}(thresholds)": iou, 
                "thresholds":th, 
                })

def save_iou_log(aiu_scores, thresholds, fnames, output_dir):
    df = pd.DataFrame(aiu_scores, columns=thresholds, index=fnames)
    df.to_csv(os.path.join(output_dir, 'iou_log.csv'))
    print('IoU log saved!!')
    print(df)

def calc_distance_metrics(preds, gts, num_hd_outliner, num_msd_outliner):
    # Variable for unit conversion from pixel to mm

    vertical = 1
    horizontal = 1
    batch_size, num_th = preds.shape[:2]
    max_img_len = np.max(preds.shape[3:])
    hd_scores = np.zeros((batch_size, num_th))
    msd_scores = np.zeros((batch_size, num_th))
    percentile = 50

    for i in range(batch_size):
        gt = torch.squeeze(gts[i, 0]).data.cpu().numpy().astype(bool)
        for j in range(num_th):
            pred = torch.squeeze(preds[i, j]).data.cpu().numpy().astype(bool)
            surface_distances = surface_distance.compute_surface_distances(gt, pred, spacing_mm=(vertical, horizontal))

            distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
            distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
            surfel_areas_gt = surface_distances["surfel_areas_gt"]
            surfel_areas_pred = surface_distances["surfel_areas_pred"]

            if len(distances_gt_to_pred) == 0 and len(distances_pred_to_gt) == 0:    
                hd_scores[i, j] = 0
            elif len(distances_gt_to_pred) == 0 or len(distances_pred_to_gt) == 0:
                hd_scores[i, j] = max_img_len
                num_hd_outliner += 1
                if num_hd_outliner % 100 == 0:
                    print(f'hd score is outlire (num:{num_hd_outliner})')
            else:
                hd_scores[i, j] = surface_distance.compute_robust_hausdorff(surface_distances, percentile)

            if np.sum(surfel_areas_gt) == 0 and np.sum(surfel_areas_pred) == 0:    
                msd_scores[i, j] = 0
            elif np.sum(surfel_areas_gt) == 0 or np.sum(surfel_areas_pred) == 0:
                msd_scores[i, j] = max_img_len
                num_msd_outliner += 1
                if num_msd_outliner % 100 == 0:
                    print(f'msd score is outlire (num:{num_msd_outliner})')
            else:
                avg_dist_gt_to_pred, avg_dist_pred_to_gt = surface_distance.compute_average_surface_distance(surface_distances)
                msd_scores[i, j] = (avg_dist_gt_to_pred + avg_dist_pred_to_gt) / 2

    return np.reshape(hd_scores, (batch_size, -1)), np.reshape(msd_scores, (batch_size, -1)), num_hd_outliner, num_msd_outliner




