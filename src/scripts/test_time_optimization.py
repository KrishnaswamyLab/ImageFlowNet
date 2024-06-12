import argparse
import ast
import os
import sys
import cv2
from typing import Tuple

import copy
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import monai
from matplotlib import pyplot as plt

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir)

from data_utils.prepare_dataset import prepare_dataset_all_subarrays
from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log
from utils.seed import seed_everything
from utils.parse import parse_settings
from utils.metrics import psnr, ssim, dice_coeff, hausdorff

# NOTE: The following imported models are actually used!
from nn.unet_ode_position_parametrized import PPODEUNet
from nn.unet_sde_position_parametrized import PPSDEUNet


def convert_variables(images: torch.Tensor,
                      timestamps: torch.Tensor,
                      device: torch.device) -> Tuple[torch.Tensor]:
    '''
    Some repetitive processing of variables.
    '''
    x_list = [images[:, i, ...].float().to(device) for i in range(images.shape[1])]
    t_arr = timestamps[0].float().to(device)
    return x_list, t_arr


def numpy_variables(*tensors: torch.Tensor) -> Tuple[np.array]:
    '''
    Some repetitive numpy casting of variables.
    '''
    return [_tensor.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0) for _tensor in tensors]

def cast_to_0to1(*np_arrays: np.array) -> Tuple[np.array]:
    '''
    Cast image to normal dynamic range between 0 and 1.
    '''
    return [np.clip((_arr + 1) / 2, 0, 1) for _arr in np_arrays]

def gray_to_rgb(*tensors: torch.Tensor) -> Tuple[np.array]:
    rgb_list = []
    for item in tensors:
        assert len(item.shape) in [2, 3]
        if len(item.shape) == 3:
            assert item.shape[-1] == 1
            rgb_list.append(np.repeat(item, 3, axis=-1))
        else:
            rgb_list.append(np.repeat(item[..., None], 3, axis=-1))

    return rgb_list



def plot_contour(image, label):
    true_contours, _hierarchy = cv2.findContours(np.uint8(label),
                                                 cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_NONE)
    for contour in true_contours:
        cv2.drawContours(image, contour, -1, (0.0, 1.0, 0.0), 2)

def plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, xT_pred, save_path: str,
                      x0_true_seg=None, xT_pred_seg=None, xT_true_seg=None) -> None:
    fig_sbs = plt.figure(figsize=(24, 10))
    plt.rcParams['font.family'] = 'serif'

    aspect_ratio = x0_true.shape[0] / x0_true.shape[1]

    assert len(x0_true.shape) in [2, 3]
    if len(x0_true.shape) == 2 or x0_true.shape[-1] == 1:
        x0_true, xT_true, x0_recon, xT_recon, xT_pred = \
            gray_to_rgb(x0_true, xT_true, x0_recon, xT_recon, xT_pred)

    # First column: Ground Truth.
    ax = fig_sbs.add_subplot(2, 6, 1)
    ax.imshow(x0_true)
    ax.set_title('GT(t=0), time: %s\n[vs GT(t=T)]: PSNR=%.2f, SSIM=%.3f' % (
        t_list[0].item(), psnr(x0_true, xT_true), ssim(x0_true, xT_true)))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 7)
    ax.imshow(xT_true)
    ax.set_title('GT(t=T), time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Second column: Reconstruction.
    ax = fig_sbs.add_subplot(2, 6, 2)
    ax.imshow(x0_recon)
    ax.set_title('Recon(t=0), time: %s' % t_list[0].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 8)
    ax.imshow(xT_recon)
    ax.set_title('Recon(t=T), time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Third column: Prediction.
    ax = fig_sbs.add_subplot(2, 6, 3)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 9)
    ax.imshow(xT_pred)
    ax.set_title('Pred(t=T), time: %s -> time: %s\n[vs GT(t=T)]: PSNR=%.2f, SSIM=%.3f' % (
        t_list[0].item(), t_list[1].item(), psnr(xT_pred, xT_true), ssim(xT_pred, xT_true)))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Fourth column: |Ground Truth t1 - Ground Truth t2|, |Ground Truth - Prediction|.
    ax = fig_sbs.add_subplot(2, 6, 4)
    ax.imshow(np.abs(x0_true - xT_true))
    ax.set_title('|GT(t=0) - GT(t=T)|, time: %s and %s\n[MAE=%.4f, MSE=%.4f]' % (
        t_list[0].item(), t_list[1].item(), np.mean(np.abs(x0_true - xT_true)), np.mean((x0_true - xT_true)**2)))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 10)
    ax.imshow(np.abs(xT_pred - xT_true))
    ax.set_title('|Pred(t=T) - GT(t=T)|, time: %s\n[MAE=%.4f, MSE=%.4f]' % (
        t_list[1].item(), np.mean(np.abs(xT_pred - xT_true)), np.mean((xT_pred - xT_true)**2)))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Fifth column: Ground Truth with segmentation.
    ax = fig_sbs.add_subplot(2, 6, 5)
    if x0_true_seg is not None:
        plot_contour(x0_true, x0_true_seg)
        ax.imshow(x0_true)
        ax.set_title('GT(t=0), time: %s\n[vs GT(t=T)]: DSC=%.3f, HD=%.2f' % (
            t_list[0].item(), dice_coeff(x0_true_seg, xT_true_seg), hausdorff(x0_true_seg, xT_true_seg)))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 11)
    if xT_true_seg is not None:
        plot_contour(xT_true, xT_true_seg)
        ax.imshow(xT_true)
        ax.set_title('GT(t=T), time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Sixth column: Prediction with segmentation.
    ax = fig_sbs.add_subplot(2, 6, 6)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 12)
    if xT_pred_seg is not None:
        plot_contour(xT_pred, xT_pred_seg)
        ax.imshow(xT_pred)
        ax.set_title('Pred(t=T), time: %s -> time: %s\n[vs GT(t=T)]: DSC=%.3f, HD=%.2f' % (
            t_list[0].item(), t_list[1].item(), dice_coeff(xT_pred_seg, xT_true_seg), hausdorff(xT_pred_seg, xT_true_seg)))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    fig_sbs.tight_layout()
    fig_sbs.savefig(save_path)
    plt.close(fig=fig_sbs)
    return


def main(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    train_set, val_set, test_set, num_image_channel, max_t = \
        prepare_dataset_all_subarrays(config=config)

    try:
        model = globals()[config.model](device=device,
                                        num_filters=config.num_filters,
                                        depth=config.depth,
                                        ode_location=config.ode_location,
                                        in_channels=num_image_channel,
                                        out_channels=num_image_channel,
                                        contrastive=config.coeff_contrastive + config.coeff_invariance > 0)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    model.to(device)

    if os.path.isfile(config.segmentor_ckpt):
        segmentor = torch.nn.Sequential(
            monai.networks.nets.DynUNet(
                spatial_dims=2,
                in_channels=1,
                out_channels=1,
                kernel_size=[5, 5, 5, 5],
                filters=[16, 32, 64, 128],
                strides=[1, 1, 1, 1],
                upsample_kernel_size=[1, 1, 1, 1]),
            torch.nn.Sigmoid()).to(device)
        segmentor.load_state_dict(torch.load(config.segmentor_ckpt, map_location=device))
        segmentor.eval()
    else:
        print('Using an identity mapping as a placeholder for segmentor.')
        segmentor = torch.nn.Identity()

    # Only relevant to ODE
    config.t_multiplier = config.ode_max_t / max_t

    assert len(test_set) == len(test_set.dataset)

    setting_str = config.save_folder.split('/')[-3]
    for best_type in ['pred_psnr', 'seg_dice']:
        if best_type == 'pred_psnr':

            model.load_weights(config.model_save_path.replace('.pty', '_best_pred_psnr.pty'), device=device)
            log('%s: Model weights successfully loaded.' % config.model,
                to_console=True)
            save_path_fig_dir = '../../visualize/test_time_optimization/opt-iter_%d_opt-lr_%s_%s/best_pred_psnr/' % (
                config.opt_iters, config.opt_lr, setting_str)
            log_dir = save_path_fig_dir + 'log.txt'

        elif best_type == 'seg_dice':

            model.load_weights(config.model_save_path.replace('.pty', '_best_seg_dice.pty'), device=device)
            log('%s: Model weights successfully loaded.' % config.model,
                to_console=True)
            save_path_fig_dir = '../../visualize/test_time_optimization/opt-iter_%d_opt-lr_%s_%s/best_seg_dice/' % (
                config.opt_iters, config.opt_lr, setting_str)
            log_dir = save_path_fig_dir + 'log.txt'

        os.makedirs(save_path_fig_dir, exist_ok=True)

        mse_loss = torch.nn.MSELoss()

        test_loss, test_recon_psnr, test_recon_ssim, test_pred_psnr, test_pred_ssim = 0, 0, 0, [], []
        test_seg_dice, test_seg_hd, test_residual_mae, test_residual_mse = [], [], [], []
        test_seg_dice_gt = []

        # Count the test samples.
        num_test_samples = 0
        for iter_idx, (images, timestamps) in enumerate(tqdm(test_set)):

            if iter_idx > config.max_testing_samples:
                break
            if images.shape[1] < 3:
                # Cannot perform test-time optimization.
                continue
            num_test_samples += 1

            assert images.shape[1] >= 3
            assert timestamps.shape[1] >= 3

            x_list, t_arr = convert_variables(images, timestamps, device)
            x_start, x_end = x_list[0], x_list[-1]
            # For compatibility
            t_list = [t_arr[0], t_arr[-1]]

            if config.opt_iters > 0:
                # Perform test-time optimization.

                # Re-load the model weights.
                if best_type == 'pred_psnr':
                    model.load_weights(config.model_save_path.replace('.pty', '_best_pred_psnr.pty'), device=device)
                elif best_type == 'seg_dice':
                    model.load_weights(config.model_save_path.replace('.pty', '_best_seg_dice.pty'), device=device)
                model.train()

                opt_optimizer = torch.optim.AdamW(model.parameters(), lr=config.opt_lr)

                opt_iter = 0
                while opt_iter < config.opt_iters:

                    # NOTE: Should not cheat using ground truth. Hence `len(x_list) - 1`.
                    idx_start, idx_end = sorted(np.random.choice(len(x_list) - 1, size=2, replace=False))
                    __x_start = x_list[idx_start].to(device)
                    __x_end = x_list[idx_end].to(device)
                    __t_start = t_arr[idx_start].to(device)
                    __t_end = t_arr[idx_end].to(device)

                    try:
                        model.freeze_time_independent()
                    except AttributeError:
                        print('`model.freeze_time_independent()` ignored.')
                    __x_end_pred = model(x=__x_start, t=(__t_end - __t_start).unsqueeze(0) * config.t_multiplier)
                    __loss_pred = mse_loss(__x_end, __x_end_pred)

                    opt_optimizer.zero_grad()
                    __loss_pred.backward()
                    opt_optimizer.step()
                    opt_iter += 1

            model.eval()
            x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
            x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))
            x_end_pred = model(x=x_start, t=(t_arr[-1] - t_arr[0]).unsqueeze(0) * config.t_multiplier)

            loss_recon = mse_loss(x_start, x_start_recon) + mse_loss(x_end, x_end_recon)
            loss_pred = mse_loss(x_end, x_end_pred)

            loss = loss_recon + loss_pred
            test_loss += loss.item()

            x_start_seg = segmentor(x_start) > 0.5
            x_end_seg = segmentor(x_end) > 0.5
            x_end_pred_seg = segmentor(x_end_pred) > 0.5

            x0_true, x0_recon, xT_true, xT_recon, xT_pred, x0_seg, xT_seg, xT_pred_seg = \
                numpy_variables(x_start, x_start_recon, x_end, x_end_recon, x_end_pred,
                                x_start_seg, x_end_seg, x_end_pred_seg)

            # NOTE: Convert to image with normal dynamic range.
            x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
                cast_to_0to1(x0_true, x0_recon, xT_true, xT_recon, xT_pred)

            test_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
            test_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
            test_pred_psnr.append(psnr(xT_true, xT_pred))
            test_pred_ssim.append(ssim(xT_true, xT_pred))

            test_seg_dice.append(dice_coeff(xT_pred_seg, xT_seg))
            test_seg_hd.append(hausdorff(xT_pred_seg, xT_seg))
            test_residual_mae.append(np.mean(np.abs(xT_pred - xT_true)))
            test_residual_mse.append(np.mean((xT_pred - xT_true)**2))
            test_seg_dice_gt.append(dice_coeff(x0_seg, xT_seg))

            # Plot the side-by-side figures.
            save_path_fig_sbs = save_path_fig_dir + 'figure_%s.png' % str(iter_idx + 1).zfill(5)
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, xT_pred, save_path_fig_sbs,
                              x0_true_seg=x0_seg, xT_pred_seg=xT_pred_seg, xT_true_seg=xT_seg)

        test_loss = test_loss / num_test_samples
        test_recon_psnr = test_recon_psnr / num_test_samples
        test_recon_ssim = test_recon_ssim / num_test_samples

        test_pred_psnr = np.array(test_pred_psnr)
        test_pred_ssim = np.array(test_pred_ssim)
        test_seg_dice = np.array(test_seg_dice)
        test_seg_hd = np.array(test_seg_hd)
        test_residual_mae = np.array(test_residual_mae)
        test_residual_mse = np.array(test_residual_mse)
        test_seg_dice_gt = np.array(test_seg_dice_gt)

        growth_dice_thr = 0.9

        test_pred_psnr_minor_growth = test_pred_psnr[test_seg_dice_gt > growth_dice_thr]
        test_pred_ssim_minor_growth = test_pred_ssim[test_seg_dice_gt > growth_dice_thr]
        test_seg_dice_minor_growth = test_seg_dice[test_seg_dice_gt > growth_dice_thr]
        test_seg_hd_minor_growth = test_seg_hd[test_seg_dice_gt > growth_dice_thr]
        test_residual_mae_minor_growth = test_residual_mae[test_seg_dice_gt > growth_dice_thr]
        test_residual_mse_minor_growth = test_residual_mse[test_seg_dice_gt > growth_dice_thr]

        test_pred_psnr_major_growth = test_pred_psnr[test_seg_dice_gt <= growth_dice_thr]
        test_pred_ssim_major_growth = test_pred_ssim[test_seg_dice_gt <= growth_dice_thr]
        test_seg_dice_major_growth = test_seg_dice[test_seg_dice_gt <= growth_dice_thr]
        test_seg_hd_major_growth = test_seg_hd[test_seg_dice_gt <= growth_dice_thr]
        test_residual_mae_major_growth = test_residual_mae[test_seg_dice_gt <= growth_dice_thr]
        test_residual_mse_major_growth = test_residual_mse[test_seg_dice_gt <= growth_dice_thr]

        log('[Best %s] Test loss: %.3f, PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f \u00B1 %.3f, SSIM (pred): %.3f \u00B1 %.3f'
            % (best_type, test_loss, test_recon_psnr, test_recon_ssim,
               np.mean(test_pred_psnr), np.std(test_pred_psnr) / np.sqrt(len(test_pred_psnr)),
               np.mean(test_pred_ssim), np.std(test_pred_ssim) / np.sqrt(len(test_pred_ssim))) + \
            ' MAE (pred): %.3f \u00B1 %.3f, MSE (pred): %.3f \u00B1 %.3f, DSC (pred): %.3f \u00B1 %.3f, HD (pred): %.3f \u00B1 %.3f'
            % (np.mean(test_residual_mae), np.std(test_residual_mae) / np.sqrt(len(test_residual_mae)),
               np.mean(test_residual_mse), np.std(test_residual_mse) / np.sqrt(len(test_residual_mse)),
               np.mean(test_seg_dice), np.std(test_seg_dice) / np.sqrt(len(test_seg_dice)),
               np.mean(test_seg_hd), np.std(test_seg_hd) / np.sqrt(len(test_seg_hd))),
            filepath=log_dir ,
            to_console=True)

        log('Minor growth (GT dice > %s) PSNR (pred): %.3f \u00B1 %.3f, SSIM (pred): %.3f \u00B1 %.3f'
            % (growth_dice_thr,
               np.mean(test_pred_psnr_minor_growth), np.std(test_pred_psnr_minor_growth) / np.sqrt(len(test_pred_psnr_minor_growth)),
               np.mean(test_pred_ssim_minor_growth), np.std(test_pred_ssim_minor_growth) / np.sqrt(len(test_pred_ssim_minor_growth))) + \
            ' MAE (pred): %.3f \u00B1 %.3f, MSE (pred): %.3f \u00B1 %.3f, DSC (pred): %.3f \u00B1 %.3f, HD (pred): %.3f \u00B1 %.3f'
            % (np.mean(test_residual_mae_minor_growth), np.std(test_residual_mae_minor_growth) / np.sqrt(len(test_residual_mae_minor_growth)),
               np.mean(test_residual_mse_minor_growth), np.std(test_residual_mse_minor_growth) / np.sqrt(len(test_residual_mse_minor_growth)),
               np.mean(test_seg_dice_minor_growth), np.std(test_seg_dice_minor_growth) / np.sqrt(len(test_seg_dice_minor_growth)),
               np.mean(test_seg_hd_minor_growth), np.std(test_seg_hd_minor_growth) / np.sqrt(len(test_seg_hd_minor_growth))),
            filepath=log_dir,
            to_console=True)

        log('Major growth (GT dice <= %s) PSNR (pred): %.3f \u00B1 %.3f, SSIM (pred): %.3f \u00B1 %.3f'
            % (growth_dice_thr,
               np.mean(test_pred_psnr_major_growth), np.std(test_pred_psnr_major_growth) / np.sqrt(len(test_pred_psnr_major_growth)),
               np.mean(test_pred_ssim_major_growth), np.std(test_pred_ssim_major_growth) / np.sqrt(len(test_pred_ssim_major_growth))) + \
            ' MAE (pred): %.3f \u00B1 %.3f, MSE (pred): %.3f \u00B1 %.3f, DSC (pred): %.3f \u00B1 %.3f, HD (pred): %.3f \u00B1 %.3f'
            % (np.mean(test_residual_mae_major_growth), np.std(test_residual_mae_major_growth) / np.sqrt(len(test_residual_mae_major_growth)),
               np.mean(test_residual_mse_major_growth), np.std(test_residual_mse_major_growth) / np.sqrt(len(test_residual_mse_major_growth)),
               np.mean(test_seg_dice_major_growth), np.std(test_seg_dice_major_growth) / np.sqrt(len(test_seg_dice_major_growth)),
               np.mean(test_seg_hd_major_growth), np.std(test_seg_hd_major_growth) / np.sqrt(len(test_seg_hd_major_growth))),
            filepath=log_dir,
            to_console=True)

        # Save to csv.
        results_df = pd.DataFrame({
            'DICE(xT_true_seg, x0_true_seg)': test_seg_dice_gt,
            'PSNR(xT_true, xT_pred)': test_pred_psnr,
            'SSIM(xT_true, xT_pred)': test_pred_ssim,
            'DICE(xT_true_seg, xT_pred_seg)': test_seg_dice,
            'HD(xT_true_seg, xT_pred_seg)': test_seg_hd,
            'MAE(xT_true, xT_pred)': test_residual_mae,
            'MSE(xT_true, xT_pred)': test_residual_mse,
        })
        results_df.to_csv(log_dir.replace('log.txt', best_type + '.csv'))

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--run-count', default=1, type=int)

    parser.add_argument('--dataset-name', default='retina_ucsf', type=str)
    parser.add_argument('--target-dim', default='(256, 256)', type=ast.literal_eval)
    parser.add_argument('--image-folder', default='UCSF_images_final_512x512', type=str)
    parser.add_argument('--output-save-folder', default='$ROOT/results/', type=str)
    parser.add_argument('--dataset-path', default='$ROOT/data/retina_ucsf/', type=str)
    parser.add_argument('--segmentor-ckpt', default='$ROOT/checkpoints/segment_retinaUCSF_seed1.pty', type=str)

    parser.add_argument('--model', default='StaticODEUNet', type=str)
    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=1e-4, type=float)
    parser.add_argument('--max-epochs', default=120, type=int)
    parser.add_argument('--batch-size', default=64, type=int)
    parser.add_argument('--ode-max-t', default=1.0, type=float)     # only relevant to ODE. Bigger is slower.
    parser.add_argument('--ode-location', default='all_connections', type=str)  # only relevant to ODE
    parser.add_argument('--depth', default=5, type=int)             # only relevant to simple unet
    parser.add_argument('--num-filters', default=64, type=int)      # only relevant to simple unet
    parser.add_argument('--diffusion-interval', default=100, type=int)      # only relevant to I2SB
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--max-testing-samples', default=500, type=int)
    parser.add_argument('--n-plot-per-epoch', default=4, type=int)

    parser.add_argument('--no-l2', action='store_true')  # only relevant to ODE
    parser.add_argument('--coeff-smoothness', default=0, type=float)  # only relevant to ODE
    parser.add_argument('--coeff-latent', default=0, type=float)
    parser.add_argument('--coeff-contrastive', default=0, type=float)
    parser.add_argument('--coeff-invariance', default=0, type=float)
    parser.add_argument('--pretrained-vision-model', default='convnext_tiny', type=str)

    # Test-time optimization
    parser.add_argument('--opt-iters', default=100, type=int)
    parser.add_argument('--opt-lr', default=1e-4, type=float)

    args = vars(parser.parse_args())
    config = AttributeHashmap(args)
    config = parse_settings(config, log_settings=False, run_count=config.run_count)

    seed_everything(config.random_seed)

    main(config=config)
