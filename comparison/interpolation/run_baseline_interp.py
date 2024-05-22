# Baseline comparison
# Image interpolation/extrapolation methods.

import argparse
import ast
import os
import sys
import cv2
from typing import Tuple, Literal

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import monai

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/src/')
# from data_utils.prepare_dataset import prepare_dataset_npt
from data_utils.prepare_dataset import prepare_dataset_all_subarrays
from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log
from utils.metrics import psnr, ssim, dice_coeff, hausdorff
from utils.seed import seed_everything


def parse_settings(config: AttributeHashmap):
    # fix typing issues
    for key in ['learning_rate', 'ode_tol']:
        if key in config.keys():
            config[key] = float(config[key])

    # fix path issues
    ROOT = '/'.join(
        os.path.dirname(os.path.abspath(__file__)).split('/')[:-2])
    for key in config.keys():
        if type(config[key]) == str and '$ROOT' in config[key]:
            config[key] = config[key].replace('$ROOT', ROOT)

    setting_str = '%s_baseline_%s_seed_%s' % (
        config.dataset_name,
        config.method,
        config.random_seed,
    )
    config.save_folder = '%s/%s' % (config.output_save_folder, setting_str)

    # Initialize log file.
    config.log_dir = config.save_folder + '/log.txt'

    return config


class Interpolator(object):
    def __init__(self, method: Literal['linear', 'cubic_spline']):
        self.method = method

    def _linear_interp(self, image_arr: torch.Tensor, time_arr: torch.Tensor, target_t: float) -> torch.Tensor:
        if len(image_arr) == 1:
            interpolated = image_arr
        else:
            x1, x2 = image_arr[-2], image_arr[-1]
            t1, t2 = time_arr[-2], time_arr[-1]
            target_t.to(t1.device)
            interpolated = x1 + (x2 - x1) * (target_t - t1) / (t2 - t1)
            interpolated = interpolated.unsqueeze(0)

        return interpolated

    def _cubic_spline_interp(self, image_arr: torch.Tensor, time_arr: torch.Tensor, target_t: float) -> torch.Tensor:
        if len(image_arr) == 1:
            interpolated = image_arr
        else:
            from scipy.interpolate import CubicSpline
            image_arr_flatten = image_arr.reshape(image_arr.shape[0], -1).cpu()
            target_t = target_t.cpu()
            t_arr = time_arr.cpu()

            scipy_interpolator = CubicSpline(t_arr, image_arr_flatten)
            interpolated = scipy_interpolator(target_t).reshape((1, *image_arr.shape[1:]))
            interpolated = torch.from_numpy(interpolated).float().to(image_arr.device)

        return interpolated

    def interp(self, image_arr: torch.Tensor, time_arr: torch.Tensor, target_t: float) -> torch.Tensor:
        if self.method == 'linear':
            interpolated = self._linear_interp(image_arr, time_arr, target_t)
        elif self.method == 'cubic_spline':
            interpolated = self._cubic_spline_interp(image_arr, time_arr, target_t)
        else:
            raise ValueError('Unsupported interpolation method: %s' % self.method)

        return interpolated


@torch.no_grad()
def test(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    # train_set, val_set, test_set, num_image_channel, max_t = \
    #     prepare_dataset_npt(config=config)
    train_set, val_set, test_set, num_image_channel, max_t = \
        prepare_dataset_all_subarrays(config=config)

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

    interpolator = Interpolator(method=config.method)

    log('Using interpolator: %s as baseline.' % config.method, to_console=True)

    assert len(test_set) == len(test_set.dataset)
    num_test_samples = min(config.max_testing_samples, len(test_set))

    save_path_fig_summary = '%s/results_interp/summary.png' % config.save_folder
    os.makedirs(os.path.dirname(save_path_fig_summary), exist_ok=True)

    deltaT_list, psnr_list, ssim_list = [], [], []
    test_loss, test_recon_psnr, test_recon_ssim, test_pred_psnr, test_pred_ssim = 0, 0, 0, [], []
    test_seg_dice, test_seg_hd, test_residual_mae, test_residual_mse = [], [], [], []
    test_seg_dice_gt = []
    for iter_idx, (images, timestamps) in enumerate(tqdm(test_set)):

        if iter_idx > config.max_testing_samples:
            break

        x_list, t_arr = convert_variables(images, timestamps, device)
        x_start, x_end = x_list[0], x_list[-1]

        # For compatibility
        t_list = [t_arr[0], t_arr[-1]]

        x_end_pred = interpolator.interp(image_arr=torch.vstack(x_list[:-1]), time_arr=t_arr[:-1], target_t=t_arr[-1])

        x_start_seg = segmentor(x_start) > 0.5
        x_end_seg = segmentor(x_end) > 0.5
        x_end_pred_seg = segmentor(x_end_pred) > 0.5

        # NOTE: Recon is meaningless for interpolation. Just using ground truth as placeholder.
        x0_true, x0_recon, xT_true, xT_recon, xT_pred, x0_seg, xT_seg, xT_pred_seg = \
            numpy_variables(x_start, x_start, x_end, x_end, x_end_pred,
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

        # Plot an overall scattering plot.
        deltaT_list.append(0)
        psnr_list.append(psnr(x0_true, x0_recon))
        ssim_list.append(ssim(x0_true, x0_recon))
        deltaT_list.append(0)
        psnr_list.append(psnr(xT_true, xT_recon))
        ssim_list.append(ssim(xT_true, xT_recon))
        deltaT_list.append((t_list[1] - t_list[0]).item())
        psnr_list.append(psnr(xT_true, xT_pred))
        ssim_list.append(ssim(xT_true, xT_pred))

        fig_summary = plt.figure(figsize=(12, 8))
        ax = fig_summary.add_subplot(2, 1, 1)
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.scatter(deltaT_list, psnr_list, color='black', s=50, alpha=0.5)
        ax.set_xlabel('Time difference', fontsize=20)
        ax.set_ylabel('PSNR', fontsize=20)
        ax = fig_summary.add_subplot(2, 1, 2)
        ax.spines[['right', 'top']].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.scatter(deltaT_list, ssim_list, color='crimson', s=50, alpha=0.5)
        ax.set_xlabel('Time difference', fontsize=20)
        ax.set_ylabel('SSIM', fontsize=20)
        fig_summary.tight_layout()
        fig_summary.savefig(save_path_fig_summary)
        plt.close(fig=fig_summary)

        # Plot the side-by-side figures.
        save_path_fig_sbs = '%s/figure_%s.png' % (
            os.path.dirname(save_path_fig_summary), str(iter_idx + 1).zfill(5))
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

    log('Test loss: %.3f, PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f \u00B1 %.3f, SSIM (pred): %.3f \u00B1 %.3f'
        % (test_loss, test_recon_psnr, test_recon_ssim,
            np.mean(test_pred_psnr), np.std(test_pred_psnr) / np.sqrt(len(test_pred_psnr)),
            np.mean(test_pred_ssim), np.std(test_pred_ssim) / np.sqrt(len(test_pred_ssim))) + \
        ' MAE (pred): %.3f \u00B1 %.3f, MSE (pred): %.3f \u00B1 %.3f, DSC (pred): %.3f \u00B1 %.3f, HD (pred): %.3f \u00B1 %.3f'
        % (np.mean(test_residual_mae), np.std(test_residual_mae) / np.sqrt(len(test_residual_mae)),
            np.mean(test_residual_mse), np.std(test_residual_mse) / np.sqrt(len(test_residual_mse)),
            np.mean(test_seg_dice), np.std(test_seg_dice) / np.sqrt(len(test_seg_dice)),
            np.mean(test_seg_hd), np.std(test_seg_hd) / np.sqrt(len(test_seg_hd))),
        filepath=config.log_dir,
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
        filepath=config.log_dir,
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
        filepath=config.log_dir,
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
    results_df.to_csv(config.log_dir.replace('log.txt', 'interp.csv'))

    return


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='`train` or `test`?', default='train')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)

    parser.add_argument('--dataset-name', default='retina_ucsf', type=str)
    parser.add_argument('--target-dim', default='(256, 256)', type=ast.literal_eval)
    parser.add_argument('--image-folder', default='UCSF_images_final_512x512', type=str)
    parser.add_argument('--output-save-folder', default='$ROOT/results/', type=str)
    parser.add_argument('--dataset-path', default='$ROOT/data/retina_ucsf/', type=str)
    parser.add_argument('--segmentor-ckpt', default='$ROOT/checkpoints/segment_retinaUCSF_seed1.pty', type=str)

    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--max-testing-samples', default=500, type=int)

    parser.add_argument('--method', default='linear', type=str)

    args = vars(parser.parse_args())
    config = AttributeHashmap(args)
    config = parse_settings(config)

    seed_everything(config.random_seed)

    test(config=config)
