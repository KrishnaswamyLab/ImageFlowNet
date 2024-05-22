import argparse
import os
import cv2
import yaml
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset
from tqdm import tqdm
import monai
import albumentations as A

from data_utils.prepare_dataset import prepare_dataset_npt
from nn.scheduler import LinearWarmupCosineAnnealingLR
from utils.attribute_hashmap import AttributeHashmap
from utils.early_stop import EarlyStopping
from utils.log_util import log
from utils.metrics import psnr, ssim, dice_coeff, hausdorff
from utils.parse import parse_settings
from utils.seed import seed_everything

# NOTE: The following imported models are actually used!
from nn.unet_cde import CDEUNet


def add_random_noise(img: torch.Tensor, max_intensity: float = 0.1) -> torch.Tensor:
    intensity = max_intensity * torch.rand(1).to(img.device)
    noise = intensity * torch.randn_like(img)
    return img + noise

def train(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    # NOTE: `additional_targets` here is a dirty hack.
    # I want to ensure the same augmentation to all images in the entire subsequence.
    # But I don't know of a more elegant way of doing so.
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=1.0),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, p=1.0),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ],
        additional_targets={
            'image_other1': 'image',
            'image_other2': 'image',
            'image_other3': 'image',
            'image_other4': 'image',
            'image_other5': 'image',
            'image_other6': 'image',
            'image_other7': 'image',
            'image_other8': 'image',
            'image_other9': 'image',
        }
    )
    transforms_list = [train_transform, None, None]
    train_set, val_set, test_set, num_image_channel, max_t = \
        prepare_dataset_npt(config=config, transforms_list=transforms_list)

    log('Using device: %s' % device, to_console=True)

    # Build the model
    assert config.model == 'CDEUNet'
    model = CDEUNet(device=device,
                    num_filters=config.num_filters,
                    depth=config.depth,
                    in_channels=num_image_channel,
                    out_channels=num_image_channel)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.9)

    model.to(device)
    model.init_params()
    ema.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                              warmup_epochs=config.max_epochs//10,
                                              max_epochs=config.max_epochs)

    early_stopper = EarlyStopping(mode='max',
                                  patience=config.patience,
                                  percentage=False)

    mse_loss = torch.nn.MSELoss()
    best_val_psnr, best_val_dice = 0, 0
    backprop_freq = config.batch_size
    if 'n_plot_per_epoch' not in config.keys():
        config.n_plot_per_epoch = 1

    os.makedirs(config.save_folder + 'train/', exist_ok=True)
    os.makedirs(config.save_folder + 'val/', exist_ok=True)

    recon_psnr_thr = 20
    recon_good_enough = False

    for epoch_idx in tqdm(range(config.max_epochs)):
        model, ema, optimizer, scheduler = \
            train_epoch(config=config, device=device, train_set=train_set, model=model,
                        epoch_idx=epoch_idx, ema=ema, optimizer=optimizer, scheduler=scheduler,
                        mse_loss=mse_loss, backprop_freq=backprop_freq, train_time_dependent=recon_good_enough)

        with ema.average_parameters():
            model.eval()
            val_recon_psnr, val_pred_psnr, val_seg_dice_xT = \
                val_epoch(config=config, device=device, val_set=val_set, model=model, epoch_idx=epoch_idx)

            if val_recon_psnr > recon_psnr_thr:
                recon_good_enough = True

            if val_pred_psnr > best_val_psnr:
                best_val_psnr = val_pred_psnr
                model.save_weights(config.model_save_path.replace('.pty', '_best_pred_psnr.pty'))
                log('%s: Model weights successfully saved for best pred PSNR.' % config.model,
                    filepath=config.log_dir,
                    to_console=False)

            if val_seg_dice_xT > best_val_dice:
                best_val_dice = val_seg_dice_xT
                model.save_weights(config.model_save_path.replace('.pty', '_best_seg_dice.pty'))
                log('%s: Model weights successfully saved for best dice xT.' % config.model,
                    filepath=config.log_dir,
                    to_console=False)

        if early_stopper.step(val_pred_psnr):
            log('Early stopping criterion met. Ending training.',
                filepath=config.log_dir,
                to_console=True)
            break
    return


def train_epoch(config: AttributeHashmap,
                device: torch.device,
                train_set: Dataset,
                model: torch.nn.Module,
                epoch_idx: int,
                ema: ExponentialMovingAverage,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                mse_loss: torch.nn.Module,
                backprop_freq: int,
                train_time_dependent: bool):
    '''
    Training epoch for many models.
    '''

    train_loss_recon, train_loss_pred, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = 0, 0, 0, 0, 0, 0
    model.train()
    optimizer.zero_grad()

    if not train_time_dependent:
        log('[Epoch %d] Will not train the time-dependent modules until the reconstruction is good enough.' % (epoch_idx + 1),
            filepath=config.log_dir,
            to_console=False)

    plot_freq = int(len(train_set) // config.n_plot_per_epoch)
    for iter_idx, (images, timestamps) in enumerate(tqdm(train_set)):

        if 'max_training_samples' in config:
            if iter_idx > config.max_training_samples:
                break

        shall_plot = iter_idx % plot_freq == 0

        # NOTE: batch size is set to 1,
        # because in Neural ODE, `eval_times` has to be a 1-D Tensor,
        # while different patients have different `timestamps` in our dataset.
        # We will simulate a bigger batch size when we handle optimizer update.

        # images: [1, N, C, H, W], containing [x_start, ..., x_end]
        # timestamps: [1, N], containing [t_start, ..., t_end]
        assert images.shape[1] >= 2
        assert timestamps.shape[1] >= 2

        x_list, t_arr = convert_variables(images, timestamps, device)
        x_noisy_list = [add_random_noise(x) for x in x_list]

        x_end_pred = model(x=torch.vstack(x_noisy_list[:-1]), t=(t_arr - t_arr[0]) * config.t_multiplier)
        import pdb
        pdb.set_trace()

        ################### Recon Loss to update Encoder/Decoder ##################
        # Unfreeze the model.
        model.unfreeze()

        x_start_recon = model(x=x_start_noisy, t=torch.zeros(1).to(device))
        x_end_recon = model(x=x_end_noisy, t=torch.zeros(1).to(device))

        loss_recon = mse_loss(x_start, x_start_recon) + mse_loss(x_end, x_end_recon)
        train_loss_recon += loss_recon.item()

        # Simulate `config.batch_size` by batched optimizer update.
        loss_recon = loss_recon / backprop_freq
        loss_recon.backward()

        ################## Pred Loss to update time-dependent modules #############
        # Freeze all time-independent modules.
        try:
            model.freeze_time_independent()
        except AttributeError:
            print('`model.freeze_time_independent()` ignored.')

        if train_time_dependent:
            assert torch.diff(t_list).item() > 0
            x_end_pred = model(x=x_start_noisy, t=torch.diff(t_list) * config.t_multiplier)
            loss_pred = mse_loss(x_end, x_end_pred)
            train_loss_pred += loss_pred.item()

            # Simulate `config.batch_size` by batched optimizer update.
            loss_pred = loss_pred / backprop_freq
            loss_pred.backward()

        else:
            # Will not train the time-dependent modules until the reconstruction is good enough.
            with torch.no_grad():
                x_end_pred = model(x=x_start_noisy, t=torch.diff(t_list) * config.t_multiplier)
                loss_pred = mse_loss(x_end, x_end_pred)
                train_loss_pred += loss_pred.item()

        # Simulate `config.batch_size` by batched optimizer update.
        if iter_idx % config.batch_size == config.batch_size - 1:
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            numpy_variables(x_start, x_start_recon, x_end, x_end_recon, x_end_pred)

        # NOTE: Convert to image with normal dynamic range.
        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            cast_to_0to1(x0_true, x0_recon, xT_true, xT_recon, xT_pred)

        train_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
        train_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
        train_pred_psnr += psnr(xT_true, xT_pred)
        train_pred_ssim += ssim(xT_true, xT_pred)

        if shall_plot:
            save_path_fig_sbs = '%s/train/figure_log_epoch%s_sample%s.png' % (
                config.save_folder, str(epoch_idx + 1).zfill(5), str(iter_idx + 1).zfill(5))
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, xT_pred, save_path_fig_sbs)

    train_loss_pred, train_loss_recon, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = \
        [item / len(train_set.dataset) for item in (train_loss_pred, train_loss_recon, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim)]

    scheduler.step()

    log('Train [%s/%s] loss [recon: %.3f, pred: %.3f], PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f'
        % (epoch_idx + 1, config.max_epochs, train_loss_recon, train_loss_pred, train_recon_psnr,
            train_recon_ssim, train_pred_psnr, train_pred_ssim),
        filepath=config.log_dir,
        to_console=False)

    return model, ema, optimizer, scheduler


@torch.no_grad()
def val_epoch(config: AttributeHashmap,
              device: torch.device,
              val_set: Dataset,
              model: torch.nn.Module,
              epoch_idx: int):
    val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim = 0, 0, 0, 0
    val_seg_dice_xT, val_seg_dice_gt = 0, 0

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

    plot_freq = int(len(val_set) // config.n_plot_per_epoch)
    for iter_idx, (images, timestamps) in enumerate(tqdm(val_set)):
        shall_plot = iter_idx % plot_freq == 0

        assert images.shape[1] == 2
        assert timestamps.shape[1] == 2

        # images: [1, 2, C, H, W], containing [x_start, x_end]
        # timestamps: [1, 2], containing [t_start, t_end]
        x_start, x_end, t_list = convert_variables(images, timestamps, device)

        x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
        x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))
        x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)

        x_start_seg = segmentor(x_start) > 0.5
        x_end_seg = segmentor(x_end) > 0.5
        x_end_pred_seg = segmentor(x_end_pred) > 0.5

        x0_true, x0_recon, xT_true, xT_recon, xT_pred, x0_seg, xT_seg, xT_pred_seg = \
            numpy_variables(x_start, x_start_recon, x_end, x_end_recon, x_end_pred,
                            x_start_seg, x_end_seg, x_end_pred_seg)

        # NOTE: Convert to image with normal dynamic range.
        x0_true, x0_recon, xT_true, xT_recon, xT_pred = \
            cast_to_0to1(x0_true, x0_recon, xT_true, xT_recon, xT_pred)

        val_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
        val_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
        val_pred_psnr += psnr(xT_true, xT_pred)
        val_pred_ssim += ssim(xT_true, xT_pred)

        val_seg_dice_xT += dice_coeff(xT_seg, xT_pred_seg)
        val_seg_dice_gt += dice_coeff(x0_seg, xT_seg)

        if shall_plot:
            save_path_fig_sbs = '%s/val/figure_log_epoch%s_sample%s.png' % (
                config.save_folder, str(epoch_idx + 1).zfill(5), str(iter_idx + 1).zfill(5))
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, xT_pred, save_path_fig_sbs,
                              x0_true_seg=x0_seg, xT_pred_seg=xT_pred_seg, xT_true_seg=xT_seg)

    del segmentor

    val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_xT, val_seg_dice_gt = \
        [item / len(val_set.dataset) for item in (
            val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_xT, val_seg_dice_gt)]

    log('Validation [%s/%s] PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f, Dice(xT_true, xT_pred): %.3f, Dice(x0_true, xT_true): %.3f.'
        % (epoch_idx + 1, config.max_epochs, val_recon_psnr,
        val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_xT, val_seg_dice_gt),
        filepath=config.log_dir,
        to_console=False)

    return val_recon_psnr, val_pred_psnr, val_seg_dice_xT


@torch.no_grad()
def test(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    train_set, val_set, test_set, num_image_channel, max_t = \
        prepare_dataset_npt(config=config)

    # Build the model
    assert config.model == 'CDEUNet'
    model = CDEUNet(device=device,
                    num_filters=config.num_filters,
                    depth=config.depth,
                    in_channels=num_image_channel,
                    out_channels=num_image_channel)
    model.to(device)

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

    for best_type in ['pred_psnr', 'seg_dice']:
        if best_type == 'pred_psnr':

            model.load_weights(config.model_save_path.replace('.pty', '_best_pred_psnr.pty'), device=device)
            log('%s: Model weights successfully loaded.' % config.model,
                to_console=True)

            save_path_fig_summary = '%s/results_best_pred_psnr/summary.png' % config.save_folder
            os.makedirs(os.path.dirname(save_path_fig_summary), exist_ok=True)

        elif best_type == 'seg_dice':

            model.load_weights(config.model_save_path.replace('.pty', '_best_seg_dice.pty'), device=device)
            log('%s: Model weights successfully loaded.' % config.model,
                to_console=True)

            save_path_fig_summary = '%s/results_best_seg_dice/summary.png' % config.save_folder
            os.makedirs(os.path.dirname(save_path_fig_summary), exist_ok=True)

        mse_loss = torch.nn.MSELoss()

        deltaT_list, psnr_list, ssim_list = [], [], []
        test_loss, test_recon_psnr, test_recon_ssim, test_pred_psnr, test_pred_ssim = 0, 0, 0, [], []
        test_seg_dice, test_seg_hd, test_residual_mae, test_residual_mse = [], [], [], []
        test_seg_dice_gt = []
        for iter_idx, (images, timestamps) in enumerate(tqdm(test_set)):
            assert images.shape[1] == 2
            assert timestamps.shape[1] == 2

            assert images.shape[1] == 2
            assert timestamps.shape[1] == 2

            x_start, x_end, t_list = convert_variables(images, timestamps, device)

            x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
            x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))
            x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)

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

        test_loss = test_loss / len(test_set.dataset)
        test_recon_psnr = test_recon_psnr / len(test_set.dataset)
        test_recon_ssim = test_recon_ssim / len(test_set.dataset)

        log('[Best %s] Test loss: %.3f, PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f'
            % (best_type, test_loss, test_recon_psnr, test_recon_ssim, np.mean(test_pred_psnr), np.mean(test_pred_ssim)) + \
            ' DSC (pred): %.3f, HD (pred): %.3f, MAE (pred): %.3f, MSE (pred): %.3f'
            % (np.mean(test_seg_dice), np.mean(test_seg_hd), np.mean(test_residual_mae), np.mean(test_residual_mse)),
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
        results_df.to_csv(config.log_dir.replace('.txt', best_type + '.csv'))

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
    parser.add_argument('--run_count', default=None, type=int)
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config = parse_settings(config, log_settings=args.mode == 'train', run_count=args.run_count)

    assert args.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if args.mode == 'train':
        train(config=config)
    #     test(config=config)
    elif args.mode == 'test':
        test(config=config)
