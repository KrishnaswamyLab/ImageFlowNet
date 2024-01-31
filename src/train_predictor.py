import argparse
import os
import sys
import cv2
import yaml
from typing import Tuple

from matplotlib import pyplot as plt
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage
from torch.utils.data import Dataset
from tqdm import tqdm
import monai
import albumentations as A

from data_utils.prepare_dataset import prepare_dataset
from nn.scheduler import LinearWarmupCosineAnnealingLR
from utils.attribute_hashmap import AttributeHashmap
from utils.early_stop import EarlyStopping
from utils.log_util import log
from utils.metrics import psnr, ssim, dice_coeff
from utils.parse import parse_settings
from utils.seed import seed_everything

# NOTE: The following imported models are actually used!
from nn.autoencoder import AutoEncoder
from nn.autoencoder_t_emb import T_AutoEncoder
from nn.autoencoder_ode import ODEAutoEncoder
from nn.unet_ode import ODEUNet
from nn.unet_t_emb import T_UNet
from nn.unet_i2sb import I2SBUNet

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/external_src/I2SB/')
from i2sb.diffusion import Diffusion
from i2sb.runner import build_optimizer_sched, make_beta_schedule


def train(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')

    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, border_mode=cv2.BORDER_REPLICATE, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        ],
        additional_targets={
            'image_other': 'image',
        }
    )
    transforms_list = [train_transform, None, None]
    train_set, val_set, test_set, num_image_channel = \
        prepare_dataset(config=config, transforms_list=transforms_list)

    log('Using device: %s' % device, to_console=True)

    # Build the model
    kwargs = {}
    if config.model == 'I2SBUNet':
        step_to_t = torch.linspace(1e-4, 1, config.diffusion_interval, device=device) * config.diffusion_interval
        betas = make_beta_schedule(n_timestep=config.diffusion_interval, linear_end=1 / config.diffusion_interval)
        betas = np.concatenate([betas[:config.diffusion_interval//2], np.flip(betas[:config.diffusion_interval//2])])
        diffusion = Diffusion(betas, device)
        kwargs = {'step_to_t': step_to_t, 'diffusion': diffusion}

    try:
        model = globals()[config.model](device=device,
                                        num_filters=config.num_filters,
                                        depth=config.depth,
                                        in_channels=num_image_channel,
                                        out_channels=num_image_channel,
                                        **kwargs)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    ema = ExponentialMovingAverage(model.parameters(), decay=0.99)

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

    recon_psnr_thr = 30
    recon_good_enough = False

    for epoch_idx in tqdm(range(config.max_epochs)):
        if config.model == 'I2SBUNet':
            model, ema, optimizer, scheduler = \
                train_epoch_I2SB(config=config, device=device, train_set=train_set, model=model,
                                 epoch_idx=epoch_idx, ema=ema, optimizer=optimizer, scheduler=scheduler,
                                 mse_loss=mse_loss, backprop_freq=backprop_freq)
        else:
            model, ema, optimizer, scheduler = \
                train_epoch(config=config, device=device, train_set=train_set, model=model,
                            epoch_idx=epoch_idx, ema=ema, optimizer=optimizer, scheduler=scheduler,
                            mse_loss=mse_loss, backprop_freq=backprop_freq, train_time_dependent=recon_good_enough)

        with ema.average_parameters():
            model.eval()
            if config.model == 'I2SBUNet':
                val_recon_psnr, val_pred_psnr, val_seg_dice_xT = \
                    val_epoch_I2SB(config=config, device=device, val_set=val_set, model=model, epoch_idx=epoch_idx, max_t=train_set.dataset.dataset.max_t)
            else:
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
        # while different patients have different [t_start, t_end] in our dataset.
        # We will simulate a bigger batch size when we handle optimizer update.

        # images: [1, 2, C, H, W], containing [x_start, x_end]
        # timestamps: [1, 2], containing [t_start, t_end]
        assert images.shape[1] == 2
        assert timestamps.shape[1] == 2

        x_start, x_end, t_list = convert_variables(images, timestamps, device)

        ################### Recon Loss to update Encoder/Decoder ##################
        # Unfreeze the model.
        model.unfreeze()

        x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
        x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))

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
            x_start_pred = model(x=x_end, t=-torch.diff(t_list) * config.t_multiplier)
            x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)

            loss_pred = mse_loss(x_start, x_start_pred) + mse_loss(x_end, x_end_pred)
            train_loss_pred += loss_pred.item()

            # Simulate `config.batch_size` by batched optimizer update.
            loss_pred = loss_pred / backprop_freq
            loss_pred.backward()

        else:
            # Will not train the time-dependent modules until the reconstruction is good enough.
            with torch.no_grad():
                x_start_pred = model(x=x_end, t=-torch.diff(t_list) * config.t_multiplier)
                x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)
                loss_pred = mse_loss(x_start, x_start_pred) + mse_loss(x_end, x_end_pred)
                train_loss_pred += loss_pred.item()

        # Simulate `config.batch_size` by batched optimizer update.
        if iter_idx % config.batch_size == config.batch_size - 1:
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

        x0_true, x0_recon, x0_pred, xT_true, xT_recon, xT_pred = \
            numpy_variables(x_start, x_start_recon, x_start_pred, x_end, x_end_recon, x_end_pred)

        train_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
        train_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
        train_pred_psnr += psnr(x0_true, x0_pred) / 2 + psnr(xT_true, xT_pred) / 2
        train_pred_ssim += ssim(x0_true, x0_pred) / 2 + ssim(xT_true, xT_pred) / 2

        if shall_plot:
            save_path_fig_sbs = '%s/train/figure_log_epoch%s_sample%s.png' % (
                config.save_folder, str(epoch_idx + 1).zfill(5), str(iter_idx + 1).zfill(5))
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred, save_path_fig_sbs)

    train_loss_pred, train_loss_recon, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = \
        [item / len(train_set.dataset) for item in (train_loss_pred, train_loss_recon, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim)]

    scheduler.step()

    log('Train [%s/%s] loss [recon: %.3f, pred: %.3f], PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f'
        % (epoch_idx + 1, config.max_epochs, train_loss_recon, train_loss_pred, train_recon_psnr,
            train_recon_ssim, train_pred_psnr, train_pred_ssim),
        filepath=config.log_dir,
        to_console=False)

    return model, ema, optimizer, scheduler


def train_epoch_I2SB(config: AttributeHashmap,
                     device: torch.device,
                     train_set: Dataset,
                     model: torch.nn.Module,
                     epoch_idx: int,
                     ema: ExponentialMovingAverage,
                     optimizer: torch.optim.Optimizer,
                     scheduler: torch.optim.lr_scheduler._LRScheduler,
                     mse_loss: torch.nn.Module,
                     backprop_freq: int):
    '''
    Training epoch for I2SB: Image-to-Image Schrodinger Bridge.
    '''

    train_loss_diffusion, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = 0, 0, 0, 0, 0
    model.train()
    optimizer.zero_grad()

    plot_freq = int(len(train_set) // config.n_plot_per_epoch)
    for iter_idx, (images, timestamps) in enumerate(tqdm(train_set)):

        if 'max_training_samples' in config:
            if iter_idx > config.max_training_samples:
                break

        shall_plot = iter_idx % plot_freq == 0

        # NOTE: batch size is set to 1,
        # because in Neural ODE, `eval_times` has to be a 1-D Tensor,
        # while different patients have different [t_start, t_end] in our dataset.
        # We will simulate a bigger batch size when we handle optimizer update.

        # images: [1, 2, C, H, W], containing [x_start, x_end]
        # timestamps: [1, 2], containing [t_start, t_end]
        assert images.shape[1] == 2
        assert timestamps.shape[1] == 2

        x_start, x_end, t_list = convert_variables(images, timestamps, device)

        delta_t_normalized = torch.diff(t_list) / train_set.dataset.dataset.max_t
        step = torch.randint(0, int(config.diffusion_interval * delta_t_normalized), (x_end.shape[0],))
        x_interp = model.diffusion.q_sample(step, x_end, x_start)
        x_interp_pseudo_gt = _compute_label(model.diffusion, step, x_end, x_interp)
        diffusion_time = model.step_to_t[step]

        x_interp_pred = model(x=x_start, t=diffusion_time)
        loss_diffusion = mse_loss(x_interp_pseudo_gt, x_interp_pred)

        # Simulate `config.batch_size` by batched optimizer update.
        loss_diffusion = loss_diffusion / backprop_freq
        loss_diffusion.backward()

        train_loss_diffusion += loss_diffusion.item()

        # Simulate `config.batch_size` by batched optimizer update.
        if iter_idx % config.batch_size == config.batch_size - 1:
            optimizer.step()
            optimizer.zero_grad()
            ema.update()

        # x_start, x_end, x_interp, x_interp_pseudo_gt, x_interp_pred = \
        #     numpy_variables(x_start, x_end, x_interp, x_interp_pseudo_gt, x_interp_pred)
        # fig = plt.figure(figsize=(10, 4))
        # ax = fig.add_subplot(1, 5, 1)
        # ax.imshow(np.clip((x_start + 1) / 2, 0, 1), cmap='gray')
        # ax = fig.add_subplot(1, 5, 2)
        # ax.imshow(np.clip((x_end + 1) / 2, 0, 1), cmap='gray')
        # ax = fig.add_subplot(1, 5, 3)
        # ax.imshow(np.clip((x_interp + 1) / 2, 0, 1), cmap='gray')
        # ax = fig.add_subplot(1, 5, 4)
        # ax.imshow(np.clip((x_interp_pseudo_gt + 1) / 2, 0, 1), cmap='gray')
        # ax = fig.add_subplot(1, 5, 5)
        # ax.imshow(np.clip((x_interp_pred + 1) / 2, 0, 1), cmap='gray')
        # fig.savefig('test.png')
        # import pdb
        # pdb.set_trace()

        # There are only for plotting purposes.
        with ema.average_parameters():
            model.eval()
            # Almost reconstruction (step = [0, 1]).
            x_start_recon = model.ddpm_sampling(x_start=x_start, steps=np.int16(np.linspace(0, 1, 2)).tolist())
            x_end_recon = model.ddpm_sampling(x_start=x_end, steps=np.int16(np.linspace(0, 1, 2)).tolist())

            assert torch.diff(t_list).item() > 0
            x_start_pred = torch.zeros_like(x_start)
            step_max = int(config.diffusion_interval * delta_t_normalized)
            x_end_pred = model.ddpm_sampling(x_start=x_end,
                                             steps=np.int16(np.linspace(0, step_max, step_max + 1)).tolist())

        model.train()

        x0_true, x0_recon, x0_pred, xT_true, xT_recon, xT_pred = \
            numpy_variables(x_start, x_start_recon, x_start_pred, x_end, x_end_recon, x_end_pred)

        train_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
        train_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
        train_pred_psnr += psnr(xT_true, xT_pred) / 2
        train_pred_ssim += ssim(xT_true, xT_pred) / 2

        if shall_plot:
            save_path_fig_sbs = '%s/train/figure_log_epoch%s_sample%s.png' % (
                config.save_folder, str(epoch_idx + 1).zfill(5), str(iter_idx + 1).zfill(5))
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred, save_path_fig_sbs)

    train_loss_diffusion, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = \
        [item / len(train_set.dataset) for item in (train_loss_diffusion, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim)]

    scheduler.step()

    log('Train [%s/%s] loss [diffusion: %.3f], PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f'
        % (epoch_idx + 1, config.max_epochs, train_loss_diffusion, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim),
        filepath=config.log_dir,
        to_console=False)

    return model, ema, optimizer, scheduler

def _compute_label(diffusion, step, x0, xt):
    """ I2SB Eq 12 """
    std_fwd = diffusion.get_std_fwd(step, xdim=x0.shape[1:])
    label = (xt - x0) / std_fwd
    return label.detach()


@torch.no_grad()
def val_epoch(config: AttributeHashmap,
              device: torch.device,
              val_set: Dataset,
              model: torch.nn.Module,
              epoch_idx: int):
    val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim = 0, 0, 0, 0
    val_seg_dice_x0, val_seg_dice_xT, val_seg_dice_gt = 0, 0, 0

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

        x_start_pred = model(x=x_end, t=-torch.diff(t_list) * config.t_multiplier)
        x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)

        x_start_seg = segmentor(x_start) > 0.5
        x_end_seg = segmentor(x_end) > 0.5
        x_start_pred_seg = segmentor(x_start_pred) > 0.5
        x_end_pred_seg = segmentor(x_end_pred) > 0.5

        x0_true, x0_recon, x0_pred, xT_true, xT_recon, xT_pred, x0_seg, xT_seg, x0_pred_seg, xT_pred_seg = \
            numpy_variables(x_start, x_start_recon, x_start_pred, x_end, x_end_recon, x_end_pred,
                            x_start_seg, x_end_seg, x_start_pred_seg, x_end_pred_seg)

        val_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
        val_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
        val_pred_psnr += psnr(x0_true, x0_pred) / 2 + psnr(xT_true, xT_pred) / 2
        val_pred_ssim += ssim(x0_true, x0_pred) / 2 + ssim(xT_true, xT_pred) / 2

        val_seg_dice_x0 += dice_coeff(x0_seg, x0_pred_seg)
        val_seg_dice_xT += dice_coeff(xT_seg, xT_pred_seg)
        val_seg_dice_gt += dice_coeff(x0_seg, xT_seg)

        if shall_plot:
            save_path_fig_sbs = '%s/val/figure_log_epoch%s_sample%s.png' % (
                config.save_folder, str(epoch_idx + 1).zfill(5), str(iter_idx + 1).zfill(5))
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred, save_path_fig_sbs,
                            x0_pred_seg=x0_pred_seg, x0_true_seg=x0_seg, xT_pred_seg=xT_pred_seg, xT_true_seg=xT_seg)

    del segmentor

    val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_x0, val_seg_dice_xT, val_seg_dice_gt = \
        [item / len(val_set.dataset) for item in (
            val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_x0, val_seg_dice_xT, val_seg_dice_gt)]

    log('Validation [%s/%s] PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f, Dice(x0_true, x0_pred): %.3f, Dice(xT_true, xT_pred): %.3f, Dice(x0_true, xT_true): %.3f.'
        % (epoch_idx + 1, config.max_epochs, val_recon_psnr,
        val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_x0, val_seg_dice_xT, val_seg_dice_gt),
        filepath=config.log_dir,
        to_console=False)

    return val_recon_psnr, val_pred_psnr, val_seg_dice_xT


@torch.no_grad()
def val_epoch_I2SB(config: AttributeHashmap,
                   device: torch.device,
                   val_set: Dataset,
                   model: torch.nn.Module,
                   epoch_idx: int,
                   max_t: int):
    val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim = 0, 0, 0, 0
    val_seg_dice_x0, val_seg_dice_xT, val_seg_dice_gt = 0, 0, 0

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

        # Almost reconstruction (step = [0, 1]).
        x_start_recon = model.ddpm_sampling(x_start=x_start, steps=np.int16(np.linspace(0, 1, 2)).tolist())
        x_end_recon = model.ddpm_sampling(x_start=x_end, steps=np.int16(np.linspace(0, 1, 2)).tolist())

        assert torch.diff(t_list).item() > 0
        delta_t_normalized = torch.diff(t_list) / max_t
        x_start_pred = torch.zeros_like(x_start)
        step_max = int(config.diffusion_interval * delta_t_normalized)
        x_end_pred = model.ddpm_sampling(x_start=x_end,
                                         steps=np.int16(np.linspace(0, step_max, step_max + 1)).tolist())


        x_start_seg = segmentor(x_start) > 0.5
        x_end_seg = segmentor(x_end) > 0.5
        x_start_pred_seg = segmentor(x_start_pred) > 0.5
        x_end_pred_seg = segmentor(x_end_pred) > 0.5

        x0_true, x0_recon, x0_pred, xT_true, xT_recon, xT_pred, x0_seg, xT_seg, x0_pred_seg, xT_pred_seg = \
            numpy_variables(x_start, x_start_recon, x_start_pred, x_end, x_end_recon, x_end_pred,
                            x_start_seg, x_end_seg, x_start_pred_seg, x_end_pred_seg)

        val_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
        val_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
        val_pred_psnr += psnr(x0_true, x0_pred) / 2 + psnr(xT_true, xT_pred) / 2
        val_pred_ssim += ssim(x0_true, x0_pred) / 2 + ssim(xT_true, xT_pred) / 2

        val_seg_dice_x0 += dice_coeff(x0_seg, x0_pred_seg)
        val_seg_dice_xT += dice_coeff(xT_seg, xT_pred_seg)
        val_seg_dice_gt += dice_coeff(x0_seg, xT_seg)

        if shall_plot:
            save_path_fig_sbs = '%s/val/figure_log_epoch%s_sample%s.png' % (
                config.save_folder, str(epoch_idx + 1).zfill(5), str(iter_idx + 1).zfill(5))
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred, save_path_fig_sbs,
                              x0_pred_seg=x0_pred_seg, x0_true_seg=x0_seg, xT_pred_seg=xT_pred_seg, xT_true_seg=xT_seg)

    del segmentor

    val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_x0, val_seg_dice_xT, val_seg_dice_gt = \
        [item / len(val_set.dataset) for item in (
            val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_x0, val_seg_dice_xT, val_seg_dice_gt)]

    log('Validation [%s/%s] PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f, Dice(x0_true, x0_pred): %.3f, Dice(xT_true, xT_pred): %.3f, Dice(x0_true, xT_true): %.3f.'
        % (epoch_idx + 1, config.max_epochs, val_recon_psnr,
        val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_x0, val_seg_dice_xT, val_seg_dice_gt),
        filepath=config.log_dir,
        to_console=False)

    return val_recon_psnr, val_pred_psnr, val_seg_dice_xT

@torch.no_grad()
def test(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    train_set, val_set, test_set, num_image_channel = \
        prepare_dataset(config=config)

    # Build the model
    try:
        model = globals()[config.model](num_filters=config.num_filters,
                                        depth=config.depth,
                                        in_channels=num_image_channel,
                                        out_channels=num_image_channel)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    model.to(device)
    model.load_weights(config.model_save_path.replace('.pty', '_best_pred_psnr.pty'), device=device)
    log('%s: Model weights successfully loaded.' % config.model,
        to_console=True)

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

    save_path_fig_summary = '%s/results/summary.png' % config.save_folder
    os.makedirs(os.path.dirname(save_path_fig_summary), exist_ok=True)

    mse_loss = torch.nn.MSELoss()

    deltaT_list, psnr_list, ssim_list = [], [], []
    test_loss, test_recon_psnr, test_recon_ssim, test_pred_psnr, test_pred_ssim = 0, 0, 0, 0, 0
    for iter_idx, (images, timestamps) in enumerate(tqdm(test_set)):
        assert images.shape[1] == 2
        assert timestamps.shape[1] == 2

        assert images.shape[1] == 2
        assert timestamps.shape[1] == 2

        x_start, x_end, t_list = convert_variables(images, timestamps, device)

        x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
        x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))

        x_start_pred = model(x=x_end, t=-torch.diff(t_list) * config.t_multiplier)
        x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)

        loss_recon = mse_loss(x_start, x_start_recon) + mse_loss(
            x_end, x_end_recon)
        loss_pred = mse_loss(x_start, x_start_pred) + mse_loss(x_end, x_end_pred)

        loss = loss_recon + loss_pred
        test_loss += loss.item()

        x_start_seg = segmentor(x_start) > 0.5
        x_end_seg = segmentor(x_end) > 0.5
        x_start_pred_seg = segmentor(x_start_pred) > 0.5
        x_end_pred_seg = segmentor(x_end_pred) > 0.5

        x0_true, x0_recon, x0_pred, xT_true, xT_recon, xT_pred, x0_seg, xT_seg, x0_pred_seg, xT_pred_seg = \
            numpy_variables(x_start, x_start_recon, x_start_pred, x_end, x_end_recon, x_end_pred,
                            x_start_seg, x_end_seg, x_start_pred_seg, x_end_pred_seg)

        test_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(
            xT_true, xT_recon) / 2
        test_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(
            xT_true, xT_recon) / 2
        test_pred_psnr += psnr(x0_true, x0_pred) / 2 + psnr(xT_true,
                                                            xT_pred) / 2
        test_pred_ssim += ssim(x0_true, x0_pred) / 2 + ssim(xT_true,
                                                            xT_pred) / 2

        # Plot an overall scattering plot.
        deltaT_list.append(0)
        psnr_list.append(psnr(x0_true, x0_recon))
        ssim_list.append(ssim(x0_true, x0_recon))
        deltaT_list.append(0)
        psnr_list.append(psnr(xT_true, xT_recon))
        ssim_list.append(ssim(xT_true, xT_recon))
        deltaT_list.append((t_list[0] - t_list[1]).item())
        psnr_list.append(psnr(x0_true, x0_pred))
        ssim_list.append(ssim(x0_true, x0_pred))
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
        if iter_idx < 20:
            save_path_fig_sbs = '%s/figure_%s.png' % (
                os.path.dirname(save_path_fig_summary), str(iter_idx + 1).zfill(5))
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred, save_path_fig_sbs,
                              x0_pred_seg=x0_pred_seg, x0_true_seg=x0_seg, xT_pred_seg=xT_pred_seg, xT_true_seg=xT_seg)

    test_loss = test_loss / len(test_set.dataset)
    test_recon_psnr = test_recon_psnr / len(test_set.dataset)
    test_recon_ssim = test_recon_ssim / len(test_set.dataset)
    test_pred_psnr = test_pred_psnr / len(test_set.dataset)
    test_pred_ssim = test_pred_ssim / len(test_set.dataset)

    log('Test loss: %.3f, PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f'
        % (test_loss, test_recon_psnr, test_recon_ssim, test_pred_psnr,
           test_pred_ssim),
        filepath=config.log_dir,
        to_console=True)
    return


def convert_variables(images: torch.Tensor,
                      timestamps: torch.Tensor,
                      device: torch.device) -> Tuple[torch.Tensor]:
    '''
    Some repetitive processing of variables.
    '''
    x_start = images[:, 0, ...].float().to(device)
    x_end = images[:, 1, ...].float().to(device)
    t_list = timestamps[0].float().to(device)
    return x_start, x_end, t_list


def numpy_variables(*tensors: torch.Tensor) -> Tuple[np.array]:
    '''
    Some repetitive numpy casting of variables.
    '''
    return [_tensor.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0) for _tensor in tensors]

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

def plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred, save_path: str,
                      x0_pred_seg=None, x0_true_seg=None, xT_pred_seg=None, xT_true_seg=None) -> None:
    fig_sbs = plt.figure(figsize=(24, 10))

    aspect_ratio = x0_true.shape[0] / x0_true.shape[1]

    assert len(x0_true.shape) in [2, 3]
    if len(x0_true.shape) == 2 or x0_true.shape[-1] == 1:
        x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred = \
            gray_to_rgb(x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred)

    # First column: Ground Truth.
    ax = fig_sbs.add_subplot(2, 6, 1)
    ax.imshow(np.clip((x0_true + 1) / 2, 0, 1))
    ax.set_title('GT, time: %s' % t_list[0].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 7)
    ax.imshow(np.clip((xT_true + 1) / 2, 0, 1))
    ax.set_title('GT, time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Second column: Reconstruction.
    ax = fig_sbs.add_subplot(2, 6, 2)
    ax.imshow(np.clip((x0_recon + 1) / 2, 0, 1))
    ax.set_title('Recon, time: %s' % t_list[0].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 8)
    ax.imshow(np.clip((xT_recon + 1) / 2, 0, 1))
    ax.set_title('Recon, time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Third column: Prediction.
    ax = fig_sbs.add_subplot(2, 6, 3)
    ax.imshow(np.clip((x0_pred + 1) / 2, 0, 1))
    ax.set_title('Pred, time: %s -> time: %s' % (t_list[1].item(), t_list[0].item()))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 9)
    ax.imshow(np.clip((xT_pred + 1) / 2, 0, 1))
    ax.set_title('Pred, time: %s -> time: %s' % (t_list[0].item(), t_list[1].item()))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Fourth column: |Ground Truth - Prediction|.
    ax = fig_sbs.add_subplot(2, 6, 4)
    ax.imshow(np.clip(np.abs((x0_true + 1) / 2 - (x0_pred + 1) / 2), 0, 1))
    ax.set_title('|GT - Pred|, time: %s' % t_list[0].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 10)
    ax.imshow(np.clip(np.abs((xT_true + 1) / 2 - (xT_pred + 1) / 2), 0, 1))
    ax.set_title('|GT - Pred|, time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Fifth column: Ground Truth with segmentation.
    ax = fig_sbs.add_subplot(2, 6, 5)
    image = np.clip((x0_true + 1) / 2, 0, 1)
    if x0_true_seg is not None:
        plot_contour(image, x0_true_seg)
        ax.imshow(image)
        ax.set_title('GT, time: %s' % t_list[0].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 11)
    image = np.clip((xT_true + 1) / 2, 0, 1)
    if xT_true_seg is not None:
        plot_contour(image, xT_true_seg)
        ax.imshow(image)
        ax.set_title('GT, time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    # Sixth column: Prediction with segmentation.
    ax = fig_sbs.add_subplot(2, 6, 6)
    image = np.clip((x0_pred + 1) / 2, 0, 1)
    if x0_pred_seg is not None:
        plot_contour(image, x0_pred_seg)
        ax.imshow(image)
        ax.set_title('Pred, time: %s -> time: %s' % (t_list[1].item(), t_list[0].item()))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 6, 12)
    image = np.clip((xT_pred + 1) / 2, 0, 1)
    if xT_pred_seg is not None:
        plot_contour(image, xT_pred_seg)
        ax.imshow(image)
        ax.set_title('Pred, time: %s -> time: %s' % (t_list[0].item(), t_list[1].item()))
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
