import argparse
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import cv2
import yaml
from data_utils.prepare_dataset import prepare_dataset
from nn.scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
import monai

from nn.autoencoder import AutoEncoder
from nn.autoencoder_t_emb import T_AutoEncoder
from nn.autoencoder_ode import ODEAutoEncoder
from nn.unet_ode import ODEUNet
from utils.attribute_hashmap import AttributeHashmap
from utils.early_stop import EarlyStopping
from utils.log_util import log
from utils.metrics import psnr, ssim, dice_coeff
from utils.parse import parse_settings
from utils.seed import seed_everything


def train(config: AttributeHashmap):
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
    model.init_params()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=10,
        warmup_start_lr=float(config.learning_rate) / 100,
        max_epochs=config.max_epochs,
        eta_min=0)
    early_stopper = EarlyStopping(mode='max',
                                  patience=config.patience,
                                  percentage=False)

    mse_loss = torch.nn.MSELoss()
    best_val_psnr = 0
    backprop_freq = config.batch_size

    save_folder_fig_log = '%s/log/' % config.output_save_path
    os.makedirs(save_folder_fig_log + 'train/', exist_ok=True)
    os.makedirs(save_folder_fig_log + 'val/', exist_ok=True)


    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss_recon, train_loss_pred, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = 0, 0, 0, 0, 0, 0
        model.train()
        optimizer.zero_grad()
        for iter_idx, (images, timestamps) in enumerate(tqdm(train_set)):

            if 'max_training_samples' in config:
                if iter_idx > config.max_training_samples:
                    break

            if 'plot_freq' not in config.keys():
                config.plot_freq = len(train_set)
            shall_plot = iter_idx % config.plot_freq == 0

            # NOTE: batch size is set to 1,
            # because in Neural ODE, `eval_times` has to be a 1-D Tensor,
            # while different patients have different [t_start, t_end] in our dataset.
            # We will simulate a bigger batch size when we handle optimizer update.

            # images: [1, 2, C, H, W], containing [x_start, x_end]
            # timestamps: [1, 2], containing [t_start, t_end]
            assert images.shape[1] == 2
            assert timestamps.shape[1] == 2

            x_start, x_end, t_list = convert_variables(images, timestamps, device)

            ##################### Recon Loss to update Encoder/Decoder ################
            # Unfreeze the model.
            model.unfreeze()

            x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
            x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))

            loss_recon = mse_loss(x_start, x_start_recon) + mse_loss(x_end, x_end_recon)
            train_loss_recon += loss_recon.item()

            # Simulate `config.batch_size` by batched optimizer update.
            loss_ = loss_recon / backprop_freq
            loss_.backward()

            ########################## Pred Loss to update ODE ########################
            # Freeze all modules other than ODE.
            try:
                model.freeze_non_ode()
            except AttributeError:
                print('`model.freeze_non_ode()` ignored.')

            assert torch.diff(t_list).item() > 0
            x_start_pred = model(x=x_end, t=-torch.diff(t_list) * config.t_multiplier)
            x_end_pred = model(x=x_start, t=torch.diff(t_list) * config.t_multiplier)

            loss_pred = mse_loss(x_start, x_start_pred) + mse_loss(x_end, x_end_pred)
            train_loss_pred += loss_pred.item()

            # Simulate `config.batch_size` by batched optimizer update.
            loss_ = loss_pred / backprop_freq
            loss_.backward()

            # Simulate `config.batch_size` by batched optimizer update.
            if iter_idx % config.batch_size == config.batch_size - 1:
                optimizer.step()
                optimizer.zero_grad()

            x0_true, x0_recon, x0_pred, xT_true, xT_recon, xT_pred = \
                numpy_variables(x_start, x_start_recon, x_start_pred, x_end, x_end_recon, x_end_pred)

            train_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
            train_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
            train_pred_psnr += psnr(x0_true, x0_pred) / 2 + psnr(xT_true, xT_pred) / 2
            train_pred_ssim += ssim(x0_true, x0_pred) / 2 + ssim(xT_true, xT_pred) / 2

            if shall_plot:
                save_path_fig_sbs = '%s/train/figure_log_epoch%s_sample%s.png' % (
                    save_folder_fig_log, str(epoch_idx).zfill(5), str(iter_idx).zfill(5))
                plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred, save_path_fig_sbs)

        train_loss_pred, train_loss_recon, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = \
            [item / len(train_set.dataset) for item in (train_loss_pred, train_loss_recon, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim)]

        lr_scheduler.step()

        log('Train [%s/%s] loss [recon: %.3f, pred: %.3f], PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f'
            % (epoch_idx + 1, config.max_epochs, train_loss_recon, train_loss_pred, train_recon_psnr,
               train_recon_ssim, train_pred_psnr, train_pred_ssim),
            filepath=config.log_dir,
            to_console=False)

        val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim = 0, 0, 0, 0
        val_seg_dice_x0, val_seg_dice_xT = 0, 0
        model.eval()
        with torch.no_grad():
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

            for iter_idx, (images, timestamps) in enumerate(tqdm(val_set)):
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

                if iter_idx == 10:
                    save_path_fig_sbs = '%s/val/figure_log_epoch_%s.png' % (
                        save_folder_fig_log, str(epoch_idx).zfill(5))
                    plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred, save_path_fig_sbs,
                                      x0_pred_seg=x0_pred_seg, x0_true_seg=x0_seg, xT_pred_seg=xT_pred_seg, xT_true_seg=xT_seg)

            del segmentor

        val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_x0, val_seg_dice_xT = \
            [item / len(val_set.dataset) for item in (
                val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_x0, val_seg_dice_xT)]

        log('Validation [%s/%s] PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f, Dice(true,pred) x0: %.3f, Dice(true,pred) xT: %.3f'
            % (epoch_idx + 1, config.max_epochs, val_recon_psnr,
               val_recon_ssim, val_pred_psnr, val_pred_ssim, val_seg_dice_x0, val_seg_dice_xT),
            filepath=config.log_dir,
            to_console=False)

        if val_pred_psnr > best_val_psnr:
            best_val_psnr = val_pred_psnr
            model.save_weights(config.model_save_path)
            log('%s: Model weights successfully saved.' % config.model,
                filepath=config.log_dir,
                to_console=False)

        if early_stopper.step(val_pred_psnr):
            log('Early stopping criterion met. Ending training.',
                filepath=config.log_dir,
                to_console=True)
            break
    return


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
    model.load_weights(config.model_save_path, device=device)
    log('%s: Model weights successfully loaded.' % config.model,
        to_console=True)

    save_path_fig_summary = '%s/results/summary.png' % config.output_save_path
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

        x0_true, x0_recon, x0_pred, xT_true, xT_recon, xT_pred = \
            numpy_variables(x_start, x_start_recon, x_start_pred, x_end, x_end_recon, x_end_pred)

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
                os.path.dirname(save_path_fig_summary), str(iter_idx).zfill(5))
            plot_side_by_side(t_list, x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred, save_path_fig_sbs)

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
    fig_sbs = plt.figure(figsize=(12, 10))

    aspect_ratio = x0_true.shape[0] / x0_true.shape[1]

    assert len(x0_true.shape) in [2, 3]
    if len(x0_true.shape) == 2 or x0_true.shape[-1] == 1:
        x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred = \
            gray_to_rgb(x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred)

    ax = fig_sbs.add_subplot(2, 3, 1)
    image = np.clip((x0_true + 1) / 2, 0, 1)
    if x0_true_seg is not None:
        plot_contour(image, x0_true_seg)
    ax.imshow(image)
    ax.set_title('GT, time: %s' % t_list[0].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 3, 4)
    image = np.clip((xT_true + 1) / 2, 0, 1)
    if xT_true_seg is not None:
        plot_contour(image, xT_true_seg)
    ax.imshow(image)
    ax.set_title('GT, time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, 3, 2)
    ax.imshow(np.clip((x0_recon + 1) / 2, 0, 1))
    ax.set_title('Recon, time: %s' % t_list[0].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 3, 5)
    ax.imshow(np.clip((xT_recon + 1) / 2, 0, 1))
    ax.set_title('Recon, time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, 3, 3)
    image = np.clip((x0_pred + 1) / 2, 0, 1)
    if x0_pred_seg is not None:
        plot_contour(image, x0_pred_seg)
    ax.imshow(image)
    ax.set_title('Pred, input time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 3, 6)
    image = np.clip((xT_pred + 1) / 2, 0, 1)
    if xT_pred_seg is not None:
        plot_contour(image, xT_pred_seg)
    ax.imshow(image)
    ax.set_title('Pred, input time: %s' % t_list[0].item())
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
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config.gpu_id = args.gpu_id
    config = parse_settings(config, log_settings=args.mode == 'train')

    assert args.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if args.mode == 'train':
        train(config=config)
    #     test(config=config)
    elif args.mode == 'test':
        test(config=config)
