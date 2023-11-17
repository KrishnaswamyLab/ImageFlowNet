'''
In this version, we use a main network to perform longitudinal prediction,
and an auxiliary network to

'''

import argparse
from typing import Tuple, List
from matplotlib import pyplot as plt
import numpy as np
import torch
import os
import yaml
from data_utils.prepare_dataset import prepare_dataset
from nn.scheduler import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from nn.autoencoder import AutoEncoder
from nn.autoencoder_t_emb import T_AutoEncoder
from nn.autoencoder_ode import ODEAutoEncoder
from nn.unet_ode import ODEUNet
from nn.aux_net import AuxNet
from utils.attribute_hashmap import AttributeHashmap
from utils.early_stop import EarlyStopping
from utils.log_util import log
from utils.metrics import psnr, ssim
from utils.parse import parse_settings
from utils.seed import seed_everything

# NOTE: The current training logic is:
# The training comes with 2 stages.
# In the first stage, the main network is trained with pixel-level losses,
# for both reconstruction and longitudinal prediction.
# In the second stage, the main network is trained with pixel-level loss for reconstruction
# but with auxiliary-network guidance on longitudinal prediction.
# In both stages, the auxiliary network is trained to perform classification.
def train(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    train_set, val_set, test_set, num_image_channel = \
        prepare_dataset(config=config)

    # Build the model
    try:
        model = globals()[config.model](num_filters=config.num_filters,
                                        depth=config.depth,
                                        use_residual=config.use_residual,
                                        in_channels=num_image_channel,
                                        out_channels=num_image_channel)
    except:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    model_aux = AuxNet(num_filters=config.num_filters,
                       depth=config.depth,
                       use_residual=config.use_residual,
                       in_channels=num_image_channel,
                       out_channels=1)

    model.to(device)
    model_aux.to(device)

    model.init_params()
    model_aux.init_params()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    optimizer_aux = torch.optim.AdamW(model_aux.parameters(), lr=config.learning_rate)
    lr_scheduler = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer,
        warmup_epochs=10,
        warmup_start_lr=float(config.learning_rate) / 100,
        max_epochs=config.max_epochs,
        eta_min=0)
    lr_scheduler_aux = LinearWarmupCosineAnnealingLR(
        optimizer=optimizer_aux,
        warmup_epochs=10,
        warmup_start_lr=float(config.learning_rate) / 100,
        max_epochs=config.max_epochs,
        eta_min=0)
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    mse_loss = torch.nn.MSELoss()
    bce_loss = torch.nn.BCELoss()
    best_val_loss = np.inf

    save_folder_fig_log = '%s/log/' % config.output_save_path
    os.makedirs(save_folder_fig_log + 'train/', exist_ok=True)
    os.makedirs(save_folder_fig_log + 'val/', exist_ok=True)

    for epoch_idx in tqdm(range(config.max_epochs)):
        running_stage1 = epoch_idx < config.epochs_stage1

        train_loss, train_loss_aux, num_correct, num_total, train_loss_recon, train_loss_pred, \
            train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        model.train()
        model_aux.train()
        optimizer.zero_grad()
        optimizer_aux.zero_grad()

        for iter_idx, (images, timestamps, pos_pair, neg_pair1, neg_pair2) in enumerate(tqdm(train_set)):
            shall_plot = iter_idx == 0

            # NOTE: batch size is set to 1,
            # because in Neural ODE, `eval_times` has to be a 1-D Tensor,
            # while different patients have different [t_start, t_end] in our dataset.
            # We will simulate a bigger batch size when we handle optimizer update.

            # images: [1, 2, C, H, W], containing [x_start, x_end]
            # timestamps: [1, 2], containing [t_start, t_end]
            assert images.shape[1] == 2
            assert timestamps.shape[1] == 2
            assert pos_pair.shape[1] == 2
            assert neg_pair1.shape[1] == 2
            assert neg_pair2.shape[1] == 2

            x_start, x_end, t_list, pos_pair, neg_pair1, neg_pair2 = \
                convert_variables(images, timestamps, pos_pair, neg_pair1, neg_pair2, device)

            # NOTE: Optimize auxiliary network.
            ones = torch.ones(images.shape[0], device=device)
            zeros = torch.zeros(images.shape[0], device=device)
            cls_prob_pos = model_aux.forward_cls(pos_pair[0], pos_pair[1]).mean(1).view(-1)
            cls_prob_neg1 = model_aux.forward_cls(neg_pair1[0], neg_pair1[1]).mean(1).view(-1)
            cls_prob_neg2 = model_aux.forward_cls(neg_pair2[0], neg_pair2[1]).mean(1).view(-1)
            loss_aux = 2 * bce_loss(cls_prob_pos, ones) + \
                bce_loss(cls_prob_neg1, zeros) + bce_loss(cls_prob_neg2, zeros)
            num_correct += (cls_prob_pos > 0.5) * 2 + (cls_prob_neg1 < 0.5) + (cls_prob_neg2 < 0.5)
            num_total += 4

            train_loss_aux += loss_aux.item()

            # Simulate `config.batch_size` by batched optimizer update.
            loss_aux.backward()
            if iter_idx % config.batch_size == 0:
                # torch.nn.utils.clip_grad_norm_(model_aux.parameters(), 0.1)
                optimizer_aux.step()
                optimizer_aux.zero_grad()

            # NOTE: Optimize main network.
            x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
            x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))

            assert torch.diff(t_list).item() > 0
            x_start_pred = model(x=x_end, t=-torch.diff(t_list))
            x_end_pred = model(x=x_start, t=torch.diff(t_list))

            loss_recon = mse_loss(x_start, x_start_recon) + mse_loss(x_end, x_end_recon)

            if shall_plot or not running_stage1:
                cls_prob_x0 = model_aux.forward_cls(x_start, x_start_pred).mean(1).view(-1)
                cls_prob_xT = model_aux.forward_cls(x_end, x_end_pred).mean(1).view(-1)

            if running_stage1:
                loss_pred = mse_loss(x_start, x_start_pred) + mse_loss(x_end, x_end_pred)
            else:
                loss_pred = bce_loss(cls_prob_x0, ones) + bce_loss(cls_prob_xT, ones)

            loss = loss_recon + loss_pred

            train_loss += loss.item()
            train_loss_recon += loss_recon.item()
            train_loss_pred += loss_pred.item()

            # Simulate `config.batch_size` by batched optimizer update.
            loss.backward()
            if iter_idx % config.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

            x0_true, x0_recon, x0_pred, xT_true, xT_recon, xT_pred = \
                numpy_variables(x_start, x_start_recon, x_start_pred, x_end, x_end_recon, x_end_pred)

            train_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
            train_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
            train_pred_psnr += psnr(x0_true, x0_pred) / 2 + psnr(xT_true, xT_pred) / 2
            train_pred_ssim += ssim(x0_true, x0_pred) / 2 + ssim(xT_true, xT_pred) / 2

            if shall_plot:
                save_path_fig_sbs = '%s/train/figure_log_epoch_%s.png' % (
                    save_folder_fig_log, str(epoch_idx).zfill(5))
                posA, posB, neg1A, neg1B, neg2A, neg2B = \
                    numpy_variables(pos_pair[0], pos_pair[1], neg_pair1[0], neg_pair1[1], neg_pair2[0], neg_pair2[1])
                plot_side_by_side(t_list, save_path_fig_sbs,
                                  x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred,
                                  posA, posB, neg1A, neg1B, neg2A, neg2B,
                                  cls_prob_x0, cls_prob_xT, cls_prob_pos, cls_prob_neg1, cls_prob_neg2)

        train_loss, train_loss_aux, train_loss_recon, train_loss_pred, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim = \
            [item / len(train_set.dataset) for item in (train_loss, train_loss_aux, train_loss_recon, train_loss_pred, train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim)]
        train_aux_acc = num_correct / num_total * 100

        lr_scheduler.step()
        lr_scheduler_aux.step()

        log('Train [%s/%s] Stage 1? %s, loss: %.3f [aux: %.3f, recon: %.3f, pred: %.3f], PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f, Aux Acc: %.3f'
            % (epoch_idx + 1, config.max_epochs, running_stage1, train_loss, train_loss_aux, train_loss_recon, train_loss_pred,
               train_recon_psnr, train_recon_ssim, train_pred_psnr, train_pred_ssim, train_aux_acc),
            filepath=config.log_dir,
            to_console=False)

        val_loss, val_loss_recon, val_loss_pred, \
            val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim = 0, 0, 0, 0, 0, 0, 0
        model.eval()
        model_aux.eval()
        with torch.no_grad():
            for iter_idx, (images, timestamps, pos_pair, neg_pair1, neg_pair2) in enumerate(tqdm(val_set)):
                shall_plot = iter_idx == 0
                # images: [1, 2, C, H, W], containing [x_start, x_end]
                # timestamps: [1, 2], containing [t_start, t_end]
                assert images.shape[1] == 2
                assert timestamps.shape[1] == 2
                assert pos_pair.shape[1] == 2
                assert neg_pair1.shape[1] == 2
                assert neg_pair2.shape[1] == 2

                x_start, x_end, t_list, pos_pair, neg_pair1, neg_pair2 = \
                    convert_variables(images, timestamps, pos_pair, neg_pair1, neg_pair2, device)

                x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
                x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))

                x_start_pred = model(x=x_end, t=-torch.diff(t_list))
                x_end_pred = model(x=x_start, t=torch.diff(t_list))

                x0_true, x0_recon, x0_pred, xT_true, xT_recon, xT_pred = \
                    numpy_variables(x_start, x_start_recon, x_start_pred, x_end, x_end_recon, x_end_pred)

                loss_recon = mse_loss(x_start, x_start_recon) + mse_loss(x_end, x_end_recon)

                if shall_plot:
                    cls_prob_pos = model_aux.forward_cls(pos_pair[0], pos_pair[1]).mean(1).view(-1)
                    cls_prob_neg1 = model_aux.forward_cls(neg_pair1[0], neg_pair1[1]).mean(1).view(-1)
                    cls_prob_neg2 = model_aux.forward_cls(neg_pair2[0], neg_pair2[1]).mean(1).view(-1)

                cls_prob_x0 = model_aux.forward_cls(x_start, x_start_pred).mean(1).view(-1)
                cls_prob_xT = model_aux.forward_cls(x_end, x_end_pred).mean(1).view(-1)

                if running_stage1:
                    loss_pred = mse_loss(x_start, x_start_pred) + mse_loss(x_end, x_end_pred)
                else:
                    loss_pred = bce_loss(cls_prob_x0, ones) + bce_loss(cls_prob_xT, ones)

                loss = loss_recon + loss_pred

                val_loss += loss.item()
                val_loss_recon += loss_recon.item()
                val_loss_pred += loss_pred.item()

                val_recon_psnr += psnr(x0_true, x0_recon) / 2 + psnr(xT_true, xT_recon) / 2
                val_recon_ssim += ssim(x0_true, x0_recon) / 2 + ssim(xT_true, xT_recon) / 2
                val_pred_psnr += psnr(x0_true, x0_pred) / 2 + psnr(xT_true, xT_pred) / 2
                val_pred_ssim += ssim(x0_true, x0_pred) / 2 + ssim(xT_true, xT_pred) / 2

                if shall_plot:
                    save_path_fig_sbs = '%s/val/figure_log_epoch_%s.png' % (
                        save_folder_fig_log, str(epoch_idx).zfill(5))
                    posA, posB, neg1A, neg1B, neg2A, neg2B = \
                        numpy_variables(pos_pair[0], pos_pair[1], neg_pair1[0], neg_pair1[1], neg_pair2[0], neg_pair2[1])
                    plot_side_by_side(t_list, save_path_fig_sbs,
                                      x0_true, xT_true, x0_recon, xT_recon, x0_pred, xT_pred,
                                      posA, posB, neg1A, neg1B, neg2A, neg2B,
                                      cls_prob_x0, cls_prob_xT, cls_prob_pos, cls_prob_neg1, cls_prob_neg2)

        val_loss, val_loss_recon, val_loss_pred, val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim = \
            [item / len(val_set.dataset) for item in (val_loss, val_loss_recon, val_loss_pred, val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim)]

        log('Validation [%s/%s] loss: %.3f [recon: %.3f, pred: %.3f], PSNR (recon): %.3f, SSIM (recon): %.3f, PSNR (pred): %.3f, SSIM (pred): %.3f'
            % (epoch_idx + 1, config.max_epochs, val_loss, val_loss_recon, val_loss_pred,
               val_recon_psnr, val_recon_ssim, val_pred_psnr, val_pred_ssim),
            filepath=config.log_dir,
            to_console=False)

        if not running_stage1:
            # Only work on best model saving and early stopping after reaching stage 2.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                model.save_weights(config.model_save_path)
                model_aux.save_weights(config.model_aux_save_path)
                log('%s: Model weights successfully saved.' % config.model,
                    filepath=config.log_dir,
                    to_console=False)
            if early_stopper.step(val_loss):
                log('Early stopping criterion met. Ending training.',
                    filepath=config.log_dir,
                    to_console=True)
                break
    return


#
#TODO: test is not updated yet.
#
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

    loss_fn = torch.nn.MSELoss()

    deltaT_list, psnr_list, ssim_list = [], [], []
    test_loss, test_recon_psnr, test_recon_ssim, test_pred_psnr, test_pred_ssim = 0, 0, 0, 0, 0
    for iter_idx, (images, timestamps) in enumerate(tqdm(test_set)):
        assert images.shape[1] == 2
        assert timestamps.shape[1] == 2

        assert images.shape[1] == 2
        assert timestamps.shape[1] == 2

        x_start = images[:, 0, ...].float().to(device)
        x_end = images[:, 1, ...].float().to(device)
        t_list = timestamps[0].float().to(device)

        x_start_recon = model(x=x_start, t=torch.zeros(1).to(device))
        x_end_recon = model(x=x_end, t=torch.zeros(1).to(device))

        x_start_pred = model(x=x_end, t=-torch.diff(t_list))
        x_end_pred = model(x=x_start, t=torch.diff(t_list))

        loss_recon = loss_fn(x_start, x_start_recon) + loss_fn(
            x_end, x_end_recon)
        loss_pred = loss_fn(x_start, x_start_pred) + loss_fn(x_end, x_end_pred)

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
                      pos_pair: torch.Tensor,
                      neg_pair1: torch.Tensor,
                      neg_pair2: torch.Tensor,
                      device: torch.device) -> Tuple[torch.Tensor]:
    '''
    Some repetitive processing of variables.
    '''
    x_start = images[:, 0, ...].float().to(device)
    x_end = images[:, 1, ...].float().to(device)
    t_list = timestamps[0].float().to(device)
    pos_pair = [pos_pair[:, 0, ...].float().to(device), pos_pair[:, 1, ...].float().to(device)]
    neg_pair1 = [neg_pair1[:, 0, ...].float().to(device), neg_pair1[:, 1, ...].float().to(device)]
    neg_pair2 = [neg_pair2[:, 0, ...].float().to(device), neg_pair2[:, 1, ...].float().to(device)]
    return x_start, x_end, t_list, pos_pair, neg_pair1, neg_pair2


def numpy_variables(*tensors: torch.Tensor) -> Tuple[np.array]:
    '''
    Some repetitive numpy casting of variables.
    '''
    return [_tensor.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0) for _tensor in tensors]


def plot_side_by_side(t_list: List[int],
                      save_path: str,
                      x0_true: np.array, xT_true: np.array,
                      x0_recon: np.array, xT_recon: np.array,
                      x0_pred: np.array, xT_pred: np.array,
                      posA: np.array, posB: np.array,
                      neg1A: np.array, neg1B: np.array,
                      neg2A: np.array, neg2B: np.array,
                      prob_x0, prob_xT,
                      prob_pos, prob_neg1, prob_neg2) -> None:
    fig_sbs = plt.figure(figsize=(36, 10))

    aspect_ratio = x0_true.shape[0] / x0_true.shape[1]

    ax = fig_sbs.add_subplot(2, 9, 1)
    ax.imshow(np.clip((x0_true + 1) / 2, 0, 1))
    ax.set_title('GT, time: %s' % t_list[0].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 9, 10)
    ax.imshow(np.clip((xT_true + 1) / 2, 0, 1))
    ax.set_title('GT, time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, 9, 2)
    ax.imshow(np.clip((x0_recon + 1) / 2, 0, 1))
    ax.set_title('Recon, time: %s' % t_list[0].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 9, 11)
    ax.imshow(np.clip((xT_recon + 1) / 2, 0, 1))
    ax.set_title('Recon, time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, 9, 3)
    ax.imshow(np.clip((x0_pred + 1) / 2, 0, 1))
    ax.set_title('Pred, input time: %s' % t_list[1].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 9, 12)
    ax.imshow(np.clip((xT_pred + 1) / 2, 0, 1))
    ax.set_title('Pred, input time: %s' % t_list[0].item())
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, 9, 5)
    ax.imshow(np.clip((x0_true + 1) / 2, 0, 1))
    ax.set_title('AuxNet Prob: %.2f' % prob_x0)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 9, 14)
    ax.set_title('x0 True/Pred')
    ax.imshow(np.clip((x0_pred + 1) / 2, 0, 1))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, 9, 6)
    ax.imshow(np.clip((xT_true + 1) / 2, 0, 1))
    ax.set_title('AuxNet Prob: %.2f' % prob_xT)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 9, 15)
    ax.set_title('xT True/Pred')
    ax.imshow(np.clip((xT_pred + 1) / 2, 0, 1))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, 9, 7)
    ax.imshow(np.clip((posA + 1) / 2, 0, 1))
    ax.set_title('AuxNet Prob: %.2f' % prob_pos)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 9, 16)
    ax.set_title('positive pair')
    ax.imshow(np.clip((posA + 1) / 2, 0, 1))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, 9, 8)
    ax.imshow(np.clip((neg1A + 1) / 2, 0, 1))
    ax.set_title('AuxNet Prob: %.2f' % prob_neg1)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 9, 17)
    ax.set_title('negative pair (same subject)')
    ax.imshow(np.clip((neg1B + 1) / 2, 0, 1))
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)

    ax = fig_sbs.add_subplot(2, 9, 9)
    ax.imshow(np.clip((neg2A + 1) / 2, 0, 1))
    ax.set_title('AuxNet Prob: %.2f' % prob_neg2)
    ax.set_axis_off()
    ax.set_aspect(aspect_ratio)
    ax = fig_sbs.add_subplot(2, 9, 18)
    ax.set_title('negative pair (different subject)')
    ax.imshow(np.clip((neg2B + 1) / 2, 0, 1))
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
