import argparse

import numpy as np
import torch
import yaml
from data_utils.prepare_dataset import prepare_dataset
from nn.ode_models import ResUNetPP_ODE
from nn.scheduler import LinearWarmupCosineAnnealingLR
from nn.unet_models import ResUNetPP
from tqdm import tqdm
from utils.attribute_hashmap import AttributeHashmap
from utils.early_stop import EarlyStopping
from utils.log_util import log
from utils.metrics import psnr, ssim
from utils.parse import parse_settings
from utils.seed import seed_everything


def train(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    train_set, val_set, test_set, num_image_channel = \
        prepare_dataset(config=config)

    # Build the model
    if config.model == 'ConvODEResUNet':
        raise NotImplementedError
        # model = ConvODEResUNet(device=device,
        #                        num_filters=config.num_filters,
        #                        in_channels=num_image_channel,
        #                        out_channels=num_image_channel,
        #                        return_t0=config.ode_return_t0,
        #                        augment_dim=config.ode_augment_dim,
        #                        time_dependent=config.ode_time_dependent,
        #                        tol=config.ode_tol,
        #                        adjoint=config.ode_adjoint,
        #                        max_num_steps=config.ode_steps)
    elif config.model == 'ResUNetPP_ODE':
        model = ResUNetPP_ODE(device=device,
                              num_filters=config.num_filters,
                              in_channels=num_image_channel,
                              out_channels=num_image_channel,
                              return_t0=config.ode_return_t0,
                              augment_dim=config.ode_augment_dim,
                              time_dependent=config.ode_time_dependent,
                              tol=config.ode_tol,
                              adjoint=config.ode_adjoint,
                              max_num_steps=config.ode_steps)
    elif config.model == 'ResUNet':
        raise NotImplementedError
        # model = ResUNet(device=device,
        #                 num_filters=config.num_filters,
        #                 in_channels=num_image_channel,
        #                 out_channels=num_image_channel)
    elif config.model == 'ResUNetPP':
        model = ResUNetPP(device=device,
                          num_filters=config.num_filters,
                          in_channels=num_image_channel,
                          out_channels=num_image_channel,
                          depth=config.model_depth)
    else:
        raise ValueError('`config.model`: %s not supported.' % config.model)

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,
                                                 warmup_epochs=10,
                                                 max_epochs=config.max_epochs,
                                                 eta_min=0)
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    loss_fn = torch.nn.MSELoss()
    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss, train_psnr, train_ssim = 0, 0, 0
        model.train()
        optimizer.zero_grad()
        for iter_idx, (images, timestamps) in enumerate(train_set):
            # NOTE: batch size is set to 1,
            # because `eval_times` has to be a 1-D Tensor,
            # while different patients have different [t_start, t_end] in our dataset.
            # We will simulate a bigger batch size when we handle optimizer update.

            # images: [1, 2, C, H, W], containing [x_start, x_end]
            # timestamps: [1, 2], containing [t_start, t_end]
            assert images.shape[1] == 2
            assert timestamps.shape[1] == 2

            x_start = images[:, 0, ...].type(torch.FloatTensor).to(device)
            x_end = images[:, 1, ...].type(torch.FloatTensor).to(device)
            eval_times = timestamps[0].type(torch.FloatTensor).to(device)

            x_pred = model(x=x_start, eval_times=eval_times)
            if len(x_pred) > 1:
                x_start_pred, x_end_pred = x_pred
                loss = loss_fn(x_start, x_start_pred) + loss_fn(
                    x_end, x_end_pred)
            else:
                x_end_pred = x_pred[0]
                loss = loss_fn(x_end, x_end_pred)

            train_loss += loss.item()

            image_true = x_end.cpu().detach().numpy().squeeze(0).transpose(
                1, 2, 0)
            image_pred = x_end_pred.cpu().detach().numpy().squeeze(
                0).transpose(1, 2, 0)

            train_psnr += psnr(image_true, image_pred)
            train_ssim += ssim(image_true, image_pred)

            # Simulate `config.batch_size` by batched optimizer update.
            loss.backward()
            if iter_idx % config.batch_size == 0:
                optimizer.step()
                optimizer.zero_grad()

        train_loss = train_loss / len(train_set.dataset)
        train_psnr = train_psnr / len(train_set.dataset)
        train_ssim = train_ssim / len(train_set.dataset)

        lr_scheduler.step()

        log('Train [%s/%s] loss: %.3f, PSNR: %.3f, SSIM: %.3f' %
            (epoch_idx + 1, config.max_epochs, train_loss, train_psnr,
             train_ssim),
            filepath=config.log_dir,
            to_console=False)

        val_loss, val_psnr, val_ssim = 0, 0, 0
        model.eval()
        with torch.no_grad():
            for (images, timestamps) in val_set:
                assert images.shape[1] == 2
                assert timestamps.shape[1] == 2

                # images: [1, 2, C, H, W], containing [x_start, x_end]
                # timestamps: [1, 2], containing [t_start, t_end]
                x_start = images[:, 0, ...].type(torch.FloatTensor).to(device)
                x_end = images[:, 1, ...].type(torch.FloatTensor).to(device)
                eval_times = timestamps[0].type(torch.FloatTensor).to(device)

                if len(x_pred) > 1:
                    x_start_pred, x_end_pred = x_pred
                    loss = loss_fn(x_start, x_start_pred) + loss_fn(
                        x_end, x_end_pred)
                else:
                    x_end_pred = x_pred[0]
                    loss = loss_fn(x_end, x_end_pred)
                val_loss += loss.item()

                image_true = x_end.cpu().numpy().squeeze(0).transpose(1, 2, 0)
                image_pred = x_end_pred.cpu().numpy().squeeze(0).transpose(
                    1, 2, 0)
                val_psnr += psnr(image_true, image_pred)
                val_ssim += ssim(image_true, image_pred)

        val_loss = val_loss / len(val_set.dataset)
        val_psnr = val_psnr / len(val_set.dataset)
        val_ssim = val_ssim / len(val_set.dataset)

        log('Validation [%s/%s] loss: %.3f, PSNR: %.3f, SSIM: %.3f' %
            (epoch_idx + 1, config.max_epochs, val_loss, val_psnr, val_ssim),
            filepath=config.log_dir,
            to_console=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(config.model_save_path)
            log('%s: Model weights successfully saved.' % config.model,
                filepath=config.log_dir,
                to_console=False)

        if early_stopper.step(val_loss):
            # If the validation loss stop decreasing, stop training.
            log('Early stopping criterion met. Ending training.',
                filepath=config.log_dir,
                to_console=True)
            break
    return


def test(config: AttributeHashmap):
    device = torch.device(
        'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
    train_set, val_set, test_set, num_image_channel = \
        prepare_dataset(config=config)

    # Build the model
    if config.model == 'ConvODEResUNet':
        model = ConvODEResUNet(device=device,
                               num_filters=config.num_filters,
                               in_channels=num_image_channel,
                               out_channels=num_image_channel,
                               augment_dim=config.ode_augment_dim,
                               time_dependent=config.ode_time_dependent,
                               tol=config.ode_tol,
                               adjoint=config.ode_adjoint,
                               max_num_steps=config.ode_steps)
    elif config.model == 'ResUNet':
        model = ResUNet(device=device,
                        num_filters=config.num_filters,
                        in_channels=num_image_channel,
                        out_channels=num_image_channel)
    else:
        raise ValueError('`config.model`: %s not supported.' % config.model)
    model.to(device)
    model.load_weights(config.model_save_path, device=device)
    log('%s: Model weights successfully loaded.' % config.model,
        to_console=True)

    loss_fn = torch.nn.MSELoss()
    # output_saver = OutputSaver(save_path=config.output_save_path,
    #                            random_seed=config.random_seed)

    # test_loss_recon, test_loss_contrastive, test_loss = 0, 0, 0
    # model.eval()

    with torch.no_grad():
        for iter_idx, (images, timestamps) in enumerate(test_set):
            assert images.shape[1] == 2
            assert timestamps.shape[1] == 2

            x_start = images[:, 0, ...].type(torch.FloatTensor).to(device)
            x_end = images[:, 1, ...].type(torch.FloatTensor).to(device)
            eval_times = timestamps[0].type(torch.FloatTensor).to(device)

            x_pred = model(x=x_start, eval_times=eval_times)
            if len(x_pred) > 1:
                x_start_pred, x_end_pred = x_pred
            else:
                x_end_pred = x_pred[0]

            from matplotlib import pyplot as plt

            x_start = (x_start + 1) / 2
            x_end = (x_end + 1) / 2
            x_end_pred = (x_end_pred + 1) / 2
            plt.subplot(1, 3, 1)
            plt.imshow(x_start.cpu().numpy().squeeze(0).transpose(1, 2, 0))
            plt.title('time: %s' % eval_times[0].item())
            plt.subplot(1, 3, 2)
            plt.imshow(x_end.cpu().numpy().squeeze(0).transpose(1, 2, 0))
            plt.title('time: %s' % eval_times[1].item())
            plt.subplot(1, 3, 3)
            plt.imshow(x_end_pred.cpu().numpy().squeeze(0).transpose(1, 2, 0))
            plt.savefig('test.png')

            import pdb
            pdb.set_trace()

    #         # Each pixel embedding recons to a patch.
    #         # Here we only take the center pixel of the reconed patch and collect into a reconed image.
    #         B, L, H, W = z.shape
    #         z_for_recon = z.permute((0, 2, 3, 1)).reshape(B, H * W, L)
    #         patch_recon = model.recon(z_for_recon)
    #         C = patch_recon.shape[2]
    #         P = patch_recon.shape[-1]
    #         patch_recon = patch_recon[:, :, :, P // 2, P // 2]
    #         patch_recon = patch_recon.permute((0, 2, 1)).reshape(B, C, H, W)

    #         output_saver.save(
    #             image_batch=x_test,
    #             recon_batch=patch_recon,
    #             label_true_batch=y_test if config.no_label is False else None,
    #             latent_batch=z)

    test_loss = test_loss / len(test_set.dataset)

    log('Test loss: %.3f' % (test_loss),
        filepath=config.log_dir,
        to_console=True)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='`train` or `test`?', required=True)
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
