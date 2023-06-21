# GAN inversion by noise vector optimization.

import argparse
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('../'))
import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-2])
sys.path.insert(0, import_dir + '/stylegan3/')

from torch_utils import gen_utils
import legacy
import dnnlib


def init_noise(generator, random_seed: int, requires_grad: bool):
    truncation_psi = 0.7

    z = np.random.RandomState(random_seed).randn(1, generator.z_dim)
    w_avg = generator.mapping(torch.from_numpy(z).to(device), None)
    w_avg = generator.mapping.w_avg + truncation_psi * (
        w_avg - generator.mapping.w_avg)
    if requires_grad:
        w_avg.requires_grad = True

    return w_avg


def forward_GAN(generator, noise: np.array) -> None:
    if len(noise.shape) == 2:
        noise = noise.unsqueeze(0)
    with torch.no_grad():
        synth_image = generator.synthesis(noise, noise_mode='const')
        # synth_image = synth_image.permute(0, 2, 3, 1).cpu().numpy()  # NCWH => NWHC
        propagate_image = synth_image.cpu().numpy()  # NCWH => NWHC
        display_image = ((synth_image + 1) * 255 / 2).permute(
            0, 2, 3, 1).squeeze(0).clamp(0, 255).to(
                torch.uint8).cpu().numpy()  # NCWH => NWHC

    return propagate_image, display_image


def invert_GAN(device,
               generator,
               image: np.array,
               epochs: int = 1000,
               num_init_attempts: int = 10,
               epochs_init_attempts: int = 100) -> None:
    '''
    Treat the input noise vector as trainable weights, freeze the model,
    and perform gradient descent on the input noise vector.

    For better convergence, we decided to train several different randomizations
    for a few epochs, and continue on the best one.
    '''

    # Freeze generator parameters.
    for param in generator.parameters():
        param.requires_grad = False

    image_true = torch.Tensor(image).to(device)

    # Try several different randomizations for a few epochs.
    best_noise_attempt, best_loss_attempt = None, np.inf

    for i in tqdm(range(num_init_attempts)):
        noise = init_noise(generator=generator,
                           random_seed=i,
                           requires_grad=True)
        optimizer = torch.optim.AdamW([noise], lr=1e-3)

        for _ in range(int(epochs_init_attempts)):
            image_pred = generator.synthesis(noise, noise_mode='const')
            loss = torch.nn.functional.mse_loss(image_pred, image_true)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if loss.item() < best_loss_attempt:
            best_noise_attempt = noise
            best_loss_attempt = loss.item()

    # Get the noise from the best randomization and continue training from there.
    noise = best_noise_attempt
    optimizer = torch.optim.AdamW([noise], lr=1e-3)

    best_noise, best_loss = noise, np.inf
    for _ in tqdm(range(int(epochs))):
        image_pred = generator.synthesis(noise, noise_mode='const')

        loss = torch.nn.functional.mse_loss(image_pred, image_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_noise = noise
            best_loss = loss.item()

    return best_noise, best_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-id',
                        help='Available GPU index.',
                        type=int,
                        default=0)
    args = parser.parse_args()

    num_samples = 4

    device = torch.device('cuda:%d' %
                          args.gpu_id if torch.cuda.is_available() else 'cpu')

    checkpoint = '../stylegan3/training-runs/00001-stylegan3-r--gpus1-batch16-gamma5/network-snapshot-001953.pkl'
    with dnnlib.util.open_url(checkpoint) as fp:
        generator = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(
            False).to(device)

    image_true_list, image_pred_list, mse_loss_list = [], [], []

    for i in range(num_samples):
        # noise_true = torch.from_numpy(
        #     np.random.RandomState(random_seed).randn(
        #         1, generator.mapping.num_ws,
        #         generator.mapping.w_dim)).to(device)
        random_seed = int(np.random.uniform(1000, 10000))
        print('random_seed for true image', random_seed)
        noise_true = init_noise(generator=generator,
                                random_seed=random_seed,
                                requires_grad=True)

        image_true, image_true_display = forward_GAN(generator,
                                                     noise=noise_true)
        noise_pred, mse_loss = invert_GAN(device, generator, image_true)
        _, image_pred_display = forward_GAN(generator, noise=noise_pred)

        image_true_list.append(image_true_display)
        image_pred_list.append(image_pred_display)
        mse_loss_list.append(mse_loss)

    # noise_true = torch.from_numpy(
    #     np.random.RandomState(1).randn(1, generator.mapping.num_ws,
    #                                    generator.mapping.w_dim)).to(device)
    # noise_true = init_noise(generator=generator,
    #                         random_seed=1,
    #                         requires_grad=True)

    image_true, image_true_display = forward_GAN(generator, noise=noise_true)

    # Plot the results.
    plt.rcParams['figure.figsize'] = [6, 2 * num_samples]
    fig = plt.figure()
    for i in range(num_samples):
        ax = fig.add_subplot(num_samples, 2, 2 * i + 1)
        ax.imshow(image_true_list[i])
        ax.axis('off')
        ax.set_title('X [MSE loss: %.2f]' % mse_loss_list[i])
        ax = fig.add_subplot(num_samples, 2, 2 * i + 2)
        ax.imshow(image_pred_list[i])
        ax.axis('off')
        ax.set_title('$G(G^{-1}(X))$')
    fig.savefig('./stylegan_inversion.png')
