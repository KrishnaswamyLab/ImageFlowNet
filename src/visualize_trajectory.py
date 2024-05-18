import argparse
import ast
import os
import cv2
from typing import Tuple, Literal
import phate

import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from data_utils.prepare_dataset import prepare_dataset_full_sequence
from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log
from utils.metrics import psnr, ssim, dice_coeff, hausdorff
from utils.seed import seed_everything
from utils.parse import parse_settings

# NOTE: The following imported models are actually used!
from nn.autoencoder import AutoEncoder
from nn.autoencoder_t_emb import T_AutoEncoder
from nn.autoencoder_ode import ODEAutoEncoder
from nn.unet_ode import ODEUNet
from nn.unet_ode_static import StaticODEUNet
from nn.unet_ode_simple import ODEUNetSimple
from nn.unet_ode_simple_static import StaticODEUNetSimple
from nn.unet_t_emb import T_UNet
from nn.unet_i2sb import I2SBUNet

from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0,0), (0,0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0],ys[0]),(xs[1],ys[1]))
        return np.min(zs)

@torch.no_grad()
def obtain_embeddings(config: AttributeHashmap):
    best_type = 'pred_psnr' # 'seg_dice'

    save_folder = '../visualize/'
    os.makedirs(save_folder, exist_ok=True)
    setting_str = config.save_folder.split('/')[-3]
    embedding_save_dir = save_folder + setting_str + '_embeddings.npz'
    info_save_dir = save_folder + setting_str + '_info.npz'
    figure_save_dir = save_folder + setting_str + '_phate.png'

    # Collect the embedding vectors
    subject_list = []
    time_arr = None

    if os.path.isfile(info_save_dir) and os.path.isfile(embedding_save_dir):
        npz_file = np.load(embedding_save_dir)
        embedding_list_by_connection = [npz_file[item] for item in npz_file.files]

        npz_file = np.load(info_save_dir)
        subject_arr = npz_file['subject_arr']
        time_arr = npz_file['time_arr']

    else:
        device = torch.device(
            'cuda:%d' % config.gpu_id if torch.cuda.is_available() else 'cpu')
        full_set, num_image_channel = \
            prepare_dataset_full_sequence(config=config)
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

        embedding_list_by_connection = None
        for iter_idx, (images, timestamps) in enumerate(tqdm(full_set)):
            x_list, t_arr = convert_variables(images, timestamps, device)

            subject_list.extend([iter_idx for _ in range(len(t_arr))])
            if time_arr is None:
                time_arr = t_arr.cpu().numpy()
            else:
                time_arr = np.concatenate((time_arr, t_arr.cpu().numpy()), axis=0)

            for t_idx in range(len(t_arr)):
                x_start = x_list[t_idx]
                t_start = t_arr[t_idx]

                embeddings_before, _ = \
                    model.return_embeddings(x=x_start, t=(t_arr[-1] - t_start).unsqueeze(0) * config.t_multiplier)
                embeddings_before = [item.cpu().numpy().reshape(1, -1) for item in embeddings_before]

                if embedding_list_by_connection is None:
                    embedding_list_by_connection = []
                    for i in range(len(embeddings_before)):
                        embedding_list_by_connection.append(embeddings_before[i])
                else:
                    for i in range(len(embeddings_before)):
                        embedding_list_by_connection[i] = np.concatenate(
                            (embedding_list_by_connection[i], embeddings_before[i]), axis=0)

        subject_arr = np.array(subject_list)

        with open(embedding_save_dir, 'wb+') as f:
            np.savez(
                f,
                *[embedding_list_by_connection[i] for i in range(len(embedding_list_by_connection))],
            )
        with open(info_save_dir, 'wb+') as f:
            np.savez(
                f,
                subject_arr=subject_arr,
                time_arr=time_arr,
            )

    # Visualize them in 2 dimensions.
    fig = plt.figure(figsize=(14, 6))
    plt.rcParams['font.family'] = 'serif'
    ax = fig.add_subplot(1, 2, 1)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.tick_params(axis='both', which='minor', labelsize=14)
    ax.set_xlabel('PHATE 1', fontsize=16)
    ax.set_ylabel('PHATE 2', fontsize=16)

    phate_op = phate.PHATE(random_state=0,
                           n_jobs=1,
                           n_components=2,
                           verbose=False)
    data_phate = phate_op.fit_transform(embedding_list_by_connection[0])
    ax.scatter(data_phate[:, 0], data_phate[:, 1], color='firebrick', alpha=1.0)
    ax.set_xticks([np.round(item, 2) for item in
                   np.linspace(data_phate[:, 0].min(), data_phate[:, 0].max(), num=5)])
    ax.set_yticks([np.round(item, 2) for item in
                   np.linspace(data_phate[:, 1].min(), data_phate[:, 1].max(), num=5)])

    # Draw arrows connecting the trajectories.
    # cmap = mpl.cm.viridis
    cmap = mpl.cm.winter
    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    time_scaling_factor = np.max(time_arr) - np.min(time_arr)

    for subject_id in np.unique(subject_arr):
        subject_embeddings = data_phate[subject_arr == subject_id]
        subject_times = time_arr[subject_arr == subject_id]

        # Assert monotonic increasing time.
        assert all(t1 < t2 for t1, t2 in zip(subject_times[:-1], subject_times[1:]))

        for e1, e2, t2 in zip(subject_embeddings[:-1], subject_embeddings[1:], subject_times[1:]):
            ax.arrow(e1[0], e1[1], e2[0]-e1[0], e2[1]-e1[1],
                     width=5e-5, head_length=3e-3, head_width=3e-3, fc='black',
                     ec=cmap(t2 / time_scaling_factor))

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.tick_params(axis='both', which='major', labelsize=14)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation='vertical')

    # Visualize them in 3 dimensions.
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.view_init(elev=30, azim=45, roll=0)
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    ax.set_xlabel('PHATE 1', fontsize=14)
    ax.set_ylabel('PHATE 2', fontsize=14)
    ax.set_zlabel('PHATE 3', fontsize=14)

    phate_op = phate.PHATE(random_state=0,
                           n_jobs=1,
                           n_components=3,
                           verbose=False)
    data_phate = phate_op.fit_transform(embedding_list_by_connection[0])
    ax.scatter(data_phate[:, 0], data_phate[:, 1], data_phate[:, 2],
               color='firebrick', s=8, alpha=1.0)
    ax.set_xticks([np.round(item, 2) for item in
                   np.linspace(data_phate[:, 0].min(), data_phate[:, 0].max(), num=5)])
    ax.set_yticks([np.round(item, 2) for item in
                   np.linspace(data_phate[:, 1].min(), data_phate[:, 1].max(), num=5)])
    ax.set_zticks([np.round(item, 2) for item in
                   np.linspace(data_phate[:, 2].min(), data_phate[:, 2].max(), num=5)])

    # Draw arrows connecting the trajectories.
    for subject_id in np.unique(subject_arr):
        subject_embeddings = data_phate[subject_arr == subject_id]
        subject_times = time_arr[subject_arr == subject_id]

        # Assert monotonic increasing time.
        assert all(t1 < t2 for t1, t2 in zip(subject_times[:-1], subject_times[1:]))

        for e1, e2, t2 in zip(subject_embeddings[:-1], subject_embeddings[1:], subject_times[1:]):
            a = Arrow3D([e1[0], e2[0]], [e1[1], e2[1]], [e1[2], e2[2]],
                        lw=2, mutation_scale=6, arrowstyle="-|>",
                        color=cmap(t2 / time_scaling_factor),
                        alpha=0.5)
            a.set_zorder(-1)
            ax.add_artist(a)

    fig.savefig(figure_save_dir, dpi=300)

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--run-count', default=None, type=int)

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
    parser.add_argument('--t-multiplier', default=0.2, type=float)  # only relevant to ODE
    parser.add_argument('--ode-location', default='all_connections', type=str)  # only relevant to ODE
    parser.add_argument('--depth', default=5, type=int)             # only relevant to simple unet
    parser.add_argument('--num-filters', default=64, type=int)      # only relevant to simple unet
    parser.add_argument('--diffusion-interval', default=100, type=int)      # only relevant to I2SB
    parser.add_argument('--num-workers', default=8, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--max-training-samples', default=2048, type=int)
    parser.add_argument('--n-plot-per-epoch', default=4, type=int)

    parser.add_argument('--no-l2', action='store_true')  # only relevant to ODE
    parser.add_argument('--coeff-smoothness', default=0, type=float)  # only relevant to ODE
    parser.add_argument('--coeff-latent', default=0, type=float)
    parser.add_argument('--coeff-contrastive', default=0, type=float)
    parser.add_argument('--pretrained-vision-model', default='convnext_tiny', type=str)

    args = vars(parser.parse_args())
    config = AttributeHashmap(args)
    config = parse_settings(config, log_settings=False, run_count=config.run_count)

    seed_everything(config.random_seed)

    obtain_embeddings(config=config)
