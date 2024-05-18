import argparse
import ast
import os

import monai
import albumentations as A
import numpy as np
import torch
from data_utils.prepare_dataset import prepare_dataset_segmentation
from tqdm import tqdm
from utils.attribute_hashmap import AttributeHashmap
from utils.log_util import log
from utils.metrics import dice_coeff, hausdorff
from utils.parse import parse_settings
from utils.seed import seed_everything


def save_weights(model_save_path: str, model):
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    torch.save(model.state_dict(), model_save_path)
    return


def load_weights(model_save_path: str, model, device: torch.device):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    return


def train(config: AttributeHashmap):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform = A.Compose(
        [
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        ]
    )
    transforms_list = [train_transform, None, None]

    train_set, val_set, _, num_image_channel = \
        prepare_dataset_segmentation(config=config, transforms_list=transforms_list)

    # Build the model
    model = torch.nn.Sequential(
        monai.networks.nets.DynUNet(
            spatial_dims=2,
            in_channels=num_image_channel,
            out_channels=1,
            kernel_size=[5, 5, 5, 5],
            filters=[16, 32, 64, 128],
            strides=[1, 1, 1, 1],
            upsample_kernel_size=[1, 1, 1, 1]),
        torch.nn.Sigmoid()).to(device)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.learning_rate)

    loss_fn = torch.nn.BCELoss()
    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss = 0
        train_metrics = {
            'dice': [],
            'hausdorff': [],
        }

        model.train()
        for iter_idx, (x_train, seg_true) in enumerate(tqdm(train_set)):
            if 'max_training_samples' in config:
                if iter_idx * config.batch_size > config.max_training_samples:
                    break

            x_train = x_train.float().to(device)
            seg_pred = model(x_train)
            seg_pred = seg_pred.squeeze(1).float().to(device)
            if len(seg_true.shape) == 4:
                seg_true = seg_true.squeeze(1)
            seg_true = seg_true.float().to(device)

            loss = loss_fn(seg_pred, seg_true)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            seg_true = seg_true.cpu().detach().numpy()
            seg_pred = (seg_pred > 0.5).cpu().detach().numpy()

            for batch_idx in range(seg_true.shape[0]):
                train_metrics['dice'].append(
                    dice_coeff(
                        label_pred=seg_pred[batch_idx, ...],
                        label_true=seg_true[batch_idx, ...]))
                train_metrics['hausdorff'].append(
                    hausdorff(label_pred=seg_pred[batch_idx, ...],
                              label_true=seg_true[batch_idx, ...]))

        train_loss = train_loss / len(train_set)

        log('Train [%s/%s] loss: %.3f, dice: %.3f \u00B1 %.3f, hausdorff:  %.3f \u00B1 %.3f.'
            %
            (epoch_idx, config.max_epochs, train_loss,
             np.mean(train_metrics['dice']), np.std(train_metrics['dice']) /
             np.sqrt(len(train_metrics['dice'])),
             np.mean(train_metrics['hausdorff']),
             np.std(train_metrics['hausdorff']) /
             np.sqrt(len(train_metrics['hausdorff']))),
            filepath=config.log_dir,
            to_console=False)

        val_loss = 0
        model.eval()
        val_metrics = {
            'dice': [],
            'hausdorff': [],
        }
        with torch.no_grad():
            for _, (x_val, seg_true) in enumerate(val_set):
                x_val = x_val.float().to(device)

                seg_pred = model(x_val)
                seg_pred = seg_pred.squeeze(1).float().to(device)
                if len(seg_true.shape) == 4:
                    seg_true = seg_true.squeeze(1)
                seg_true = seg_true.float().to(device)

                loss = loss_fn(seg_pred, seg_true)

                val_loss += loss.item()

                seg_true = seg_true.cpu().detach().numpy()
                seg_pred = (seg_pred > 0.5).cpu().detach().numpy()

                for batch_idx in range(seg_true.shape[0]):
                    val_metrics['dice'].append(
                        dice_coeff(label_pred=seg_pred,
                                   label_true=seg_true))
                    val_metrics['hausdorff'].append(
                        hausdorff(label_pred=seg_pred[batch_idx, ...],
                                  label_true=seg_true[batch_idx, ...]))

        val_loss = val_loss / len(val_set)

        log('Validation [%s/%s] loss: %.3f, dice: %.3f \u00B1 %.3f, hausdorff: %.3f \u00B1 %.3f.'
            %
            (epoch_idx, config.max_epochs, val_loss,
             np.mean(val_metrics['dice']),
             np.std(val_metrics['dice']) / np.sqrt(len(val_metrics['dice'])),
             np.mean(val_metrics['hausdorff']),
             np.std(val_metrics['hausdorff']) / np.sqrt(
                 len(val_metrics['hausdorff']))),
            filepath=config.log_dir,
            to_console=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_weights(config.model_save_path, model)
            log('Model weights successfully saved.',
                filepath=config.log_dir,
                to_console=False)

    return


def test(config: AttributeHashmap):
    device = torch.device('cpu')
    _, _, test_set, num_image_channel = \
        prepare_dataset_segmentation(config=config)

    # Build the model
    model = torch.nn.Sequential(
        monai.networks.nets.DynUNet(
            spatial_dims=2,
            in_channels=num_image_channel,
            out_channels=1,
            kernel_size=[5, 5, 5, 5],
            filters=[16, 32, 64, 128],
            strides=[1, 1, 1, 1],
            upsample_kernel_size=[1, 1, 1, 1]),
        torch.nn.Sigmoid()).to(device)

    load_weights(config.model_save_path, model, device=device)
    log('Model weights successfully loaded.',
        to_console=True)

    loss_fn = torch.nn.BCELoss()

    test_loss = 0
    test_metrics = {
        'dice': [],
        'hausdorff': [],
    }
    model.eval()

    with torch.no_grad():
        for _, (x_test, seg_true) in enumerate(test_set):
            x_test = x_test.float().to(device)
            seg_pred = model(x_test)
            seg_pred = seg_pred.squeeze(1).type(
                torch.FloatTensor).to(device)
            if len(seg_true.shape) == 4:
                seg_true = seg_true.squeeze(1)
            seg_true = seg_true.float().to(device)

            loss = loss_fn(seg_pred, seg_true)

            seg_true = seg_true.cpu().detach().numpy()
            seg_pred = (seg_pred > 0.5).cpu().detach().numpy()

            for batch_idx in range(seg_true.shape[0]):
                test_metrics['dice'].append(
                    dice_coeff(
                        label_pred=seg_pred[batch_idx, ...],
                        label_true=seg_true[batch_idx, ...]))
                test_metrics['hausdorff'].append(
                    hausdorff(
                        label_pred=seg_pred[batch_idx, ...],
                        label_true=seg_true[batch_idx, ...]))

            test_loss += loss.item()

    test_loss = test_loss / len(test_set)

    log('Test loss: %.3f, dice: %.3f \u00B1 %.3f, hausdorff: %.3f \u00B1 %.3f.'
        % (test_loss, np.mean(test_metrics['dice']),
           np.std(test_metrics['dice']) / np.sqrt(len(test_metrics['dice'])),
           np.mean(test_metrics['hausdorff']),
           np.std(test_metrics['hausdorff']) / np.sqrt(
               len(test_metrics['hausdorff']))),
        filepath=config.log_dir,
        to_console=True)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entry point.')
    parser.add_argument('--mode', help='`train` or `test`?', default='train')
    parser.add_argument('--gpu-id', help='Index of GPU device', default=0)
    parser.add_argument('--run-count', default=None, type=int)

    parser.add_argument('--dataset-name', default='retina_ucsf', type=str)
    parser.add_argument('--target-dim', default='(256, 256)', type=ast.literal_eval)
    # parser.add_argument('--image-folder', default='UCSF_images_final_512x512', type=str)
    # parser.add_argument('--mask-folder', default='UCSF_masks_final_512x512', type=str)
    # parser.add_argument('--dataset-path', default='$ROOT/data/retina_ucsf/', type=str)
    parser.add_argument('--segmentor-ckpt', default='$ROOT/checkpoints/segment_retinaUCSF_seed1.pty', type=str)

    parser.add_argument('--random-seed', default=1, type=int)
    parser.add_argument('--learning-rate', default=1e-3, type=float)
    parser.add_argument('--max-epochs', default=120, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-workers', default=4, type=int)
    parser.add_argument('--train-val-test-ratio', default='6:2:2', type=str)
    parser.add_argument('--max-training-samples', default=1024, type=int)  # this also extends dataset

    args = vars(parser.parse_args())
    config = AttributeHashmap(args)
    config = parse_settings(config, segmentor=True, log_settings=config.mode == 'train', run_count=config.run_count)

    assert config.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if config.mode == 'train':
        train(config=config)
        test(config=config)
    elif config.mode == 'test':
        test(config=config)
