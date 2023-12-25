import argparse
import os

import monai
import albumentations as A
import numpy as np
import torch
import yaml
from data_utils.prepare_dataset import prepare_dataset_segmentation
from tqdm import tqdm
from utils.attribute_hashmap import AttributeHashmap
from utils.early_stop import EarlyStopping
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
    early_stopper = EarlyStopping(mode='min',
                                  patience=config.patience,
                                  percentage=False)

    loss_fn = torch.nn.BCELoss()
    best_val_loss = np.inf

    for epoch_idx in tqdm(range(config.max_epochs)):
        train_loss = 0
        train_metrics = {
            'dice': [],
            'hausdorff': [],
        }

        model.train()
        for _, (x_train, seg_true) in enumerate(tqdm(train_set)):
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

        if early_stopper.step(val_loss):
            # If the validation loss stop decreasing, stop training.
            log('Early stopping criterion met. Ending training.',
                filepath=config.log_dir,
                to_console=True)
            break
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='`train` or `test`?', required=True)
    parser.add_argument('--config',
                        help='Path to config yaml file.',
                        required=True)
    args = vars(parser.parse_args())

    args = AttributeHashmap(args)
    config = AttributeHashmap(yaml.safe_load(open(args.config)))
    config.config_file_name = args.config
    config = parse_settings(config, log_settings=args.mode == 'train')

    assert args.mode in ['train', 'test']

    seed_everything(config.random_seed)

    if args.mode == 'train':
        train(config=config)
        test(config=config)
    elif args.mode == 'test':
        test(config=config)
