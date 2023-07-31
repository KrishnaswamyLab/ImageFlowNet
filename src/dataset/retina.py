import itertools
from glob import glob
from typing import List, Tuple, Literal

import cv2
import numpy as np
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from data_utils.extend import ExtendedDataset
from data_utils.split import split_indices


def build_dataset(opt, log, split: str, num_workers: int = 4):
    # Read dataset.
    if opt.dataset_name.name == 'retina':
        dataset = RetinaDataset(base_path=opt.dataset_dir,
                                target_dim=(opt.image_size, opt.image_size))
    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    num_image_channel = dataset.num_image_channel()

    # Load into DataLoader
    ratios = [float(c) for c in opt.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = split_indices(
        indices=indices, splits=ratios, random_seed=opt.seed)

    assert split in ['train', 'val', 'test'], \
        'build_dataset: split can only be one of `train`, `val`, or `test`.'

    if split == 'train':
        subset = RetinaSubset(retina_dataset=dataset,
                              subset_indices=train_indices,
                              return_format=opt.dataset_return_format)
        min_batch_per_epoch = 5
        desired_len = max(len(subset), min_batch_per_epoch)
        subset = ExtendedDataset(dataset=subset, desired_len=desired_len)

    elif split == 'val':
        subset = RetinaSubset(retina_dataset=dataset,
                              subset_indices=val_indices,
                              return_format=opt.dataset_return_format)

    elif split == 'test':
        subset = RetinaSubset(retina_dataset=dataset,
                              subset_indices=test_indices,
                              return_format=opt.dataset_return_format)

    # if split == 'train':
    #     subset_loader = DataLoader(dataset=subset,
    #                                batch_size=1,
    #                                shuffle=True,
    #                                num_workers=num_workers,
    #                                pin_memory=True,
    #                                drop_last=True)

    # else:
    #     subset_loader = DataLoader(dataset=subset,
    #                                batch_size=1,
    #                                shuffle=False,
    #                                num_workers=num_workers,
    #                                pin_memory=True,
    #                                drop_last=True)

    log.info(
        f"[Dataset] Built Retina dataset ({split} set), size={len(subset)}!")

    # return subset_loader, num_image_channel
    return subset


class RetinaDataset(Dataset):

    def __init__(self,
                 base_path: str = '../../data/',
                 image_folder: str = 'AREDS_2014_images_512x512/',
                 target_dim: Tuple[int] = (512, 512),
                 sample_pairs: bool = False):
        '''
        Information regarding the dataset.

        Files are named in the following format:
            `ID visitNo F2 laterality.jpg`

            ID: patient identification (5 digits)
            visitNo: 00, 02, 04, etc (each visit represents 6 months interval)
            laterality: LE LS = left eye; RE RS = right eye.

        The special thing here is that different patients may have different number of visits.
        - If a patient has fewer than 2 visits, we ignore the patient.
        - When a patient's index is queried, we return images from all visits of that patient.
        - We need to be extra cautious that the data is split on the patient level rather than image pair level.

        NOTE: since different patients may have different number of visits, the returned array will
        not necessarily be of the same shape. Due to the concatenation requirements, we can only
        set batch size to 1 in the downstream Dataloader.
        '''
        super(RetinaDataset, self).__init__()

        self.target_dim = target_dim
        self.sample_pairs = sample_pairs
        all_image_folders = sorted(glob('%s/%s/*/' %
                                        (base_path, image_folder)))

        self.image_by_patient = []

        for folder in all_image_folders:
            paths = sorted(glob('%s/*.jpg' % (folder)))
            if len(paths) >= 2:
                self.image_by_patient.append(paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 3


class RetinaSubset(RetinaDataset):

    def __init__(self,
                 retina_dataset: RetinaDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs',
                                              'array']):
        '''
        A subset of RetinaDataset.

        In RetinaDataset, we carefully isolated the (variable number of) images from
        different patients, and in train/val/test split we split the data by
        patient rather than by image.

        Now we have 3 instances of RetinaSubset, one for each train/val/test set.
        In each set, we can safely unpack the images out.
        We want to organize the images such that each time `__getitem__` is called,
        it gets a pair of [x_start, x_end] and [t_start, t_end].
        '''
        super(RetinaSubset, self).__init__()

        self.target_dim = retina_dataset.target_dim
        self.return_format = return_format

        self.image_by_patient = [
            retina_dataset.image_by_patient[i] for i in subset_indices
        ]

        self.all_image_pairs = []
        for image_list in self.image_by_patient:
            pair_indices = list(
                itertools.combinations(np.arange(len(image_list)), r=2))
            for (idx1, idx2) in pair_indices:
                self.all_image_pairs.append(
                    [image_list[idx1], image_list[idx2]])

    def __len__(self) -> int:
        if self.return_format == 'one_pair':
            # If we only return 1 pair of images per patient...
            return len(self.image_by_patient)
        elif self.return_format == 'all_pairs':
            # If we return all pairs of images per patient...
            return len(self.all_image_pairs)
        elif self.return_format == 'array':
            # If we return all images as an array per patient...
            return len(self.image_by_patient)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if self.return_format == 'one_pair':
            image_list = self.image_by_patient[idx]
            pair_indices = list(
                itertools.combinations(np.arange(len(image_list)), r=2))
            sampled_pair = [
                image_list[i]
                for i in pair_indices[np.random.choice(len(pair_indices))]
            ]
            images = np.array([
                load_image(p, target_dim=self.target_dim) for p in sampled_pair
            ])
            timestamps = np.array([get_time(p) for p in sampled_pair])

        elif self.return_format == 'all_pairs':
            queried_pair = self.all_image_pairs[idx]
            images = np.array([
                load_image(p, target_dim=self.target_dim) for p in queried_pair
            ])
            timestamps = np.array([get_time(p) for p in queried_pair])

        elif self.return_format == 'array':
            queried_patient = self.image_by_patient[idx]
            images = np.array([
                load_image(p, target_dim=self.target_dim)
                for p in queried_patient
            ])
            timestamps = np.array([get_time(p) for p in queried_patient])

        return images, timestamps


def load_image(path: str, target_dim: Tuple[int] = None) -> np.array:
    ''' Load image as numpy array from a path string.'''
    if target_dim is not None:
        image = np.array(
            cv2.resize(
                cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                             code=cv2.COLOR_BGR2RGB), target_dim))
    else:
        image = np.array(
            cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                         code=cv2.COLOR_BGR2RGB))

    # Normalize image.
    image = (image / 255 * 2) - 1

    # Channel last to channel first to comply with Torch.
    image = np.moveaxis(image, -1, 0)

    return image


def get_time(path: str) -> float:
    ''' Get the timestamp information from a path string. '''
    time = path.split()[1]
    # Shall be 2 or 3 digits
    assert len(time) in [2, 3]
    time = float(time)
    return time
