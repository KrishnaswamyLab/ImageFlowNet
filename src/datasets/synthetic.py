import itertools
import os
from typing import Literal
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class SyntheticDataset(Dataset):

    def __init__(self,
                 base_path: str = '../../data/synthesized/',
                 image_folder: str = 'base/',
                 target_dim: Tuple[int] = (256, 256)):
        '''
        NOTE: since different patients may have different number of visits, the returned array will
        not necessarily be of the same shape. Due to the concatenation requirements, we can only
        set batch size to 1 in the downstream Dataloader.
        '''
        super().__init__()

        self.target_dim = target_dim
        all_image_folders = sorted(glob('%s/%s/*/' % (base_path, image_folder)))

        self.image_by_patient = []

        for folder in all_image_folders:
            paths = sorted(glob('%s/*.png' % (folder)))
            if len(paths) >= 2:
                self.image_by_patient.append(paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 3


class SyntheticSubset(SyntheticDataset):

    def __init__(self,
                 main_dataset: SyntheticDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs',
                                              'array']):
        '''
        A subset of SyntheticDataset.

        In SyntheticDataset, we carefully isolated the (variable number of) images from
        different patients, and in train/val/test split we split the data by
        patient rather than by image.

        Now we have 3 instances of SyntheticSubset, one for each train/val/test set.
        In each set, we can safely unpack the images out.
        We want to organize the images such that each time `__getitem__` is called,
        it gets a pair of [x_start, x_end] and [t_start, t_end].
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format

        self.image_by_patient = [
            main_dataset.image_by_patient[i] for i in subset_indices
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
    time = path.split('time_')[1].replace('.png', '')
    # Shall be 3 digits
    assert len(time) == 3
    time = float(time)
    return time
