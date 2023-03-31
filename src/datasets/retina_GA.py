import itertools
import os
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset


class RetinaGA(Dataset):

    def __init__(self,
                 base_path: str = '../../data/retina_GA',
                 image_folder: str = 'AREDS_2014_images/',
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
        super(RetinaGA, self).__init__()

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


class RetinaGASubset(RetinaGA):

    def __init__(self,
                 retina_GA_dataset: RetinaGA = None,
                 subset_indices: List[int] = None,
                 sample_pairs: bool = False):
        '''
        A subset of RetinaGA Dataset.

        In RetinaGA Dataset, we carefully isolated the (variable number of) images from
        different patients, and in train/val/test split we split the data by
        patient rather than by image.

        Now we have 3 instances of RetinaGASubset, one for each train/val/test set.
        In each set, we can safely unpack the images out.
        We want to organize the images such that each time `__getitem__` is called,
        it gets a pair of [x_start, x_end] and [t_start, t_end].
        '''
        super(RetinaGASubset, self).__init__()

        self.target_dim = retina_GA_dataset.target_dim
        self.sample_pairs = sample_pairs

        self.image_by_patient = [
            retina_GA_dataset.image_by_patient[i] for i in subset_indices
        ]

        self.all_image_pairs = []
        for image_list in self.image_by_patient:
            pair_indices = list(
                itertools.combinations(np.arange(len(image_list)), r=2))
            for (idx1, idx2) in pair_indices:
                self.all_image_pairs.append(
                    [image_list[idx1], image_list[idx2]])

    def __len__(self) -> int:
        if self.sample_pairs:
            # If we only sample 1 pair of images per patient...
            return len(self.image_by_patient)
        else:
            # If we unpack all the images...
            return len(self.all_image_pairs)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if self.sample_pairs is True:
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

        else:
            queried_pair = self.all_image_pairs[idx]
            images = np.array([
                load_image(p, target_dim=self.target_dim) for p in queried_pair
            ])
            timestamps = np.array([get_time(p) for p in queried_pair])

        return images, timestamps


def load_image(path: str, target_dim: Tuple[int] = (256, 256)) -> np.array:
    ''' Load image as numpy array from a path string.'''
    image = np.array(
        cv2.resize(
            cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR),
                            code=cv2.COLOR_BGR2RGB), target_dim))
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
