'''
Work in progress
'''


import itertools
from typing import Literal
from glob import glob
from typing import List, Tuple

import os
import cv2
import numpy as np
from torch.utils.data import Dataset


def load_image(path: str, target_dim: Tuple[int] = None, normalize: bool = True) -> np.array:
    ''' Load image as numpy array from a path string.'''
    if target_dim is not None:
        image = np.array(
            cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), target_dim))
    else:
        image = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    # Normalize image.
    if normalize:
        image = (image / 255 * 2) - 1

    # Add the channel dimension to comply with Torch.
    image = image[None, :, :]

    return image


def get_time(path: str) -> float:
    ''' Get the timestamp information from a path string. '''
    time = os.path.basename(path).split('_')[2]
    # Shall be 2 or 3 digits
    assert len(time) in [2, 3]
    time = float(time)
    return time


class RetinaUCSFDataset(Dataset):

    def __init__(self,
                 base_path: str = '../../data/retina_ucsf/',
                 image_folder: str = 'UCSF_images_final_512x512/',
                 target_dim: Tuple[int] = (512, 512)):
        '''
        The special thing here is that different patients may have different number of visits.
        - If a patient has fewer than 2 visits, we ignore the patient.
        - When a patient's index is queried, we return images from all visits of that patient.
        - We need to be extra cautious that the data is split on the patient level rather than image pair level.

        NOTE: since different patients may have different number of visits, the returned array will
        not necessarily be of the same shape. Due to the concatenation requirements, we can only
        set batch size to 1 in the downstream Dataloader.
        '''
        super().__init__()

        self.target_dim = target_dim
        all_image_folders = sorted(glob('%s/%s/*/' %
                                        (base_path, image_folder)))

        self.image_by_patient = []

        for folder in all_image_folders:
            paths = sorted(glob('%s/*.png' % (folder)))
            if len(paths) >= 2:
                self.image_by_patient.append(paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class RetinaUCSFSubset(RetinaUCSFDataset):

    def __init__(self,
                 main_dataset: RetinaUCSFDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs'],
                 pos_neg_pairs: bool = False):
        '''
        A subset of RetinaUCSFDataset.

        In RetinaUCSFDataset, we carefully isolated the (variable number of) images from
        different patients, and in train/val/test split we split the data by
        patient rather than by image.

        Now we have 3 instances of RetinaSubset, one for each train/val/test set.
        In each set, we can safely unpack the images out.
        We want to organize the images such that each time `__getitem__` is called,
        it gets a pair of [x_start, x_end] and [t_start, t_end].

        pos_neg_pairs is a dummy input argument.
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

        return images, timestamps


class RetinaUCSFSegDataset(Dataset):

    def __init__(self,
                 base_path: str = '../../data/retina_ucsf/',
                 image_folder: str = 'UCSF_images_final_512x512/',
                 mask_folder: str = 'UCSF_masks_final_512x512/',
                 target_dim: Tuple[int] = (512, 512)):
        '''
        This dataset is for segmentation.
        '''
        super().__init__()

        self.target_dim = target_dim
        all_image_folders = sorted(glob('%s/%s/*/' % (base_path, image_folder)))

        self.image_by_patient = []
        self.mask_by_patient = []
        for im_folder in all_image_folders:
            image_paths = sorted(glob('%s/*.png' % im_folder))
            mask_paths = []
            for image_path_ in image_paths:
                mask_path_ = '_'.join(image_path_.split('_')[:-1]).replace(
                    image_folder, mask_folder) + '_GA_mask.png'
                assert os.path.isfile(mask_path_)
                mask_paths.append(mask_path_)
            self.image_by_patient.append(image_paths)
            self.mask_by_patient.append(mask_paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class RetinaUCSFSegSubset(RetinaUCSFSegDataset):

    def __init__(self,
                 main_dataset: RetinaUCSFDataset = None,
                 subset_indices: List[int] = None,
                 transforms = None):
        '''
        A subset of RetinaUCSFSegDataset.
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim

        image_by_patient = [
            main_dataset.image_by_patient[i] for i in subset_indices
        ]
        mask_by_patient = [
            main_dataset.mask_by_patient[i] for i in subset_indices
        ]

        self.image_list = [image for patient_folder in image_by_patient for image in patient_folder]
        self.mask_list = [mask for patient_folder in mask_by_patient for mask in patient_folder]
        assert len(self.image_list) == len(self.mask_list)

        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_list)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image = load_image(self.image_list[idx], target_dim=self.target_dim, normalize=False)
        mask = load_image(self.mask_list[idx], target_dim=self.target_dim, normalize=False)

        if self.transforms is not None:
            transformed = self.transforms(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]

        image = image / 255 * 2 - 1
        mask = mask > 128

        return image, mask