import itertools
from typing import Literal
from glob import glob
from typing import List, Tuple

import os
import cv2
import numpy as np
from torch.utils.data import Dataset

root_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])


class RetinaAREDSDataset(Dataset):

    def __init__(self,
                 base_path: str = root_dir + '/data/retina_areds/',
                 image_folder: str = 'AREDS_2014_images_512x512/',
                 eye_mask_folder: str = None,
                 target_dim: Tuple[int] = (512, 512)):
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
        super().__init__()

        self.target_dim = target_dim
        all_image_folders = sorted(glob('%s/%s/*/' %
                                        (base_path, image_folder)))
        self.eye_mask_folder = eye_mask_folder

        if self.eye_mask_folder is not None:
            all_eye_mask_folders = sorted(glob('%s/%s/*/' %
                                               (base_path, eye_mask_folder)))

        self.image_by_patient = []
        if self.eye_mask_folder is None:
            for image_folder in all_image_folders:
                image_paths = sorted(glob('%s/*.jpg' % (image_folder)))
                if len(image_paths) >= 2:
                    self.image_by_patient.append(image_paths)
        else:
            self.eye_mask_by_patient = []
            for image_folder, eye_mask_folder in zip(all_image_folders, all_eye_mask_folders):
                assert image_folder.split('/')[-1] == eye_mask_folder.split('/')[-1]
                image_paths = sorted(glob('%s/*.jpg' % (image_folder)))
                eye_mask_paths = sorted(glob('%s/*.jpg' % (eye_mask_folder)))
                assert len(image_paths) == len(eye_mask_paths)
                if len(image_paths) >= 2:
                    self.image_by_patient.append(image_paths)
                    self.eye_mask_by_patient.append(eye_mask_paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 3


class RetinaAREDSSubset(RetinaAREDSDataset):

    def __init__(self,
                 main_dataset: RetinaAREDSDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs'],
                 pos_neg_pairs: bool = False,
                 time_close_thr: int = 4,
                 time_far_thr: int = 12):
        '''
        A subset of RetinaAREDSDataset.

        In RetinaAREDSDataset, we carefully isolated the (variable number of) images from
        different patients, and in train/val/test split we split the data by
        patient rather than by image.

        Now we have 3 instances of RetinaSubset, one for each train/val/test set.
        In each set, we can safely unpack the images out.
        We want to organize the images such that each time `__getitem__` is called,
        it gets a pair of [x_start, x_end] and [t_start, t_end].
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format
        self.pos_neg_pairs = pos_neg_pairs
        self.time_close_thr = time_close_thr
        self.time_far_thr = time_far_thr
        self.mask_common_area = main_dataset.eye_mask_folder is not None

        self.image_by_patient = [
            main_dataset.image_by_patient[i] for i in subset_indices
        ]

        self.patient_idx_list = []
        self.all_image_pairs = []
        for _patient_idx, image_list in enumerate(self.image_by_patient):
            pair_indices = list(
                itertools.combinations(np.arange(len(image_list)), r=2))
            for (idx1, idx2) in pair_indices:
                self.all_image_pairs.append(
                    [image_list[idx1], image_list[idx2]])
                self.patient_idx_list.append(_patient_idx)

        if self.mask_common_area:
            self.eye_mask_by_patient = [
                main_dataset.eye_mask_by_patient[i] for i in subset_indices
            ]
            self.all_eye_mask_pairs = []
            for _patient_idx, eye_mask_list in enumerate(self.eye_mask_by_patient):
                pair_indices = list(
                    itertools.combinations(np.arange(len(eye_mask_list)), r=2))
                for (idx1, idx2) in pair_indices:
                    self.all_eye_mask_pairs.append(
                        [eye_mask_list[idx1], eye_mask_list[idx2]])

    def __len__(self) -> int:
        if self.return_format == 'one_pair':
            # If we only return 1 pair of images per patient...
            return len(self.image_by_patient)
        elif self.return_format == 'all_pairs':
            # If we return all pairs of images per patient...
            return len(self.all_image_pairs)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        '''
        If `self.pos_neg_pairs` is True:
            These cases are treated as "positive pairs".
            1. The image from the same patient, from nearby timestamps.
            These cases are treated as "negative pairs".
            1. The image from a different patient.
            2. The image from the same patient, from far-apart timestamps.
        '''
        if self.return_format == 'one_pair':
            image_list = self.image_by_patient[idx]
            pair_paths = list(
                itertools.combinations(np.arange(len(image_list)), r=2))
            rand_idx = np.random.choice(len(pair_paths))
            sampled_pair = [
                image_list[i]
                for i in pair_paths[rand_idx]
            ]
            images = np.array([
                load_image(p, target_dim=self.target_dim) for p in sampled_pair
            ])
            timestamps = np.array([get_time(p) for p in sampled_pair])
            if self.mask_common_area:
                eye_mask_list = self.eye_mask_by_patient[idx]
                eye_mask_pair = [
                    eye_mask_list[i]
                    for i in pair_paths[rand_idx]
                ]
                eye_masks = np.array([
                    load_mask(p, target_dim=self.target_dim) for p in eye_mask_pair
                ])
                assert len(eye_masks) == 2
                dice_coeff = DICE(eye_masks[0], eye_masks[1])

        elif self.return_format == 'all_pairs':
            queried_pair = self.all_image_pairs[idx]
            images = np.array([
                load_image(p, target_dim=self.target_dim) for p in queried_pair
            ])
            timestamps = np.array([get_time(p) for p in queried_pair])

            if self.mask_common_area:
                queried_eye_mask_pair = self.all_eye_mask_pairs[idx]
                eye_masks = np.array([
                    load_mask(p, target_dim=self.target_dim) for p in queried_eye_mask_pair
                ])
                assert len(eye_masks) == 2
                dice_coeff = DICE(eye_masks[0], eye_masks[1])

        if not self.pos_neg_pairs:
            if self.mask_common_area:
                # Apply the eye masks.
                if images.shape[1] != 1:
                    eye_masks = np.repeat(eye_masks, images.shape[1], axis=1)
                images[0][~eye_masks[0]] = images.min()
                images[0][~eye_masks[1]] = images.min()
                images[1][~eye_masks[0]] = images.min()
                images[1][~eye_masks[1]] = images.min()
                return images, timestamps, dice_coeff
            else:
                return images, timestamps

        #TODO: Have not enabled self.mask_common_area along with self.pos_neg_pairs!

        if self.return_format == 'one_pair':
            curr_patient_idx = idx
        elif self.return_format == 'all_pairs':
            curr_patient_idx = self.patient_idx_list[idx]

        all_times = np.array([get_time(i) for i in self.image_by_patient[curr_patient_idx]])
        time_diff = np.diff(all_times)
        if min(time_diff) <= self.time_close_thr:
            # Select from the close-enough time pairs.
            valid_locs = np.where(time_diff <= self.time_close_thr)[0]
            selected_loc = np.random.choice(valid_locs)
            time_pair_close_loc = [selected_loc, selected_loc + 1]
        else:
            # Select the closest time pair.
            selected_loc = time_diff.argmin()
            time_pair_close_loc = [selected_loc, selected_loc + 1]

        pair_paths = [self.image_by_patient[curr_patient_idx][i] for i in time_pair_close_loc]
        pos_pair = np.array([
            load_image(p, target_dim=self.target_dim) for p in pair_paths
        ])

        if min(time_diff) >= self.time_far_thr:
            # Select from the far-enough time pairs.
            valid_locs = np.where(time_diff >= self.time_far_thr)[0]
            selected_loc = np.random.choice(valid_locs)
            time_pair_far_loc = [selected_loc, selected_loc + 1]
        else:
            # Select the farthest time pair.
            time_pair_far_loc = [0, -1]

        pair_paths = [self.image_by_patient[curr_patient_idx][i] for i in time_pair_far_loc]
        neg_pair1 = np.array([
            load_image(p, target_dim=self.target_dim) for p in pair_paths
        ])

        other_patient_idx = np.random.choice(
            np.concatenate((np.arange(curr_patient_idx), np.arange(curr_patient_idx + 1, len(self.image_by_patient)))))
        sampled_image_self = self.image_by_patient[curr_patient_idx][
            np.random.choice(np.arange(len(self.image_by_patient[curr_patient_idx])))]
        sampled_image_other = self.image_by_patient[other_patient_idx][
            np.random.choice(np.arange(len(self.image_by_patient[other_patient_idx])))]

        pair_paths = [sampled_image_self, sampled_image_other]
        neg_pair2 = np.array([
            load_image(p, target_dim=self.target_dim) for p in pair_paths
        ])

        return images, timestamps, pos_pair, neg_pair1, neg_pair2


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
    image = (image / 255) * 2 - 1

    # Channel last to channel first to comply with Torch.
    image = np.moveaxis(image, -1, 0)

    return image

def load_mask(path: str, target_dim: Tuple[int] = None) -> np.array:
    ''' Load binary mask as numpy array from a path string.'''
    if target_dim is not None:
        mask = np.array(
            cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), target_dim))
    else:
        mask = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    # Normalize image.
    assert mask.max() in [0, 255]
    mask = mask > 128

    # Channel first to comply with Torch.
    mask = mask[None, ...]

    return mask

def get_time(path: str) -> float:
    ''' Get the timestamp information from a path string. '''
    time = path.split()[1]
    # Shall be 2 or 3 digits
    assert len(time) in [2, 3]
    time = float(time)
    return time


def DICE(mask1: np.array, mask2: np.array) -> float:
    '''
    Dice Coefficient between 2 binary masks.
    '''

    if isinstance(mask1.min(), bool):
        mask1 = np.uint8(mask1)
    if isinstance(mask2.min(), bool):
        mask2 = np.uint8(mask2)

    assert mask1.min() in [0, 1] and mask2.min() in [0, 1], \
        'min values for masks are not in [0, 1]: mask1: %s, mask2: %s' % (mask1.min(), mask2.min())
    assert mask1.max() == 1 and mask2.max() == 1, \
        'max values for masks are not 1: mask1: %s, mask2: %s' % (mask1.max(), mask2.max())

    assert mask1.shape == mask2.shape, \
        'mask shapes do not match: %s vs %s' % (mask1.shape, mask2.shape)

    intersection = np.logical_and(mask1, mask2).sum()
    denom = np.sum(mask1) + np.sum(mask2)
    epsilon = 1e-9

    dice = 2 * intersection / (denom + epsilon)

    return dice