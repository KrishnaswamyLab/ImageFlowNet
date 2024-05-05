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


def normalize_image(image: np.array) -> np.array:
    '''
    Normalize an image by z-score (zero mean, 0.5 variance), then clipped to [-1, 1].
    '''
    voxel_ndarray = image.copy()
    voxel_ndarray = voxel_ndarray.flatten()
    upper_bound = np.percentile(voxel_ndarray, 99.5)
    lower_bound = np.percentile(voxel_ndarray, 00.5)
    image = np.clip(image, lower_bound, upper_bound)
    image = (image - image.mean()) / (max(2 * image.std(), 1e-8))
    image = np.clip(image, -1.0, 1.0)
    return image


def load_image(path: str, target_dim: Tuple[int] = None, normalize: bool = True) -> np.array:
    ''' Load image as numpy array from a path string.'''
    if target_dim is not None:
        image = np.array(
            cv2.resize(cv2.imread(path, cv2.IMREAD_GRAYSCALE), target_dim))
    else:
        image = np.array(cv2.imread(path, cv2.IMREAD_GRAYSCALE))

    # Normalize image.
    if normalize:
        image = normalize_image(image)

    return image

def add_channel_dim(array: np.array) -> np.array:
    assert len(array.shape) == 2
    # Add the channel dimension to comply with Torch.
    array = array[None, :, :]
    return array

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
        self.all_image_folders = sorted(glob('%s/%s/*/' %
                                        (base_path, image_folder)))

        self.image_by_patient = []

        self.max_t = 0
        for folder in self.all_image_folders:
            paths = sorted(glob('%s/*.png' % (folder)))
            if len(paths) >= 2:
                self.image_by_patient.append(paths)
            for p in paths:
                self.max_t = max(self.max_t, get_time(p))

    def return_statistics(self) -> None:
        # NOTE: "patient" in the context outside means "eye" here.
        # The following printout is using a more precise terminology.
        unique_eye_list = np.unique([os.path.basename(item.rstrip('/')) for item in self.all_image_folders])
        unique_patient_list = np.unique([item.replace('_LE', '').replace('_RE', '') for item in unique_eye_list])
        print('Number of unique patients:', len(unique_patient_list))
        print('Number of unique eyes:', len(unique_eye_list))

        num_visit_map = {}
        for item in self.image_by_patient:
            num_visit = len(item)
            if num_visit not in num_visit_map.keys():
                num_visit_map[num_visit] = 1
            else:
                num_visit_map[num_visit] += 1
        for k, v in sorted(num_visit_map.items()):
            print('%d visits: %d eyes.' % (k, v))
        return

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class RetinaUCSFSubset(RetinaUCSFDataset):

    def __init__(self,
                 main_dataset: RetinaUCSFDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs', 'all_subsequences', 'full_sequence'],
                 transforms = None,
                #  min_time_diff: int = 6,
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

        pos_neg_pairs is a dummy input argument for backward compatibility.
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format
        self.transforms = transforms
        # self.min_time_diff = min_time_diff

        self.image_by_patient = [
            main_dataset.image_by_patient[i] for i in subset_indices
        ]

        self.all_image_pairs = []
        self.all_subsequences = []
        for image_list in self.image_by_patient:
            pair_indices = list(itertools.combinations(np.arange(len(image_list)), r=2))
            for (idx1, idx2) in pair_indices:
                self.all_image_pairs.append(
                    [image_list[idx1], image_list[idx2]])

            for num_items in range(2, len(image_list)+1):
                subsequence_indices_list = list(itertools.combinations(np.arange(len(image_list)), r=num_items))
                for subsequence_indices in subsequence_indices_list:
                    self.all_subsequences.append([image_list[idx] for idx in subsequence_indices])

    def __len__(self) -> int:
        if self.return_format == 'one_pair':
            # If we only return 1 pair of images per patient...
            return len(self.image_by_patient)
        elif self.return_format == 'all_pairs':
            # If we return all pairs of images per patient...
            return len(self.all_image_pairs)
        elif self.return_format == 'all_subsequences':
            # If we return all subsequences of images per patient...
            return len(self.all_subsequences)
        elif self.return_format == 'full_sequence':
            # If we return the full sequences of images per patient...
            return len(self.image_by_patient)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if self.return_format == 'one_pair':
            image_list = self.image_by_patient[idx]
            pair_indices = list(
                itertools.combinations(np.arange(len(image_list)), r=2))

            # # Find valid time pairs that are far enough from each other.
            # all_times = np.array([get_time(i) for i in image_list])
            # time_diff = np.diff(all_times)
            # if min(time_diff) >= self.min_time_diff:
            #     # Select from the far-enough time pairs.
            #     valid_locs = np.where(time_diff >= self.min_time_diff)[0]
            #     selected_loc = np.random.choice(valid_locs)
            #     time_pair_far_loc = [selected_loc, selected_loc + 1]
            # else:
            #     # Select the farthest time pair.
            #     time_pair_far_loc = [0, -1]

            # sampled_pair = [image_list[i] for i in time_pair_far_loc]

            # images = np.array([
            #     load_image(p, target_dim=self.target_dim, normalize=False) for p in sampled_pair
            # ])
            # timestamps = np.array([get_time(p) for p in sampled_pair])

            sampled_pair = [
                image_list[i]
                for i in pair_indices[np.random.choice(len(pair_indices))]
            ]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in sampled_pair
            ])
            timestamps = np.array([get_time(img) for img in sampled_pair])

        elif self.return_format == 'all_pairs':
            queried_pair = self.all_image_pairs[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_pair
            ])
            timestamps = np.array([get_time(img) for img in queried_pair])

        elif self.return_format == 'all_subsequences':
            queried_sequence = self.all_subsequences[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        elif self.return_format == 'full_sequence':
            queried_sequence = self.image_by_patient[idx]
            images = np.array([
                load_image(img, target_dim=self.target_dim, normalize=False) for img in queried_sequence
            ])
            timestamps = np.array([get_time(img) for img in queried_sequence])

        if self.return_format in ['one_pair', 'all_pairs']:
            assert len(images) == 2
            image1, image2 = images[0], images[1]
            if self.transforms is not None:
                transformed = self.transforms(image=image1, image_other=image2)
                image1 = transformed["image"]
                image2 = transformed["image_other"]

            image1 = normalize_image(image1)
            image2 = normalize_image(image2)

            image1 = add_channel_dim(image1)
            image2 = add_channel_dim(image2)

            images = np.vstack((image1[None, ...], image2[None, ...]))

        elif self.return_format in ['all_subsequences', 'full_sequence']:
            num_images = len(images)
            assert num_images >= 2
            assert num_images < 10  # NOTE: see `additional_targets` in `transform`.

            # Unpack the subsequence.
            image_list = np.rollaxis(images, axis=0)

            data_dict = {'image': image_list[0]}
            for idx in range(num_images - 1):
                data_dict['image_other%d' % (idx + 1)] = image_list[idx + 1]

            if self.transforms is not None:
                data_dict = self.transforms(**data_dict)

            images = normalize_image(add_channel_dim(data_dict['image']))[None, ...]
            for idx in range(num_images - 1):
                images = np.vstack((images,
                                    normalize_image(add_channel_dim(data_dict['image_other%d' % (idx + 1)]))[None, ...]))

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

        image = normalize_image(image)
        mask = mask > 128

        image = add_channel_dim(image)
        mask = add_channel_dim(mask)

        return image, mask


if __name__ == '__main__':
    print('Before preprocessing...')
    dataset = RetinaUCSFDataset(image_folder='UCSF_images_512x512')
    dataset.return_statistics()

    print('After preprocessing...')
    dataset = RetinaUCSFDataset(image_folder='UCSF_images_final_512x512')
    dataset.return_statistics()
