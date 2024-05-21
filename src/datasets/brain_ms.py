'''
A longitudinal brain Multiple Sclerosis dataset, ISBI 2015
"Longitudinal multiple sclerosis lesion segmentation: Resource and challenge"
'''


import itertools
from typing import Literal
from glob import glob
from typing import List, Tuple

import os
import cv2
import numpy as np
from torch.utils.data import Dataset

root_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])


def brain_MS_split():
    '''
    Hard-coded to make sure split is correct
    '''
    train_indices = [
        34, 36, 37, 38, 39, 40,
        41, 42, 43, 44, 46, 47, 48, 49,
        51, 52, 53, 54, 56, 57, 58, 59,
        61, 62, 63, 64, 66, 67, 68, 69,
        71, 72, 73, 74, 76, 77
    ], # subject 2
    val_indices = [
        35, 40, 45, 50, 55, 60, 65, 70, 75
    ], # subject 2
    test_indices = [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
        31, 32, 33, 34, 35, 36, 37, 38, 78
    ] # subject 1 and subject 3
    return train_indices, val_indices, test_indices


def normalize_image(image: np.array) -> np.array:
    '''
    Image already normalized on scan level.
    Just transform to [-1, 1] and clipped to [-1, 1].
    '''
    image = image / 255.0 * 2 - 1
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
    time = os.path.basename(path).split('_')[1].replace('.png', '')
    # Shall be 2 or 3 digits
    assert len(time) in [2, 3]
    time = float(time)
    return time


class BrainMSDataset(Dataset):

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_MS/',
                 image_folder: str = 'brain_MS_images_256x256/',
                 target_dim: Tuple[int] = (256, 256)):
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
        # NOTE: "patient" in the context outside means "slices" here.
        # The following printout is using a more precise terminology.
        unique_slice_list = np.unique([os.path.basename(item.rstrip('/')) for item in self.all_image_folders])
        unique_patient_list = np.unique([item.split('_')[0] for item in unique_slice_list])
        print('Number of unique patients:', len(unique_patient_list))
        print('Number of unique slices:', len(unique_slice_list))

        num_visit_map = {}
        for item in self.image_by_patient:
            num_visit = len(item)
            if num_visit not in num_visit_map.keys():
                num_visit_map[num_visit] = 1
            else:
                num_visit_map[num_visit] += 1
        for k, v in sorted(num_visit_map.items()):
            print('%d visits: %d slices.' % (k, v))
        return

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class BrainMSSubset(BrainMSDataset):

    def __init__(self,
                 main_dataset: BrainMSDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs', 'all_subsequences', 'all_subarrays', 'full_sequence'],
                 transforms = None,
                 transforms_aug = None):
        '''
        A subset of BrainMSDataset.

        In BrainMSDataset, we carefully isolated the (variable number of) images from
        different patients, and in train/val/test split we split the data by
        patient rather than by image.

        Now we have 3 instances of BrainMSSubset, one for each train/val/test set.
        In each set, we can safely unpack the images out.
        We want to organize the images such that each time `__getitem__` is called,
        it gets a pair of [x_start, x_end] and [t_start, t_end].
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format
        self.transforms = transforms
        self.transforms_aug = transforms_aug

        self.image_by_patient = [
            main_dataset.image_by_patient[i] for i in subset_indices
        ]

        self.all_image_pairs = []
        self.all_subsequences = []
        self.all_subarrays = []
        for image_list in self.image_by_patient:
            pair_indices = list(itertools.combinations(np.arange(len(image_list)), r=2))
            for (idx1, idx2) in pair_indices:
                self.all_image_pairs.append(
                    [image_list[idx1], image_list[idx2]])
                self.all_subarrays.append(image_list[idx1 : idx2+1])

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
        elif self.return_format == 'all_subarrays':
            # If we return all subarrays of images per patient...
            return len(self.all_subarrays)
        elif self.return_format == 'full_sequence':
            # If we return the full sequences of images per patient...
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

        elif self.return_format == 'all_subarrays':
            queried_sequence = self.all_subarrays[idx]
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

            if self.transforms_aug is not None:
                transformed_aug = self.transforms_aug(image=image1, image_other=image1)
                image1_aug = transformed_aug["image"]
                image1_aug = normalize_image(image1_aug)
                image1_aug = add_channel_dim(image1_aug)

            image1 = normalize_image(image1)
            image2 = normalize_image(image2)

            image1 = add_channel_dim(image1)
            image2 = add_channel_dim(image2)

            if self.transforms_aug is not None:
                images = np.vstack((image1[None, ...], image2[None, ...], image1_aug[None, ...]))
            else:
                images = np.vstack((image1[None, ...], image2[None, ...]))

        elif self.return_format in ['all_subsequences', 'all_subarrays', 'full_sequence']:
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


class BrainMSSegDataset(Dataset):

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_MS/',
                 image_folder: str = 'brain_MS_images_256x256/',
                 mask_folder: str = 'brain_MS_masks_256x256/',
                 target_dim: Tuple[int] = (256, 256)):
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
                mask_path_ = image_path_.replace('.png', '').replace(
                    image_folder, mask_folder) + '_MS_mask.png'
                assert os.path.isfile(mask_path_)
                mask_paths.append(mask_path_)
            self.image_by_patient.append(image_paths)
            self.mask_by_patient.append(mask_paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class BrainMSSegSubset(BrainMSSegDataset):

    def __init__(self,
                 main_dataset: BrainMSSegDataset = None,
                 subset_indices: List[int] = None,
                 transforms = None):
        '''
        A subset of BrainMSSegDataset.
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
    dataset = BrainMSDataset(image_folder='brain_MS_images_256x256')
    dataset.return_statistics()
