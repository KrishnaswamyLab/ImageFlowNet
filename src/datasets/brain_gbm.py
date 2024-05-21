'''
A longitudinal brain Glioblastoma dataset.
"The LUMIERE dataset: Longitudinal Glioblastoma MRI with expert RANO evaluation"
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


def normalize_image(image: np.array) -> np.array:
    '''
    Image already normalized on scan level.
    Just transform to [-1, 1] and clipped to [-1, 1].
    '''
    assert image.min() >= 0 and image.max() <= 255
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
    time = os.path.basename(path).replace('week_', '').split('-')[0].replace('.png', '')
    # Shall be 2 or 3 digits
    assert len(time) in [2, 3]
    time = float(time)
    return time


class BrainGBMDataset(Dataset):

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_LUMIERE/',
                 image_folder: str = 'LUMIERE_images_tumor1200px_256x256/',
                 max_slice_per_patient: int = 20,
                 target_dim: Tuple[int] = (256, 256)):
        '''
        The special thing here is that different patients may have different number of visits.
        - If a patient has fewer than 2 visits, we ignore the patient.
        - When a patient's index is queried, we return images from all visits of that patient.
        - We need to be extra cautious that the data is split on the patient level rather than image pair level.

        NOTE: since different patients may have different number of visits, the returned array will
        not necessarily be of the same shape. Due to the concatenation requirements, we can only
        set batch size to 1 in the downstream Dataloader.

        NOTE: This dataset is structured like this:
        LUMIERE_images_final_256x256
        -- Patient-XX
            -- slice_YY
                -- week_ZZ.png

        LUMIERE_masks_final_256x256
        -- Patient-XX
            -- slice_YY
                -- week_ZZ_GBM_mask.png

        So we will organize outputs on the unit of slices.
        Each slice is essentially treated as a separate trajectory.
        But importantly, data partitioning is done on the unit of patients.
        '''
        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient

        self.all_patient_folders = sorted(glob('%s/%s/*/' % (base_path, image_folder)))
        self.all_patient_ids = [os.path.basename(item.rstrip('/')) for item in self.all_patient_folders]
        self.patient_id_to_slice_id = []  # maps the patient id to a list of corresponding slice ids.

        self.image_by_slice = []

        self.max_t = 0

        curr_slice_idx = 0
        for folder in self.all_patient_folders:

            num_slices_curr_patient = 0
            slice_arr = np.array(sorted(glob('%s/slice*/' % (folder))))

            if self.max_slice_per_patient is not None \
                and len(slice_arr) > self.max_slice_per_patient:
                subset_ids = np.linspace(0, len(slice_arr)-1, self.max_slice_per_patient)
                subset_ids = np.array([int(item) for item in subset_ids])
                slice_arr = slice_arr[subset_ids]

            for curr_slice in slice_arr:
                paths = sorted(glob('%s/week*.png' % curr_slice))

                '''
                Ignore week 0!!!
                Week 0 is pre-operation, which means tumors will be cut!
                This dynamics may be too complicated to learn.
                If we ignore week 0, the remaining will likely be natural growth of tumor.
                '''
                paths = [p for p in paths if 'week_000' not in p]

                if len(paths) >= 2:
                    self.image_by_slice.append(paths)
                    num_slices_curr_patient += 1
                for p in paths:
                    self.max_t = max(self.max_t, get_time(p))

            self.patient_id_to_slice_id.append(np.arange(curr_slice_idx, curr_slice_idx + num_slices_curr_patient))
            curr_slice_idx += num_slices_curr_patient


    def return_statistics(self) -> None:
        print('max time (weeks):', self.max_t)

        unique_patient_list = np.unique(self.all_patient_ids)
        print('Number of unique patients:', len(unique_patient_list))
        print('Number of unique slices:', len(self.image_by_slice))

        num_visit_map = {}
        for item in self.image_by_slice:
            num_visit = len(item)
            if num_visit not in num_visit_map.keys():
                num_visit_map[num_visit] = 1
            else:
                num_visit_map[num_visit] += 1
        for k, v in sorted(num_visit_map.items()):
            print('%d visits: %d slices.' % (k, v))
        return

    def __len__(self) -> int:
        return len(self.all_patient_ids)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class BrainGBMSubset(BrainGBMDataset):

    def __init__(self,
                 main_dataset: BrainGBMDataset = None,
                 subset_indices: List[int] = None,
                 return_format: str = Literal['one_pair', 'all_pairs', 'all_subsequences', 'all_subarrays', 'full_sequence'],
                 transforms = None,
                 transforms_aug = None):
        '''
        A subset of BrainGBMDataset.

        In BrainGBMDataset, we carefully isolated the (variable number of) images from
        different patients, and in train/val/test split we split the data by
        patient rather than by image.

        Now we have 3 instances of BrainGBMSubset, one for each train/val/test set.
        In each set, we can safely unpack the images out.
        We want to organize the images such that each time `__getitem__` is called,
        it gets a pair of [x_start, x_end] and [t_start, t_end].
        '''
        super().__init__()

        self.target_dim = main_dataset.target_dim
        self.return_format = return_format
        self.transforms = transforms
        self.transforms_aug = transforms_aug

        self.image_by_slice = []

        for patient_id in subset_indices:
            slice_ids = main_dataset.patient_id_to_slice_id[patient_id]
            self.image_by_slice.extend([main_dataset.image_by_slice[i] for i in slice_ids])

        self.all_image_pairs = []
        self.all_subsequences = []
        self.all_subarrays = []
        for image_list in self.image_by_slice:
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
            return len(self.image_by_slice)
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
            return len(self.image_by_slice)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        if self.return_format == 'one_pair':
            image_list = self.image_by_slice[idx]
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
            queried_sequence = self.image_by_slice[idx]
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
            assert num_images < 20  # NOTE: see `additional_targets` in `transform`.

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


class BrainGBMSegDataset(Dataset):

    def __init__(self,
                 base_path: str = root_dir + '/data/brain_LUMIERE/',
                 image_folder: str = 'LUMIERE_images_tumor1200px_256x256/',
                 mask_folder: str = 'LUMIERE_masks_tumor1200px_256x256/',
                 max_slice_per_patient: int = 20,
                 target_dim: Tuple[int] = (256, 256)):
        '''
        This dataset is for segmentation.
        '''
        super().__init__()

        self.target_dim = target_dim
        self.max_slice_per_patient = max_slice_per_patient

        all_patient_folders = sorted(glob('%s/%s/Patient-*/' % (base_path, image_folder)))

        self.image_by_patient = []
        self.mask_by_patient = []

        for patient_folder in all_patient_folders:
            curr_patient_slice_folders = np.array(sorted(glob('%s/slice*/' % patient_folder)))

            if self.max_slice_per_patient is not None \
                and len(curr_patient_slice_folders) > self.max_slice_per_patient:
                subset_ids = np.linspace(0, len(curr_patient_slice_folders)-1, self.max_slice_per_patient)
                subset_ids = np.array([int(item) for item in subset_ids])
                curr_patient_slice_folders = curr_patient_slice_folders[subset_ids]

            for im_folder in curr_patient_slice_folders:
                image_paths = sorted(glob('%s/*.png' % im_folder))
                mask_paths = []
                for image_path_ in image_paths:
                    mask_path_ = image_path_.replace('.png', '').replace(
                        image_folder, mask_folder) + '_GBM_mask.png'
                    assert os.path.isfile(mask_path_)
                    mask_paths.append(mask_path_)
                self.image_by_patient.append(image_paths)
                self.mask_by_patient.append(mask_paths)

    def __len__(self) -> int:
        return len(self.image_by_patient)

    def num_image_channel(self) -> int:
        ''' Number of image channels. '''
        return 1


class BrainGBMSegSubset(BrainGBMSegDataset):

    def __init__(self,
                 main_dataset: BrainGBMSegDataset = None,
                 subset_indices: List[int] = None,
                 transforms = None):
        '''
        A subset of BrainGBMSegDataset.
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

        # I believe this means necrosis and contrast enhancement.
        # necrosis: 85, contrast enhancement: 170, edema: 255.
        assert mask.min() == 0 and mask.max() <= 255
        mask = np.logical_and(mask > 0, mask < 250)

        image = add_channel_dim(image)
        mask = add_channel_dim(mask)

        return image, mask


if __name__ == '__main__':
    print('Full set.')
    dataset = BrainGBMDataset(max_slice_per_patient=None)
    dataset.return_statistics()

    print('Subset with max 20 slices per patient.')
    dataset = BrainGBMDataset(max_slice_per_patient=20)
    dataset.return_statistics()
