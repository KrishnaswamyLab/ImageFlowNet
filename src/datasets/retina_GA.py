import os
import random
import itertools
from glob import glob
from typing import Tuple

import numpy as np
import cv2
from torch.utils.data import Dataset


class RetinaGA(Dataset):

    def __init__(self,
                 target_dim: Tuple[int] = (512, 512),
                 base_path: str = '../../data/retina_GA',
                 image_folder: str = 'images/',
                 sample_pairs: bool = True,
                 train_identity: bool = False,
                 random_seed: int = 1):
        '''
        Information regarding the dataset.

        Files are named in the following format:
            `ID_laterality_visitNo_Type.roi`

            ID: patient identification (5 digits)
            laterality: LE = left eye; RE = right eye.
            visitNo: 00, 02, 04, etc (each visit represents 6 months interval)
            Type: FV = foveal center; GA = geographic atrophy; OD = optic disc; RP = registration point

        We will only use the GA images.

        The special thing here is that different patients may have different number of visits.
        - If a patient has fewer than 2 visits, we ignore the patient.
        - We keep track of two lists as we load the dataset.
            1) A list of indices (non-negative integers) representing the patient.
            2) A list of paths of the same patient's GA images from different visits.
          During training, as we query a patient's index, we can sample a pair of that patient's images.
          During evaluation and testing, we can use all combinations of the pairs.
            We can do macro or micro averages as we wish.
        - We need to be extra cautious that the data is split on the patient level rather than image pair level.
        '''
        self.target_dim = target_dim

        # True = return a sampled pair of each patient's images.
        # False = return all pairs of each patient's images.
        self.sample_pairs = sample_pairs

        # True = allow sampling with replacement, yielding same image as input and output.
        self.train_identity = train_identity
        self.random_seed = random_seed
        all_image_folders = sorted(glob('%s/%s/*/' % (base_path, image_folder)))

        patient_counter = 0
        self.indices = []
        self.image_sets = []

        for folder in all_image_folders:
            paths = sorted([p for p in glob('%s/*.jpg' % (folder)) if 'GA' in os.path.basename(p)])
            if len(paths) >= 2:
                self.indices.append(patient_counter)
                patient_counter += 1
                self.image_sets.append(paths)

        np.random.seed(self.random_seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx) -> Tuple[np.array, np.array]:
        image_paths_of_patient = self.image_sets[idx]

        if self.sample_pairs is True:
            if self.train_identity:
                sampled_pairs = np.random.choice(image_paths_of_patient, size=2, replace=True)
            else:
                sampled_pairs = np.random.choice(image_paths_of_patient, size=2, replace=False)
            images = np.array([load_image(p, target_dim=self.target_dim) for p in sampled_pairs])
            timestamps = np.array([get_time(p) for p in sampled_pairs])
            pair_indices = np.array([0, 1])

        else:
            images = np.array([load_image(p, target_dim=self.target_dim) for p in image_paths_of_patient])
            timestamps = np.array([get_time(p) for p in image_paths_of_patient])
            pair_indices = np.array(list(itertools.combinations(np.arange(len(image_paths_of_patient)), r=2)))

        return images, timestamps, pair_indices

    def num_image_channel(self) -> int:
        return 3

def load_image(path: str, target_dim: Tuple[int] = (256, 256)) -> np.array:
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
    if '_LE_' in path:
        time = path.split('_LE_')[1].split('_GA')[0]
    else:
        assert '_RE_' in path
        time = path.split('_RE_')[1].split('_GA')[0]
    time = float(time)
    return time