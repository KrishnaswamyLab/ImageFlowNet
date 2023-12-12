import os
import sys
from glob import glob
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from typing import Tuple, List, Iterable
from segment_anything import SamPredictor, sam_model_registry


class SAM_Segmenter(object):
    def __init__(self, device: torch.device, checkpoint: str):
        '''
        Initialize a Segment Anything Model (SAM) model.
        '''
        sam_model = sam_model_registry["default"](checkpoint=checkpoint).to(device)
        self.predictor = SamPredictor(sam_model)

    def segment(self, image: np.array):
        '''
        Run Segment Anything Model (SAM) using a box prompt.
        '''
        # Estimate the prompt box.
        image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        x_array, y_array = np.where(image_gray > np.percentile(image_gray, 75))
        prompt_box = np.array([x_array.min(), y_array.min(), x_array.max(), y_array.max()])

        self.predictor.set_image(image)
        segments, _, _ = self.predictor.predict(box=prompt_box)
        segments = segments.transpose(1, 2, 0)

        mask_idx = segments.sum(axis=(0, 1)).argmax()
        mask = segments[..., mask_idx]
        return mask


def crop_longitudinal(base_folder_source: str, base_folder_target: str):
    '''
    Crop the longitudinal images.
    These images are already spatially registered.
    We only need to crop them such that only the overlapping region of the images remain.

    For the case in `data/retina_areds/AREDS_2014_images_aligned_512x512/`,
    each folder represents a series of longitudinal images to be cropped.
    '''

    # SuperRetina config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam_segmenter = SAM_Segmenter(device=device, checkpoint='../../external_src/SAM/sam_vit_h_4b8939.pth')

    source_image_folders = sorted(glob(base_folder_source + '/*'))

    for folder in tqdm(source_image_folders):
        subject_folder_name = os.path.basename(folder)
        os.makedirs(base_folder_target + '/' + subject_folder_name + '/', exist_ok=True)

        image_list = sorted(glob(folder + '/*.jpg'))
        assert len(image_list) > 0

        if len(image_list) == 1:
            # Directly copy the image over if there is only 1 image.
            image_path = image_list[0]
            image_name = os.path.basename(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            cv2.imwrite(base_folder_target + '/' + subject_folder_name + '/' + image_name, image)

        # Build a common mask for the images.
        common_mask = None
        for image_path in image_list:
            image_name = os.path.basename(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            mask = sam_segmenter.segment(image)
            if common_mask is None:
                common_mask = mask
            else:
                common_mask = np.logical_and(common_mask, mask)

        # Apply the common mask on all images.
        assert common_mask is not None
        for image_path in image_list:
            image_name = os.path.basename(image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image[~common_mask] = 0
            cv2.imwrite(base_folder_target + '/' + subject_folder_name + '/' + image_name, image)


if __name__ == '__main__':
    base_folder_source = '../../data/retina_areds/AREDS_2014_images_aligned_512x512/'
    base_folder_target = '../../data/retina_areds/AREDS_2014_images_aligned_cropped_512x512/'

    crop_longitudinal(base_folder_source=base_folder_source,
                      base_folder_target=base_folder_target)
