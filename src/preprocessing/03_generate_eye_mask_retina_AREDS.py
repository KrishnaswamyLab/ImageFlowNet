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
        image_green = image[:, :, 1]
        x_array, y_array = np.where(image_green > np.percentile(image_green, 50))
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

    # SAM config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sam_segmenter = SAM_Segmenter(device=device, checkpoint='../../external_src/SAM/sam_vit_h_4b8939.pth')

    source_image_folders = sorted(glob(base_folder_source + '/*'))

    for folder in tqdm(source_image_folders):
        subject_folder_name = os.path.basename(folder)
        os.makedirs(base_folder_target + '/' + subject_folder_name + '/', exist_ok=True)

        image_list = sorted(glob(folder + '/*.jpg'))
        assert len(image_list) > 0

        # Generate the eye mask and save it to the target folder.
        for image_path in image_list:
            image_name = os.path.basename(image_path)
            image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            mask = sam_segmenter.segment(image)
            mask = np.uint8(mask) * 255
            mask_name = image_name.replace('.jpg', '_eye_mask.jpg')
            cv2.imwrite(base_folder_target + '/' + subject_folder_name + '/' + mask_name, mask)

if __name__ == '__main__':
    base_folder_source = '../../data/retina_areds/AREDS_2014_images_aligned_512x512/'
    base_folder_target = '../../data/retina_areds/AREDS_2014_eye_masks_aligned_512x512/'

    crop_longitudinal(base_folder_source=base_folder_source,
                      base_folder_target=base_folder_target)
