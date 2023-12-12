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

    def predict(self, image: np.array):
        '''
        Run Segment Anything Model (SAM) using a box prompt.
        '''
        # Use the top 50% brightest regions to estimate the prompt box.
        # Use green channel only.
        x_array, y_array = np.where(image[:, :, 1] > np.percentile(image[:, :, 1], 50))
        prompt_box = np.array([x_array.min(), y_array.min(), x_array.max(), y_array.max()])

        self.predictor.set_image(image)
        segments, _, _ = self.predictor.predict(box=prompt_box)
        segments = segments.transpose(1, 2, 0)

        mask_idx = segments.sum(axis=(0,1)).argmax()
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
    sam_predictor = SAM_Segmenter(device=device, checkpoint='../../external_src/SAM/sam_vit_h_4b8939.pth')

    source_image_folders = sorted(glob(base_folder_source + '/*'))

    for folder in tqdm(source_image_folders):
        image_list = sorted(glob(folder + '/*.jpg'))
        if len(image_list) <= 2:
            # Can ignore this folder if there is fewer than 2 images.
            pass

        # Save the cropped images.
        subject_folder_name = os.path.basename(folder)
        for i, image_path in enumerate(image_list):
            moving_image_name = os.path.basename(moving_image_path)
            moving_image = cv2.imread(moving_image_path, cv2.IMREAD_COLOR)

            goodMatch, status = match_kps(predict_config, descriptors[i], descriptors[fixed_idx])
            H_m = find_homography(goodMatch, status, cv_kpts[i], cv_kpts[fixed_idx],
                                num_matches_thr=predict_config['num_match_thr'], verbose=False)

            if H_m is not None:
                aligned_image = map_image(H_m, moving_image, fixed_image.shape)
                cv2.imwrite(base_folder_target + '/' + subject_folder_name + '/' + moving_image_name, aligned_image)
                success += 1
            else:
                print("Failed to align the two images! %s and %s" % (fixed_image_path, moving_image_path))
            total += 1

    print('Registration success rate: (%.2f%%) %d/%d' % (success/total*100, success, total))


if __name__ == '__main__':
    base_folder_source = '../../data/retina_areds/AREDS_2014_images_aligned_512x512/'
    base_folder_target = '../../data/retina_areds/AREDS_2014_images_aligned_cropped_512x512/'

    crop_longitudinal(base_folder_source=base_folder_source,
                      base_folder_target=base_folder_target)
