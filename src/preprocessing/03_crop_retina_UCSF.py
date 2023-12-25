import os
import sys
from glob import glob
import torch
import cv2
import numpy as np
from tqdm import tqdm


def find_biggest_intersecting_square(object_mask: np.array) -> np.array:
    assert len(object_mask.shape) == 2
    assert object_mask.min() == 0 and object_mask.max() == 1

    # `square_size[i, j]` is the side length of the biggest square
    # whose bottom right corner is at [i, j].
    square_size = np.int16(np.zeros_like(object_mask))
    for i in range(square_size.shape[0]):
        for j in range(square_size.shape[1]):
            if i == 0 or j == 0:
                square_size[i, j] = np.int16(object_mask[i, j])
            elif object_mask[i, j] == 0:
                square_size[i, j] = 0
            else:
                square_size[i, j] = 1 + np.min([
                    square_size[i-1, j], square_size[i, j-1], square_size[i-1, j-1]
                ])

    # Return the top-left and the bottom-right squares, in case one is better than the other
    max_size = square_size.max()
    bottomright_arr = np.where(square_size == max_size)
    # top left square.
    h_br, w_br = bottomright_arr[0][0], bottomright_arr[1][0]
    h_tl, w_tl = h_br - max_size, w_br - max_size
    square_mask_tl = np.zeros_like(object_mask)
    square_mask_tl[h_tl + 1: h_br, w_tl + 1 : w_br] = 1

    # bottom right square.
    h_br, w_br = bottomright_arr[0][-1], bottomright_arr[1][-1]
    h_tl, w_tl = h_br - max_size, w_br - max_size
    square_mask_br = np.zeros_like(object_mask)
    square_mask_br[h_tl + 1: h_br, w_tl + 1 : w_br] = 1

    return square_mask_tl, square_mask_br

def crop_longitudinal(output_shape,
                      base_folder_source: str,
                      base_mask_folder_source: str,
                      base_fg_mask_folder_source: str,
                      base_folder_target: str,
                      base_mask_folder_target: str):
    '''
    Crop the longitudinal images.

    These longitudinal images are already registered.

    We need to crop them such that the final images only have foreground regions (`fg_mask`),
    and they contain (hopefully) the entire geographic atropy mask (`mask`).
    '''

    source_image_folders = sorted(glob(base_folder_source + '/*'))

    for folder in tqdm(source_image_folders):
        image_list = sorted(glob(folder + '/*.png'))
        if len(image_list) <= 2:
            # Can ignore this folder if there is fewer than 2 images.
            pass

        # Find the corresponding geographic atrophy masks and foreground masks.
        ga_mask_list = ['_'.join(img_path.split('_')[:-1]).replace(base_folder_source, base_mask_folder_source) + '_GA_mask.png' for img_path in image_list]
        fg_mask_list = [img_path.replace(base_folder_source, base_fg_mask_folder_source).replace('.png', '_foreground_mask.png') for img_path in image_list]

        all_ga_intersection = None
        all_fg_intersection = None

        for i in range(len(image_list)):
            ga_mask_path = ga_mask_list[i]
            fg_mask_path = fg_mask_list[i]

            ga_mask = cv2.imread(ga_mask_path, cv2.IMREAD_GRAYSCALE)
            fg_mask = cv2.imread(fg_mask_path, cv2.IMREAD_GRAYSCALE)

            assert ga_mask.max() == 255
            assert fg_mask.max() == 255

            if all_ga_intersection is None:
                all_ga_intersection = ga_mask > 128
                all_fg_intersection = fg_mask > 128
            else:
                all_ga_intersection = np.logical_and(all_ga_intersection, ga_mask > 128)
                all_fg_intersection = np.logical_and(all_fg_intersection, fg_mask > 128)

        # Check that `all_ga_intersection` is fully inside `all_fg_intersection`.
        assert (np.logical_or(all_ga_intersection, all_fg_intersection) == all_fg_intersection).all()

        # Find a square that is fully inside `all_fg_intersection`.
        assert all_fg_intersection.min() in [0, 1]

        if all_fg_intersection.min() == 1:
            common_fg_mask = all_fg_intersection
        else:
            common_fg_mask_tl, common_fg_mask_br = find_biggest_intersecting_square(all_fg_intersection)
            if np.logical_and(all_ga_intersection, common_fg_mask_tl).sum() > np.logical_and(all_ga_intersection, common_fg_mask_br).sum():
                common_fg_mask = common_fg_mask_tl
            else:
                common_fg_mask = common_fg_mask_br

        mask_arr = np.where(common_fg_mask)
        h_tl, w_tl = mask_arr[0][0], mask_arr[1][0]
        h_br, w_br = mask_arr[0][-1], mask_arr[1][-1]

        for i, image_path in enumerate(image_list):
            image_path = image_list[i]
            ga_mask_path = ga_mask_list[i]

            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            ga_mask = cv2.imread(ga_mask_path, cv2.IMREAD_GRAYSCALE)

            image = image[h_tl : h_br, w_tl : w_br, :]
            ga_mask = ga_mask[h_tl : h_br, w_tl : w_br]

            image = cv2.resize(image, dsize=output_shape[::-1], interpolation=cv2.INTER_CUBIC)
            ga_mask = cv2.resize(ga_mask, dsize=output_shape[::-1], interpolation=cv2.INTER_CUBIC)
            ga_mask = np.uint8((ga_mask > 128) * 255)

            final_image_path = image_path.replace(base_folder_source, base_folder_target)
            final_ga_mask_path = ga_mask_path.replace(base_mask_folder_source, base_mask_folder_target)

            os.makedirs(os.path.dirname(final_image_path), exist_ok=True)
            cv2.imwrite(final_image_path, image)
            os.makedirs(os.path.dirname(final_ga_mask_path), exist_ok=True)
            cv2.imwrite(final_ga_mask_path, ga_mask)


if __name__ == '__main__':
    base_folder_source = '../../data/retina_ucsf/UCSF_images_aligned_512x512/'
    base_mask_folder_source = '../../data/retina_ucsf/UCSF_masks_aligned_512x512/'
    base_fg_mask_folder_source = '../../data/retina_ucsf/UCSF_FG_masks_aligned_512x512/'
    base_folder_target = '../../data/retina_ucsf/UCSF_images_final_512x512/'
    base_mask_folder_target = '../../data/retina_ucsf/UCSF_masks_final_512x512/'

    output_shape = (512, 512)

    crop_longitudinal(output_shape=output_shape,
                      base_folder_source=base_folder_source,
                      base_mask_folder_source=base_mask_folder_source,
                      base_fg_mask_folder_source=base_fg_mask_folder_source,
                      base_folder_target=base_folder_target,
                      base_mask_folder_target=base_mask_folder_target)
