import cv2
import os
import numpy as np
from tqdm import tqdm
from typing import Tuple
from matplotlib import colormaps


def _generate_longitudinal(image_shape: Tuple[int] = (256, 256),
                           num_images: int = 10,
                           initial_radius: Tuple[int] = (18, 16),
                           final_radius: Tuple[int] = (36, 48),
                           random_seed: int = None):
    '''
    Generate longitudinal images of an big rectangle containing a small ellipse.
    The big square (eye) remains unchanged, while the small ellipse (geographic atrophy) grows.
    '''

    images = [np.zeros((*image_shape, 3), dtype=np.uint8) for _ in range(num_images)]

    if random_seed is not None:
        np.random.seed(random_seed)

    color_rectangle = np.uint8(np.array(colormaps['copper'](np.random.choice(range(colormaps['copper'].N)))[:3]) * 255)
    color_ellipse = np.uint8(np.array(colormaps['Wistia'](np.random.choice(range(colormaps['Wistia'].N)))[:3]) * 255)

    # First generate the big rectangle.
    square_tl = [int(np.random.uniform(1/8*image_shape[i], 1/4*image_shape[i]/4)) for i in range(2)]
    square_br = [int(np.random.uniform(3/4*image_shape[i], 7/8*image_shape[i])) for i in range(2)]
    square_centroid = np.mean([square_tl, square_br], axis=0)
    for image in images:
        image[square_tl[0]:square_br[0],
              square_tl[1]:square_br[1], :] = color_rectangle

    # Then generate the increasingly bigger ellipses.
    ellipse_centroid = [int(np.random.uniform(square_tl[i]+final_radius[i],
                                              square_br[i]-final_radius[i])) for i in range(2)]
    radius_x_list = np.linspace(initial_radius[0], final_radius[0], num_images)
    radius_y_list = np.linspace(initial_radius[1], final_radius[1], num_images)
    for i, image in enumerate(images):
        x_arr = np.linspace(0, image_shape[0]-1, image_shape[0])[:, None]
        y_arr = np.linspace(0, image_shape[1]-1, image_shape[1])
        ellipse_mask = ((x_arr-ellipse_centroid[0])/radius_x_list[i])**2 + \
            ((y_arr-ellipse_centroid[1])/radius_y_list[i])**2 <= 1
        image[ellipse_mask, :] = color_ellipse

    # OpenCV color channel convertion for saving.
    images = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images]
    return images, square_centroid


def synthesize_dataset(save_folder: str = '../data/synthesized/', num_subjects: int = 200):
    '''
    Synthesize 4 datasets.
    1. The first dataset has no spatial variation. It has pixel-level alignment temporally.
    2. The second dataset has a predictable translation factor.
    3. The third dataset has a predictable rotation factor.
    4. The fourth dataset is irregular. At each time point, we randomly pick an image from 1/2/3 at that time point.
    '''

    for subject_idx in tqdm(range(num_subjects)):
        images, square_centroid = _generate_longitudinal(random_seed=subject_idx)
        images_trans, images_rot = [], []

        # Do nothing.
        dataset = 'base'
        os.makedirs(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5), exist_ok=True)
        for time_idx, img in enumerate(images):
            cv2.imwrite(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5) + '/subject_%s_time_%s.png' % (
                str(subject_idx).zfill(5), str(time_idx).zfill(3)), img)

        # Add translation.
        dataset = 'translation'
        os.makedirs(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5), exist_ok=True)
        max_trans_x, max_trans_y = 32, 32
        for time_idx, img in enumerate(images):
            translation_x = int(2 * max_trans_x / (len(images) - 1) * time_idx - max_trans_x)
            translation_y = int(max_trans_y * np.cos(time_idx / len(images) * 2*np.pi))
            translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])
            img_trans = cv2.warpAffine(img, translation_matrix, (img.shape[0], img.shape[1]))
            cv2.imwrite(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5) + '/subject_%s_time_%s.png' % (
                str(subject_idx).zfill(5), str(time_idx).zfill(3)), img_trans)
            images_trans.append(img_trans)

        # Add rotation.
        dataset = 'rotation'
        os.makedirs(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5), exist_ok=True)
        for time_idx, img in enumerate(images):
            angle = np.linspace(0, 180, len(images))[time_idx]
            rotation_matrix = cv2.getRotationMatrix2D((square_centroid[1], square_centroid[0]), angle, 1)
            img_rot = cv2.warpAffine(img, rotation_matrix, (img.shape[0], img.shape[1]))
            cv2.imwrite(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5) + '/subject_%s_time_%s.png' % (
                str(subject_idx).zfill(5), str(time_idx).zfill(3)), img_rot)
            images_rot.append(img_rot)

        # Randomly pick from previous lists.
        dataset = 'mixing'
        os.makedirs(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5), exist_ok=True)
        for time_idx in range(len(images)):
            chosen_list = np.random.choice(['images', 'images_trans', 'images_rot'])
            cv2.imwrite(save_folder + dataset + '/subject_%s' % str(subject_idx).zfill(5) + '/subject_%s_time_%s.png' % (
                str(subject_idx).zfill(5), str(time_idx).zfill(3)), eval(chosen_list)[time_idx])
    return


if __name__ == '__main__':
    synthesize_dataset()