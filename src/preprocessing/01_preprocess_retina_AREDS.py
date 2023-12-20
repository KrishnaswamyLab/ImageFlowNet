import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm

if __name__ == '__main__':
    image_folder = '../../data/retina_areds/AREDS_2014_images/'

    image_paths = sorted(glob(image_folder + '*/*.jpg'))
    out_shape = np.array((512, 512))

    for pth in tqdm(image_paths):
        out_path = pth.replace('AREDS_2014_images',
                               'AREDS_2014_images_512x512')
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        img = cv2.imread(pth)
        reshape_ratio = img.shape[:2] / out_shape
        tmp_out_shape = np.int16(img.shape[:2] / reshape_ratio.max())

        img = cv2.resize(img,
                         dsize=tmp_out_shape[::-1],
                         interpolation=cv2.INTER_CUBIC)

        if img.shape[0] == img.shape[1]:
            final_img = img
        elif img.shape[0] > img.shape[1]:
            final_img = np.zeros((*out_shape, 3), dtype=np.uint8)
            delta_size = final_img.shape[1] - img.shape[1]
            final_img[:, delta_size // 2:final_img.shape[1] -
                      delta_size // 2, :] = img
        else:
            final_img = np.zeros((*out_shape, 3), dtype=np.uint8)
            delta_size = final_img.shape[0] - img.shape[0]
            final_img[delta_size // 2:final_img.shape[0] -
                      delta_size // 2, :, :] = img

        cv2.imwrite(out_path, final_img)
