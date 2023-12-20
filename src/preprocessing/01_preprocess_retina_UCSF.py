import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
from read_roi import read_roi_file
from read_roi import read_roi_zip
from PIL import Image, ImageDraw


def polygon_to_mask(x_list, y_list, mask_shape):
    polygon_coords = [(x, y) for (x, y) in zip(x_list, y_list)]
    mask = Image.new('L', mask_shape[:2], 0)
    ImageDraw.Draw(mask).polygon(polygon_coords, outline=1, fill=1)
    mask = np.array(mask)
    return mask


def resize_and_pad(img, out_shape):
    reshape_ratio = img.shape[:2] / out_shape[:2]
    tmp_out_shape = np.int16(img.shape[:2] / reshape_ratio.max())

    img = cv2.resize(img,
                     dsize=tmp_out_shape[::-1],
                     interpolation=cv2.INTER_CUBIC)

    if img.shape[0] == img.shape[1]:
        final_img = img
    elif img.shape[0] > img.shape[1]:
        final_img = np.zeros((out_shape), dtype=np.uint8)
        delta_size = final_img.shape[1] - img.shape[1]
        final_img[:, delta_size // 2:final_img.shape[1] -
                  delta_size // 2, ...] = img
    else:
        final_img = np.zeros((out_shape), dtype=np.uint8)
        delta_size = final_img.shape[0] - img.shape[0]
        final_img[delta_size // 2:final_img.shape[0] -
                  delta_size // 2, :, ...] = img

    return final_img


if __name__ == '__main__':
    image_folder = '../../data/retina_ucsf/Images/Raw Images/'
    roi_folder = '../../data/retina_ucsf/Images/Graded Images and ROI Files/LS_review/ImageJROI/All/'

    image_paths = sorted(glob(image_folder + '*/*.tif'))
    # Currently, shape has to be a square.
    out_shape_image = np.array((512, 512, 3))
    out_shape_mask = np.array((512, 512))

    for pth in tqdm(image_paths):

        folder_name = pth.split('/')[-2]
        grader = os.path.basename(pth).split(folder_name)[0]
        unique_identifier = '_'.join(os.path.basename(pth).split('_')[:3])

        # Ignore the "graded" images.
        if len(grader) > 0:
            continue

        # Save the image
        out_path_image = pth.replace('Images/Raw Images/',
                                     'UCSF_images_512x512/').replace('.tif', '.png')
        os.makedirs(os.path.dirname(out_path_image), exist_ok=True)

        img = cv2.imread(pth)
        raw_image_shape = img.shape

        img = resize_and_pad(img, out_shape_image)
        cv2.imwrite(out_path_image, img)

        # Find the corresponding ROI files.
        roi_files = sorted(glob(roi_folder + folder_name + '/' + unique_identifier + '*.roi'))
        roi_zip_files = sorted(glob(roi_folder + folder_name + '/' + unique_identifier + '*.zip'))

        # Convert the ROI files to masks.
        for roi_file in roi_files:
            _roi = read_roi_file(roi_file)

            assert len(_roi.keys()) == 1
            for k in _roi.keys():
                _roi_item = _roi[k]

            roi_save_name = _roi_item['name']

            assert unique_identifier in _roi_item['name']

            mask = np.zeros(raw_image_shape[:2])

            assert _roi_item['type'] in ['point', 'polygon']
            if _roi_item['type'] == 'point':
                #TODO: Use a small square to represent the point.
                continue

            elif _roi_item['type'] == 'polygon':
                # Convert the polygon to a mask.
                mask = polygon_to_mask(_roi_item['x'], _roi_item['y'], raw_image_shape)
                mask = resize_and_pad(mask, out_shape_mask)

            out_path_mask = pth.replace('Images/Raw Images/',
                                        'UCSF_masks_512x512/').replace(
                                            os.path.basename(pth), roi_save_name + '_mask.png')
            os.makedirs(os.path.dirname(out_path_mask), exist_ok=True)
            cv2.imwrite(out_path_mask, mask)

        # Convert the ROI zip files to masks.
        for roi_zip_file in roi_zip_files:
            _roi = read_roi_zip(roi_zip_file)
            roi_save_name = os.path.basename(roi_zip_file).replace('.roi', '').replace('.zip', '')
            assert unique_identifier in roi_save_name

            assert len(_roi.keys()) > 1

            mask = None
            for k in _roi.keys():
                _roi_item = _roi[k]

                assert _roi_item['type'] == 'polygon'

                curr_mask = polygon_to_mask(_roi_item['x'], _roi_item['y'], raw_image_shape)
                curr_mask = resize_and_pad(curr_mask, out_shape_mask)

                if mask is None:
                    mask = curr_mask
                else:
                    mask = np.logical_or(mask, curr_mask)
            mask = np.uint8(mask * 255)

            out_path_mask = pth.replace('Images/Raw Images/',
                                        'UCSF_masks_512x512/').replace(
                                            os.path.basename(pth), roi_save_name + '_mask.png')
            os.makedirs(os.path.dirname(out_path_mask), exist_ok=True)
            cv2.imwrite(out_path_mask, mask)

