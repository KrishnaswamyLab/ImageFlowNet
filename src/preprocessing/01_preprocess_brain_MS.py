import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import nibabel as nib


def normalize(mri_scan):
    assert np.min(mri_scan) == 0
    lower_bound = 0
    upper_bound = np.percentile(mri_scan, 99.90)
    mri_scan = np.clip(mri_scan, lower_bound, upper_bound)
    mri_scan = mri_scan / upper_bound
    return np.uint8(mri_scan * 255)


if __name__ == '__main__':

    image_folder = '../../data/brain_MS/brain_MS_images/'
    out_shape = np.array((256, 256))
    file_extension = '_flair_pp.nii'

    min_num_pixel_for_lesion = 250

    subject_dirs = sorted(glob(image_folder + '*'))

    for folder in tqdm(subject_dirs):
        subject_name = folder.split('/')[-1]

        scan_paths = sorted(glob(image_folder + subject_name + '/*%s' % file_extension))
        mask_paths = sorted(glob(image_folder + subject_name + '/*mask1.nii'))

        assert len(scan_paths) == len(mask_paths)

        slices_with_ms = []
        for m_pth in (mask_paths):
            # Only use the mask to find which slices have MS.
            mask_nii = nib.load(m_pth)
            mask = mask_nii.get_fdata()
            assert mask.shape == (181, 217, 181)
            slices_with_ms.extend(np.argwhere(mask.sum(axis=(0, 1)) > min_num_pixel_for_lesion).flatten())

        slices_with_ms = np.unique(slices_with_ms)

        for s_pth, m_pth in zip(scan_paths, mask_paths):
            out_path_image = s_pth.replace('brain_MS_images', 'brain_MS_images_256x256')
            out_path_mask = s_pth.replace('brain_MS_images', 'brain_MS_masks_256x256')

            scan_nii = nib.load(s_pth)
            scan = scan_nii.get_fdata()
            scan = normalize(scan)
            assert scan.shape == (181, 217, 181)

            mask_nii = nib.load(m_pth)
            mask = mask_nii.get_fdata()
            mask = mask * 255
            assert mask.shape == (181, 217, 181)

            # Get the slices with MS.
            for i in slices_with_ms:
                out_fname_image = out_path_image.replace(subject_name, '%s_slice%s' % (subject_name, str(i).zfill(3))).replace(file_extension, '.png')
                out_fname_mask = out_path_mask.replace(subject_name, '%s_slice%s' % (subject_name, str(i).zfill(3))).replace(file_extension, '_MS_mask.png')
                os.makedirs(os.path.dirname(out_fname_image), exist_ok=True)
                os.makedirs(os.path.dirname(out_fname_mask), exist_ok=True)

                img = scan[:, :, i]
                msk = mask[:, :, i]
                reshape_ratio = img.shape[:2] / out_shape
                tmp_out_shape = np.int16(img.shape[:2] / reshape_ratio.max())

                img = cv2.resize(img,
                                 dsize=tmp_out_shape[::-1],
                                 interpolation=cv2.INTER_CUBIC)
                msk = cv2.resize(msk,
                                 dsize=tmp_out_shape[::-1],
                                 interpolation=cv2.INTER_NEAREST)

                if img.shape[0] == img.shape[1]:
                    final_img = img
                    final_mask = msk

                elif img.shape[0] > img.shape[1]:
                    final_img = np.zeros(out_shape, dtype=np.uint8)
                    final_mask = np.zeros(out_shape, dtype=np.uint8)
                    delta_size = final_img.shape[1] - img.shape[1]
                    final_img[:, delta_size // 2 + 1 : final_img.shape[1] - delta_size // 2] = img
                    final_mask[:, delta_size // 2 + 1 : final_img.shape[1] - delta_size // 2] = msk
                else:
                    final_img = np.zeros(out_shape, dtype=np.uint8)
                    final_mask = np.zeros(out_shape, dtype=np.uint8)
                    delta_size = final_img.shape[0] - img.shape[0]
                    final_img[delta_size // 2 + 1 : final_img.shape[0] -  delta_size // 2, :] = img
                    final_mask[delta_size // 2 + 1 : final_img.shape[0] -  delta_size // 2, :] = msk

                cv2.imwrite(out_fname_image, final_img)
                cv2.imwrite(out_fname_mask, final_mask)
