'''
All-in-one preprocessing script for the LUMIERE dataset.

1. Find the longitudinal FLAIR MRI images and tumor masks.
2. Perform affine registration across all images in the same longitudinal series.
3. Identify slices that show significant tumors.
4. Organize the resulting dataset in the following format.

   LUMIERE_images_final_256x256
   -- Patient-XX
      -- slice_YY
        -- week_ZZ.png

   LUMIERE_masks_final_256x256
   -- Patient-XX
      -- slice_YY
        -- week_ZZ_GBM_mask.png
'''


import os
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm
import nibabel as nib
import ants


def normalize(mri_scan):
    assert np.min(mri_scan) == 0
    # lower_bound = 0
    # upper_bound = np.percentile(mri_scan, 99.90)
    blood_value = np.mean(mri_scan[mri_scan > np.percentile(mri_scan, 99.90)])
    # mri_scan = np.clip(mri_scan, lower_bound, upper_bound)
    mri_scan = mri_scan / blood_value / 1.5
    # print(blood_value, mri_scan.max(), blood_value / mri_scan.max(), blood_value / np.percentile(mri_scan, 99.95))

    return np.uint8(mri_scan * 255)


if __name__ == '__main__':
    out_shape = np.array((256, 256))
    min_tumor_size_pixel = 2500

    image_folder = '../../data/brain_LUMIERE/raw_images/'
    final_image_folder = '../../data/brain_LUMIERE/LUMIERE_images_final_%dx%d' % (out_shape[0], out_shape[1])
    final_mask_folder = '../../data/brain_LUMIERE/LUMIERE_masks_final_%dx%d' % (out_shape[0], out_shape[1])

    patient_dirs = sorted(glob(image_folder + '*'))

    for p_folder in tqdm(patient_dirs):
        patient_str = p_folder.split('/')[-1]

        # Find all valid time points.
        time_folders = sorted(glob(p_folder + '/week-*'))
        time_list, scan_path_list, mask_path_list = [], [], []
        for t_folder in time_folders:
            time_str = os.path.basename(t_folder).replace('week-', '')

            # Find the contrast-T1 and tumor mask.
            scan_path = t_folder + '/DeepBraTumIA-segmentation/atlas/skull_strip/ct1_skull_strip.nii.gz'
            mask_path = t_folder + '/DeepBraTumIA-segmentation/atlas/segmentation/seg_mask.nii.gz'

            if os.path.isfile(scan_path) and os.path.isfile(mask_path):
                # De-duplicate.
                if '-' in time_str and \
                    len(time_list) > 0 and '-' in time_list[-1] and \
                    time_str.split('-')[0] == time_list[-1].split('-')[0]:
                    continue
                time_list.append(time_str)
                scan_path_list.append(scan_path)
                mask_path_list.append(mask_path)

        # Ignore patients with fewer than 2 time points.
        if len(time_list) < 2:
            continue

        # Quick sanity check to make sure the tumor is big enough in at least 1 slice.
        # Otherwise skip the patient.
        mask_list = [np.uint8(nib.load(m_pth).get_fdata()) for m_pth in mask_path_list]
        stacked_mask = np.stack(mask_list, axis=0)
        biggest_tumor_size = (stacked_mask > 0).sum(axis=(1, 3)).max()
        if biggest_tumor_size < min_tumor_size_pixel:
            continue

        # Perform affine registration towards the first time point in this series.
        scan_list = [normalize(nib.load(s_pth).get_fdata()) for s_pth in scan_path_list]
        mask_list = [np.uint8(nib.load(m_pth).get_fdata()) for m_pth in mask_path_list]
        assert len(scan_list) == len(mask_list)
        assert scan_list[0].shape == mask_list[0].shape

        final_stacked_scan = np.zeros((len(scan_list), *scan_list[0].shape), dtype=np.uint8)
        final_stacked_scan[0] = scan_list[0]
        final_stacked_mask = np.zeros((len(mask_list), *mask_list[0].shape), dtype=np.uint8)
        final_stacked_mask[0] = mask_list[0]

        for scan_idx in range(1, len(scan_list)):
            fixed_scan = scan_list[0]
            moving_scan = scan_list[scan_idx]
            moving_mask = mask_list[scan_idx]

            fixed_ants = ants.from_numpy(fixed_scan)
            moving_ants = ants.from_numpy(moving_scan)
            moving_mask_ants = ants.from_numpy(moving_mask)

            reg_affine = ants.registration(fixed_ants,
                                           moving_ants,
                                           'Affine')

            registered_scan = ants.apply_transforms(fixed=fixed_ants,
                                                    moving=moving_ants,
                                                    transformlist=reg_affine['fwdtransforms'],
                                                    interpolator='linear')
            registered_mask = ants.apply_transforms(fixed=fixed_ants,
                                                    moving=moving_mask_ants,
                                                    transformlist=reg_affine['fwdtransforms'],
                                                    interpolator='nearestNeighbor')

            final_stacked_scan[scan_idx] = registered_scan.numpy().astype(np.uint8)
            final_stacked_mask[scan_idx] = registered_mask.numpy().astype(np.uint8)

        # Find the "interesting" slices.
        tumor_size_by_slice = (final_stacked_mask > 0).sum(axis=(1, 3)).max(axis=0)
        interesting_slices = np.argwhere(tumor_size_by_slice > min_tumor_size_pixel).ravel()

        # Save the "interesting" slices.
        for scan_idx in range(len(scan_list)):
            for slice_idx in interesting_slices:
                out_fname_image = '%s/%s/slice_%s/week_%s.png' % (
                    final_image_folder,
                    patient_str,
                    slice_idx,
                    time_list[scan_idx],
                )
                out_fname_mask = '%s/%s/slice_%s/week_%s_GBM_mask.png' % (
                    final_mask_folder,
                    patient_str,
                    slice_idx,
                    time_list[scan_idx],
                )
                os.makedirs(os.path.dirname(out_fname_image), exist_ok=True)
                os.makedirs(os.path.dirname(out_fname_mask), exist_ok=True)

                # Resize image and mask.
                img = final_stacked_scan[scan_idx][:, slice_idx, :].T[::-1, :]
                msk = final_stacked_mask[scan_idx][:, slice_idx, :].T[::-1, :]

                reshape_ratio = img.shape[:2] / out_shape
                tmp_out_shape = np.int16(img.shape[:2] / reshape_ratio.max())

                img = cv2.resize(img, dsize=tmp_out_shape[::-1], interpolation=cv2.INTER_CUBIC)
                msk = cv2.resize(msk, dsize=tmp_out_shape[::-1], interpolation=cv2.INTER_NEAREST)

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
