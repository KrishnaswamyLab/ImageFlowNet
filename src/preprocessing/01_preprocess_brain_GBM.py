'''
All-in-one preprocessing script for the LUMIERE dataset.

1. Find the longitudinal FLAIR MRI images and tumor masks.
2. Perform affine + diffeomorphic registration across all images in the same longitudinal series.
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

from sklearn.linear_model import LinearRegression


def normalize(mri_scan):
    assert np.min(mri_scan) == 0
    # Approximately the pixel intensity of the blood.
    blood_value = np.mean(mri_scan[mri_scan > np.percentile(mri_scan, 99.90)])
    mri_scan = mri_scan / blood_value / 1.5
    return np.uint8(mri_scan * 255)

def rescale_mask(mask):
    assert mask.min() == 0 and mask.max() <= 3
    return np.uint8(mask * 85)  # scale it to [0, 255]

def correct_receiver_gain(mri_scan_fixed,
                          mri_scan_to_adjust,
                          tumor_mask):
    '''
    Use the first MRI scan to correct for the second.
    Using the non-tumor region to adjust for the receiver gain.
    Note: only return the adjusted, second MRI scan.

    Assuming scans are registered.
    '''

    brain_mask = mri_scan_to_adjust > 0
    non_tumor_brain_mask = np.logical_and(brain_mask, ~tumor_mask)

    X = mri_scan_to_adjust[non_tumor_brain_mask].reshape(-1, 1)
    y = mri_scan_fixed[non_tumor_brain_mask].reshape(-1, 1)

    # Fit linear model
    lm = LinearRegression(fit_intercept=False)
    lm.fit(X, y)

    # Extract gain (slope)
    gain = lm.coef_[0][0]

    corrected_scan = np.zeros_like(mri_scan_to_adjust, dtype=np.uint8)
    corrected_scan = gain * mri_scan_to_adjust

    return corrected_scan


if __name__ == '__main__':
    out_shape = np.array((256, 256))
    min_tumor_size_pixel = 1200

    # I believe the masks are:
    # necrosis: 1, contrast enhancement: 2, edema: 3.
    # For this script, we only care about the non-edema tumor.

    image_folder = '../../data/brain_LUMIERE/raw_images/'
    final_image_folder = '../../data/brain_LUMIERE/LUMIERE_images_tumor%dpx_%dx%d' % (min_tumor_size_pixel, out_shape[0], out_shape[1])
    final_mask_folder = '../../data/brain_LUMIERE/LUMIERE_masks_tumor%dpx_%dx%d' % (min_tumor_size_pixel, out_shape[0], out_shape[1])

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
        biggest_tumor_size = (np.logical_and(stacked_mask > 0, stacked_mask < 3)).sum(axis=(1, 3)).max()
        if biggest_tumor_size < min_tumor_size_pixel:
            continue

        # Perform registration towards the first time point in this series.
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

            affine_ants = ants.apply_transforms(fixed=fixed_ants,
                                                moving=moving_ants,
                                                transformlist=reg_affine['fwdtransforms'],
                                                interpolator='linear')
            affine_mask_ants = ants.apply_transforms(fixed=fixed_ants,
                                                     moving=moving_mask_ants,
                                                     transformlist=reg_affine['fwdtransforms'],
                                                     interpolator='nearestNeighbor')

            # Very mild diffeomorphic registration to avoid deminishing structural changes.
            reg_diffeomorphic = ants.registration(fixed_ants,
                                                  affine_ants,
                                                  'SyN',
                                                  reg_iterations=(4, 2, 1))

            diffeo_ants = ants.apply_transforms(fixed=fixed_ants,
                                                moving=affine_ants,
                                                transformlist=reg_diffeomorphic['fwdtransforms'],
                                                interpolator='linear')
            diffeo_mask_ants = ants.apply_transforms(fixed=fixed_ants,
                                                     moving=affine_mask_ants,
                                                     transformlist=reg_diffeomorphic['fwdtransforms'],
                                                     interpolator='nearestNeighbor')

            final_stacked_scan[scan_idx] = diffeo_ants.numpy().astype(np.uint8)
            final_stacked_mask[scan_idx] = diffeo_mask_ants.numpy().astype(np.uint8)

        # Receiver gain and offset correction.
        for scan_idx in range(1, len(scan_list)):
            union_tumor_mask = np.logical_or(final_stacked_mask[0] > 0,
                                             final_stacked_mask[scan_idx] > 0)
            final_stacked_scan[scan_idx] = correct_receiver_gain(final_stacked_scan[0],
                                                                 final_stacked_scan[scan_idx],
                                                                 union_tumor_mask)

        # Find the "interesting" slices.
        tumor_size_by_slice = \
            (np.logical_and(final_stacked_mask > 0, final_stacked_mask < 3)).sum(axis=(1, 3)).max(axis=0)
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

                final_mask = rescale_mask(final_mask)
                cv2.imwrite(out_fname_image, final_img)
                cv2.imwrite(out_fname_mask, final_mask)
