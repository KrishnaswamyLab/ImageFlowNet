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


import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/external_src/SuperRetina/')

from common.common_util import pre_processing, simple_nms, remove_borders, \
    sample_keypoint_desc
from model.super_retina import SuperRetina

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def model_run(predict_config, model, batched_tensors, device):
    '''
    Input a batch with two images to SuperRetina
    '''
    batched_tensors = batched_tensors.to(device)

    with torch.no_grad():
        detector_pred, descriptor_pred = model(batched_tensors)

    scores = simple_nms(detector_pred, predict_config['nms_size'])

    b, _, h, w = detector_pred.shape
    scores = scores.reshape(-1, h, w)

    keypoints = [
        torch.nonzero(s > predict_config['nms_thresh'])
        for s in scores]

    scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

    # Discard keypoints near the image borders
    keypoints, scores = list(zip(*[
        remove_borders(k, s, 4, h, w)
        for k, s in zip(keypoints, scores)]))

    keypoints = [torch.flip(k, [1]).float().data for k in keypoints]

    descriptors = [sample_keypoint_desc(k[None], d[None], 8)[0].cpu().data
                for k, d in zip(keypoints, descriptor_pred)]
    keypoints = [k.cpu() for k in keypoints]

    return keypoints, descriptors


def match_kps(predict_config, moving_desc, fixed_desc) -> Tuple[List]:
    '''
    Match 2 sets of keypoints baed on their descriptors.
    '''
    goodMatch, status = [], []
    knn_matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = knn_matcher.knnMatch(moving_desc, fixed_desc, k=2)
    for m, n in matches:
        if m.distance < predict_config['knn_thresh'] * n.distance:
            goodMatch.append(m)
            status.append(True)
        else:
            status.append(False)
    return goodMatch, status


def find_homography(goodMatch: List, status: List, moving_cv_kpts, fixed_cv_kpts, num_matches_thr, verbose: bool = False):
    '''
    Find homography from 2 sets of OpenCV keypoints.
    '''
    H_m = None
    good = goodMatch.copy()

    if len(goodMatch) >= num_matches_thr:
        src_pts = [moving_cv_kpts[m.queryIdx].pt for m in good]
        src_pts = np.float32(src_pts).reshape(-1, 1, 2)
        dst_pts = [fixed_cv_kpts[m.trainIdx].pt for m in good]
        dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

        H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)

        good = np.array(good)[mask.ravel() == 1]
        status = np.array(status)
        temp = status[status==True]
        temp[mask.ravel() == 0] = False
        status[status==True] = temp

    if verbose:
        inliers_num_rate = mask.sum() / len(mask.ravel())
        print("The rate of inliers: {:.3f}%".format(inliers_num_rate * 100))

    return H_m


def map_image(H_m, moving_image: np.array, fixed_image_shape: Iterable[int]) -> np.array:
    '''
    Map the moving image to align with the fixed image using the given homography.
    '''
    h, w = fixed_image_shape[:2]
    moving_align = cv2.warpPerspective(moving_image, H_m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

    return moving_align


def drawMatches(imageA, imageB, kpsA, kpsB, matches, status):
    '''
    Draw the matching keypoints across 2 images.
    '''
    # initialize the output visualization image
    (hA, wA) = imageA.shape[:2]
    (hB, wB) = imageB.shape[:2]
    vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
    if len(imageA.shape) == 2:
        imageA = cv2.cvtColor(imageA, cv2.COLOR_GRAY2RGB)
        imageB = cv2.cvtColor(imageB, cv2.COLOR_GRAY2RGB)

    vis[0:hA, 0:wA] = imageA
    vis[0:hB, wA:] = imageB

    # loop over the matches
    for (match, _), s in zip(matches, status):
        trainIdx, queryIdx = match.trainIdx, match.queryIdx
        # only process the match if the keypoint was successfully
        # matched
        if s == 1:
            # draw the match
            ptA = (int(kpsA[queryIdx].pt[0]), int(kpsA[queryIdx].pt[1]))
            ptB = (int(kpsB[trainIdx].pt[0]) + wA, int(kpsB[trainIdx].pt[1]))
            cv2.line(vis, ptA, ptB, (0, 255, 0), 2)

        # return the visualization
    return vis


def register_and_save(predict_config, model, image_transformer, device,
                      base_folder_source: str,
                      base_mask_folder_source: str,
                      base_folder_target: str,
                      base_mask_folder_target: str):
    source_image_folders = sorted(glob(base_folder_source + '/*'))

    success, total = 0, 0

    for folder in tqdm(source_image_folders):
        image_list = sorted(glob(folder + '/*.png'))
        if len(image_list) <= 2:
            # Can ignore this folder if there is fewer than 2 images.
            pass

        # Batch process all images at once through the model.
        batched_tensor = None
        image_height, image_width = None, None
        for _image_path in image_list:
            # Read the image as grayscale.
            _image = cv2.imread(_image_path, cv2.IMREAD_GRAYSCALE)

            if image_height is not None:
                assert image_height == _image.shape[0]
                assert image_width == _image.shape[1]
            image_height, image_width = _image.shape[:2]

            # The image is grayscale.
            assert len(_image.shape) == 2
            _image = pre_processing(_image)
            _image = (_image * 255).astype(np.uint8)
            # scaled image size
            _tensor = image_transformer(Image.fromarray(_image))

            if batched_tensor is None:
                batched_tensor = _tensor[None, ...]
            else:
                batched_tensor = torch.cat((batched_tensor, _tensor[None, ...]))

        # Keypoints & Descriptor
        keypoints, descriptors = model_run(predict_config, model, batched_tensor, device)
        descriptors = [desc.permute(1, 0).numpy() for desc in descriptors]

        # Mapping keypoints to scaled keypoints.
        # 30 is keypoints size, which can be ignored
        cv_kpts = [[cv2.KeyPoint(int(i[0] / predict_config['model_image_width'] * image_width),
                                int(i[1] / predict_config['model_image_height'] * image_height), 30)
                        for i in kps] for kps in keypoints]

        assert len(image_list) == len(keypoints)
        assert len(image_list) == len(descriptors)
        assert len(image_list) == len(cv_kpts)

        # Now we can perform paired keypoint matching!
        # We will try to maximize the matchable image pairs per longitudinal sequence.
        # Let's use a matrix to track the # of matches.
        num_matches_matrix = np.zeros((len(image_list), len(image_list)), dtype=np.int16)
        for i in range(len(image_list)):
            for j in range(len(image_list)):
                goodMatch, _ = match_kps(predict_config, descriptors[i], descriptors[j])
                num_matches_matrix[i, j] = len(goodMatch)

        # Find the best fixed image (destination) for the matching.
        num_matches_matrix_bin = num_matches_matrix > predict_config['num_match_thr']
        fixed_idx = num_matches_matrix_bin.sum(axis=1).argmax()

        # First save the fixed image to the target folder.
        subject_folder_name = os.path.basename(folder)
        fixed_image_path = image_list[fixed_idx]
        fixed_image_name = os.path.basename(fixed_image_path)
        fixed_image = cv2.imread(fixed_image_path, cv2.IMREAD_COLOR)
        os.makedirs(base_folder_target + '/' + subject_folder_name + '/', exist_ok=True)
        cv2.imwrite(base_folder_target + '/' + subject_folder_name + '/' + fixed_image_name, fixed_image)

        # Also save the mask corresponding to the fixed image to the target folder.
        unique_identifier = '_'.join(os.path.basename(fixed_image_name).split('_')[:3])
        subject_folder_name = fixed_image_path.split('/')[-2]
        fixed_mask_path_list = glob(base_mask_folder_source + subject_folder_name + '/' + unique_identifier + '*_mask.png')
        for fixed_mask_path in fixed_mask_path_list:
            fixed_mask = cv2.imread(fixed_mask_path, cv2.IMREAD_GRAYSCALE)
            os.makedirs(base_mask_folder_target + '/' + subject_folder_name + '/', exist_ok=True)
            cv2.imwrite(base_mask_folder_target + '/' + subject_folder_name + '/' + os.path.basename(fixed_mask_path), fixed_mask)

        # Now register every image in the series to that fixed image.
        for i, moving_image_path in enumerate(image_list):
            if i == fixed_idx:
                continue
            moving_image_name = os.path.basename(moving_image_path)
            moving_image = cv2.imread(moving_image_path, cv2.IMREAD_COLOR)

            goodMatch, status = match_kps(predict_config, descriptors[i], descriptors[fixed_idx])
            H_m = find_homography(goodMatch, status, cv_kpts[i], cv_kpts[fixed_idx],
                                  num_matches_thr=predict_config['num_match_thr'], verbose=False)

            if H_m is not None:
                aligned_image = map_image(H_m, moving_image, fixed_image.shape)
                cv2.imwrite(base_folder_target + '/' + subject_folder_name + '/' + moving_image_name, aligned_image)

                unique_identifier = '_'.join(os.path.basename(moving_image_name).split('_')[:3])
                subject_folder_name = moving_image_path.split('/')[-2]
                moving_mask_path_list = glob(base_mask_folder_source + subject_folder_name + '/' + unique_identifier + '*_mask.png')
                for moving_mask_path in moving_mask_path_list:
                    moving_mask = cv2.imread(moving_mask_path, cv2.IMREAD_GRAYSCALE)
                    aligned_mask = map_image(H_m, moving_mask, fixed_image.shape)
                    os.makedirs(base_mask_folder_target + '/' + subject_folder_name + '/', exist_ok=True)
                    cv2.imwrite(base_mask_folder_target + '/' + subject_folder_name + '/' + os.path.basename(moving_mask_path), aligned_mask)

                success += 1

            else:
                print("Failed to align the two images! %s and %s" % (fixed_image_path, moving_image_path))

            total += 1


    print('Registration success rate: (%.2f%%) %d/%d' % (success/total*100, success, total))


def register_longitudinal(predict_config,
                          base_folder_source: str,
                          base_mask_folder_source: str,
                          base_folder_target: str,
                          base_mask_folder_target: str):
    '''
    Register the longitudinal images.

    For the case in `data/retina_ucsf/UCSF_images_512x512/`,
    each folder represents a series of longitudinal images to be registered.

    We also need to apply the transformation to the corresponding masks in
    `data/retina_ucsf/UCSF_masks_512x512/`,
    '''

    # SuperRetina config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loading Model
    checkpoint = torch.load(predict_config['model_save_path'], map_location=device)
    model = SuperRetina()
    model.load_state_dict(checkpoint['net'])
    model.to(device)

    image_transformer = transforms.Compose([
        transforms.Resize((predict_config['model_image_height'], predict_config['model_image_width'])),
        transforms.ToTensor()
    ])

    # Running Registration
    register_and_save(predict_config=predict_config,
                      model=model,
                      image_transformer=image_transformer,
                      device=device,
                      base_folder_source=base_folder_source,
                      base_mask_folder_source=base_mask_folder_source,
                      base_folder_target=base_folder_target,
                      base_mask_folder_target=base_mask_folder_target)


if __name__ == '__main__':
    config = {}
    config['model_save_path'] = '../../external_src/SuperRetina/save/SuperRetina.pth'
    config['model_image_width'] = 512
    config['model_image_height'] = 512
    config['use_matching_trick'] = False

    config['nms_size'] = 10
    config['nms_thresh'] = 0.1
    config['knn_thresh'] = 0.8
    config['num_match_thr'] = 15

    base_folder_source = '../../data/retina_ucsf/UCSF_images_512x512/'
    base_mask_folder_source = '../../data/retina_ucsf/UCSF_masks_512x512/'
    base_folder_target = '../../data/retina_ucsf/UCSF_images_aligned_512x512/'
    base_mask_folder_target = '../../data/retina_ucsf/UCSF_masks_aligned_512x512/'

    register_longitudinal(predict_config=config,
                          base_folder_source=base_folder_source,
                          base_mask_folder_source=base_mask_folder_source,
                          base_folder_target=base_folder_target,
                          base_mask_folder_target=base_mask_folder_target)
