import os
import sys
from glob import glob
import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import_dir = '/'.join(os.path.realpath(__file__).split('/')[:-3])
sys.path.insert(0, import_dir + '/external_src/SuperRetina/')

from common.common_util import pre_processing, simple_nms, remove_borders, \
    sample_keypoint_desc
from model.super_retina import SuperRetina

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def model_run(predict_config, model, moving_tensor, fixed_tensor, device):
    '''
    Input a batch with two images to SuperRetina
    '''
    inputs = torch.cat((moving_tensor.unsqueeze(0), fixed_tensor.unsqueeze(0)))
    inputs = inputs.to(device)

    with torch.no_grad():
        detector_pred, descriptor_pred = model(inputs)

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

    # moving_mask = np.zeros((512, 512))
    # moving_mask[200:, 200:] = 1
    # fixed_mask = np.zeros((512, 512))
    # fixed_mask[:400, :400] = 1

    # retain_idx = []
    # for i, kp in enumerate(keypoints[0]):
    #     if moving_mask[int(kp[0]), int(kp[1])] == 1:
    #         retain_idx.append(i)
    # keypoints[0] = keypoints[0][retain_idx]
    # descriptors[0] = descriptors[0][:, retain_idx]

    # retain_idx = []
    # for i, kp in enumerate(keypoints[1]):
    #     if fixed_mask[int(kp[0]), int(kp[1])] == 1:
    #         retain_idx.append(i)
    # keypoints[1] = keypoints[1][retain_idx]
    # descriptors[1] = descriptors[1][:, retain_idx]

    return keypoints, descriptors


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


def register_and_save(predict_config, model, image_transformer, device, base_folder_source: str, base_folder_target: str):
    source_image_folders = sorted(glob(base_folder_source + '/*'))

    for folder in tqdm(source_image_folders):
        image_list = sorted(glob(folder + '/*.jpg'))
        if len(image_list) <= 2:
            # Can ignore this folder if there is fewer than 2 images.
            pass

        fixed_image_path = image_list[0]

        subject_folder_name = os.path.basename(folder)
        fixed_image_name = os.path.basename(fixed_image_path)

        for moving_image_path in image_list[1:]:
            moving_image_name = os.path.basename(moving_image_path)

            moving_image = cv2.imread(moving_image_path, cv2.IMREAD_COLOR)
            fixed_image = cv2.imread(fixed_image_path, cv2.IMREAD_COLOR)

            assert moving_image.shape == fixed_image.shape

            image_height, image_width = fixed_image.shape[:2]

            # Use the green channel as model input
            moving_image = moving_image[:, :, 1]
            moving_image = pre_processing(moving_image)

            fixed_image = fixed_image[:, :, 1]
            fixed_image = pre_processing(fixed_image)

            moving_image = (moving_image * 255).astype(np.uint8)
            fixed_image = (fixed_image * 255).astype(np.uint8)

            # scaled image size
            moving_tensor = image_transformer(Image.fromarray(moving_image))
            fixed_tensor = image_transformer(Image.fromarray(fixed_image))

            ################################# Detector + Descriptor #################################
            keypoints, descriptors = model_run(predict_config, model, moving_tensor, fixed_tensor, device)
            moving_keypoints, fixed_keypoints = keypoints[0], keypoints[1]
            moving_desc, fixed_desc = descriptors[0].permute(1, 0).numpy(), descriptors[1].permute(1, 0).numpy()

            # mapping keypoints to scaled keypoints
            # 30 is keypoints size, which can be ignored
            cv_kpts_moving = [cv2.KeyPoint(int(i[0] / predict_config['model_image_width'] * image_width),
                                        int(i[1] / predict_config['model_image_height'] * image_height), 30)
                            for i in moving_keypoints]
            cv_kpts_fixed = [cv2.KeyPoint(int(i[0] / predict_config['model_image_width'] * image_width),
                                        int(i[1] / predict_config['model_image_height'] * image_height), 30)
                            for i in fixed_keypoints]


            ################################# Keypoint Matching #################################
            goodMatch = []
            status = []
            try:
                knn_matcher = cv2.BFMatcher(cv2.NORM_L2)
                matches = knn_matcher.knnMatch(moving_desc, fixed_desc, k=2)
                for m, n in matches:
                    if m.distance < predict_config['knn_thresh'] * n.distance:
                        goodMatch.append(m)
                        status.append(True)
                    else:
                        status.append(False)
            except Exception:
                pass

            ################################# Find Homography #################################
            H_m = None
            inliers_num_rate = 0
            good = goodMatch.copy()
            if len(goodMatch) >= 10:
                src_pts = [cv_kpts_moving[m.queryIdx].pt for m in good]
                src_pts = np.float32(src_pts).reshape(-1, 1, 2)
                dst_pts = [cv_kpts_fixed[m.trainIdx].pt for m in good]
                dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

                H_m, mask = cv2.findHomography(src_pts, dst_pts, cv2.LMEDS)

                good = np.array(good)[mask.ravel() == 1]
                status = np.array(status)
                temp = status[status==True]
                temp[mask.ravel() == 0] = False
                status[status==True] = temp
                inliers_num_rate = mask.sum() / len(mask.ravel())

            # print("The rate of inliers: {:.3f}%".format(inliers_num_rate * 100))

            ################################# Map Image #################################
            if H_m is not None:
                h, w = image_height, image_width
                moving_align = cv2.warpPerspective(moving_image, H_m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

                merged = np.zeros((h, w, 3), dtype=np.uint8)

                if len(moving_align.shape) == 3:
                    moving_align = cv2.cvtColor(moving_align, cv2.COLOR_BGR2GRAY)
                if len(fixed_image.shape) == 3:
                    fixed_gray = cv2.cvtColor(fixed_image, cv2.COLOR_BGR2GRAY)
                else:
                    fixed_gray = fixed_image
                merged[:, :, 0] = moving_align
                merged[:, :, 1] = fixed_gray

            else:
                print("Failed to align the two images! %s and %s" % (fixed_image_path, moving_image_path))
                continue

            if H_m is not None:
                fixed_image = cv2.imread(fixed_image_path, cv2.IMREAD_COLOR)
                moving_image = cv2.imread(moving_image_path, cv2.IMREAD_COLOR)
                h, w = image_height, image_width
                moving_align = cv2.warpPerspective(moving_image, H_m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0))
                os.makedirs(base_folder_target + '/' + subject_folder_name + '/', exist_ok=True)
                cv2.imwrite(base_folder_target + '/' + subject_folder_name + '/' + fixed_image_name, fixed_image)
                cv2.imwrite(base_folder_target + '/' + subject_folder_name + '/' + moving_image_name, moving_align)


def register_longitudinal(predict_config, base_folder_source: str, base_folder_target: str):
    '''
    Register the longitudinal images onto the earliest image.

    For the case in `data/retina_areds/AREDS_2014_images_512x512/`,
    each folder represents a series of longitudinal images to be registered.
    '''

    # SuperRetina config
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ###################################### Loading Model ######################################
    checkpoint = torch.load(predict_config['model_save_path'], map_location=device)
    model = SuperRetina()
    model.load_state_dict(checkpoint['net'])
    model.to(device)

    image_transformer = transforms.Compose([
        transforms.Resize((predict_config['model_image_height'], predict_config['model_image_width'])),
        transforms.ToTensor()
    ])

    #################################### Running Registration ###################################
    register_and_save(predict_config=predict_config,
                      model=model,
                      image_transformer=image_transformer,
                      device=device,
                      base_folder_source=base_folder_source,
                      base_folder_target=base_folder_target)


if __name__ == '__main__':
    config = {}
    config['model_save_path'] = '../../external_src/SuperRetina/save/SuperRetina.pth'
    config['model_image_width'] = 512
    config['model_image_height'] = 512
    config['use_matching_trick'] = False

    config['nms_size'] = 10
    config['nms_thresh'] = 0.1
    config['knn_thresh'] = 0.8

    base_folder_source = '../../data/retina_areds/AREDS_2014_images_512x512/'
    base_folder_target = '../../data/retina_areds/AREDS_2014_images_aligned_512x512/'

    register_longitudinal(predict_config=config,
                          base_folder_source=base_folder_source,
                          base_folder_target=base_folder_target)
