import numpy as np
from skimage.metrics import hausdorff_distance, structural_similarity


def psnr(image1, image2, max_value=1):
    '''
    Assuming data range is [0, 1].
    '''
    assert image1.shape == image2.shape

    eps = 1e-12

    mse = np.mean((image1 - image2)**2)
    return 20 * np.log10(max_value / np.sqrt(mse + eps))


def ssim(image1: np.array, image2: np.array, data_range=1, **kwargs) -> float:
    '''
    Please make sure the data are provided in [H, W, C] shape.

    Assuming data range is [0, 1] --> `data_range` = 1.
    '''
    assert image1.shape == image2.shape

    H, W = image1.shape[:2]

    if min(H, W) < 7:
        win_size = min(H, W)
        if win_size % 2 == 0:
            win_size -= 1
    else:
        win_size = None

    if len(image1.shape) == 3:
        channel_axis = -1
    else:
        channel_axis = None

    return structural_similarity(image1,
                                 image2,
                                 data_range=data_range,
                                 channel_axis=channel_axis,
                                 win_size=win_size,
                                 **kwargs)


def dice_coeff(label_pred: np.array, label_true: np.array) -> float:
    epsilon = 1e-12
    intersection = np.logical_and(label_pred, label_true).sum()
    dice = (2 * intersection + epsilon) / (label_pred.sum() +
                                           label_true.sum() + epsilon)
    return dice


def hausdorff(label_pred: np.array, label_true: np.array) -> float:
    if np.sum(label_pred) == 0 or np.sum(label_true) == 0:
        # If `label_pred` or `label_true` is all zeros,
        # return the max Euclidean distance.
        H, W = label_true.shape
        return np.sqrt((H**2 + W**2))
    else:
        return hausdorff_distance(label_pred, label_true)
