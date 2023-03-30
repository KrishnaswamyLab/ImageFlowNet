from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, random_split


def split_dataset(dataset: Dataset,
                  splits: Tuple[float, ] = (0.8, 0.1, 0.1),
                  random_seed: int = 0) -> Tuple[Dataset, ]:
    """
    Splits data into non-overlapping datasets of given proportions.

    Either a "train/validation/test" split
    Or a "train/validation" split is supported.
    """
    assert len(splits) in [2, 3]

    splits = np.array(splits)
    splits = splits / np.sum(splits)

    n = len(dataset)
    if len(splits) == 2:
        val_size = int(splits[1] * n)
        train_size = n - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(random_seed))
        return train_set, val_set
    else:
        val_size = int(splits[1] * n)
        test_size = int(splits[2] * n)
        train_size = n - val_size - test_size
        train_set, val_set, test_set = random_split(
            dataset, [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(random_seed))
        return train_set, val_set, test_set


def split_indices(indices: List[int] = None,
                  splits: Tuple[float, ] = (0.8, 0.1, 0.1),
                  random_seed: int = 0):
    """
    Splits indices into non-overlapping subsets of given proportions.

    Either a "train/validation/test" split
    Or a "train/validation" split is supported.
    """
    assert len(splits) in [2, 3]

    splits = np.array(splits)
    splits = splits / np.sum(splits)

    rng = np.random.default_rng(seed=random_seed)
    indices = rng.permutation(indices)

    n = len(indices)
    if len(splits) == 2:
        val_size = int(splits[1] * n)
        train_size = n - val_size
        train_indices = sorted(indices[:train_size])
        val_indices = sorted(indices[train_size:])
        return train_indices, val_indices
    else:
        val_size = int(splits[1] * n)
        test_size = int(splits[2] * n)
        train_size = n - val_size - test_size
        train_indices = sorted(indices[:train_size])
        val_indices = sorted(indices[train_size:train_size + val_size])
        test_indices = sorted(indices[train_size + val_size:])
        return train_indices, val_indices, test_indices
