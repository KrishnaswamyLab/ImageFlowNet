from data_utils.extend import ExtendedDataset
from data_utils.split import split_indices
from datasets.retina_areds import RetinaAREDSDataset, RetinaAREDSSubset
from datasets.retina_ucsf import RetinaUCSFDataset, RetinaUCSFSubset, RetinaUCSFSegDataset, RetinaUCSFSegSubset
from datasets.brain_ms import BrainMSDataset, BrainMSSubset, BrainMSSegDataset, BrainMSSegSubset, brain_MS_split
from datasets.brain_gbm import BrainGBMDataset, BrainGBMSubset, BrainGBMSegDataset, BrainGBMSegSubset
from datasets.synthetic import SyntheticDataset, SyntheticSubset
from torch.utils.data import DataLoader
from utils.attribute_hashmap import AttributeHashmap


def prepare_dataset(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for predicting one future timepoint from one earlier timepoint.
    '''

    # Read dataset.
    if config.dataset_name == 'retina_areds':
        dataset = RetinaAREDSDataset(eye_mask_folder=config.eye_mask_folder)
        Subset = RetinaAREDSSubset

    elif config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.target_dim)
        Subset = BrainGBMSubset

    elif config.dataset_name == 'synthetic':
        dataset = SyntheticDataset(base_path=config.dataset_path,
                                   image_folder=config.image_folder,
                                   target_dim=config.target_dim)
        Subset = SyntheticSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_aug = None
    if len(transforms_list) == 4:
        transforms_train, transforms_val, transforms_test, transforms_aug = transforms_list
    else:
        transforms_train, transforms_val, transforms_test = transforms_list

    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='one_pair',
                       transforms=transforms_train,
                       transforms_aug=transforms_aug)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_pairs',
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_pairs',
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set), min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=1,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_npt(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for predicting one future timepoint from potentially multiple earlier timepoints.
    '''

    # Read dataset.
    if config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.target_dim)
        Subset = BrainGBMSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='all_subsequences',
                       transforms=transforms_train)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_subsequences',
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_subsequences',
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set), min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=1,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_full_sequence(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for iterating over all full sequences.
    '''

    # Read dataset.
    if config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.target_dim)
        Subset = BrainGBMSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    full_set = Subset(main_dataset=dataset,
                      subset_indices=indices,
                      return_format='full_sequence',
                      transforms=None)

    full_set = DataLoader(dataset=full_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return full_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_all_subarrays(config: AttributeHashmap, transforms_list = [None, None, None]):
    '''
    Prepare the dataset for iterating over all subarrays.
    This means we will not consider subsequences with dropped out intermediate values.
    '''

    # Read dataset.
    if config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSDataset(target_dim=config.target_dim)
        Subset = BrainMSSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMDataset(target_dim=config.target_dim)
        Subset = BrainGBMSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format='all_subarrays',
                       transforms=transforms_train)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_subarrays',
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_subarrays',
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set), min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=1,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=1,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=1,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel(), dataset.max_t


def prepare_dataset_segmentation(config: AttributeHashmap, transforms_list = [None, None, None]):
    # Read dataset.
    if config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFSegDataset(target_dim=config.target_dim)
        Subset = RetinaUCSFSegSubset

    elif config.dataset_name == 'brain_ms':
        dataset = BrainMSSegDataset(target_dim=config.target_dim)
        Subset = BrainMSSegSubset

    elif config.dataset_name == 'brain_gbm':
        dataset = BrainGBMSegDataset(target_dim=config.target_dim)
        Subset = BrainGBMSegSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=1)

    if config.dataset_name == 'brain_ms':
        train_indices, val_indices, test_indices = brain_MS_split()

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       transforms=transforms_train)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     transforms=transforms_val)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      transforms=transforms_test)

    min_sample_per_epoch = 5
    if 'max_training_samples' in config.keys():
        min_sample_per_epoch = config.max_training_samples
    desired_len = max(len(train_set) / config.batch_size, min_sample_per_epoch)
    train_set = ExtendedDataset(dataset=train_set, desired_len=desired_len)

    train_set = DataLoader(dataset=train_set,
                           batch_size=config.batch_size,
                           shuffle=True,
                           num_workers=config.num_workers)
    val_set = DataLoader(dataset=val_set,
                         batch_size=config.batch_size,
                         shuffle=False,
                         num_workers=config.num_workers)
    test_set = DataLoader(dataset=test_set,
                          batch_size=config.batch_size,
                          shuffle=False,
                          num_workers=config.num_workers)

    return train_set, val_set, test_set, dataset.num_image_channel()
