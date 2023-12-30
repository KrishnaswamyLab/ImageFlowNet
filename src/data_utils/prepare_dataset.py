from data_utils.extend import ExtendedDataset
from data_utils.split import split_indices
from datasets.retina_areds import RetinaAREDSDataset, RetinaAREDSSubset
from datasets.retina_ucsf import RetinaUCSFDataset, RetinaUCSFSubset, RetinaUCSFSegDataset, RetinaUCSFSegSubset
from datasets.synthetic import SyntheticDataset, SyntheticSubset
from torch.utils.data import DataLoader
from utils.attribute_hashmap import AttributeHashmap


def prepare_dataset(config: AttributeHashmap, transforms_list = [None, None, None]):
    # Read dataset.
    if config.dataset_name == 'retina_areds':
        dataset = RetinaAREDSDataset(base_path=config.dataset_path,
                                     image_folder=config.image_folder,
                                     eye_mask_folder=config.eye_mask_folder,
                                     target_dim=config.target_dim)
        Subset = RetinaAREDSSubset

    elif config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFDataset(base_path=config.dataset_path,
                                    image_folder=config.image_folder,
                                    target_dim=config.target_dim)
        Subset = RetinaUCSFSubset

    elif config.dataset_name == 'synthetic':
        dataset = SyntheticDataset(base_path=config.dataset_path,
                                   image_folder=config.image_folder,
                                   target_dim=config.target_dim)
        Subset = SyntheticSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    num_image_channel = dataset.num_image_channel()

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=config.random_seed)

    transforms_train, transforms_val, transforms_test = transforms_list
    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format=config.return_format,
                       transforms=transforms_train,
                       pos_neg_pairs=config.pos_neg_pairs)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_pairs',
                     transforms=transforms_val,
                     pos_neg_pairs=config.pos_neg_pairs)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_pairs',
                      transforms=transforms_test,
                      pos_neg_pairs=config.pos_neg_pairs)

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

    return train_set, val_set, test_set, num_image_channel


def prepare_dataset_segmentation(config: AttributeHashmap, transforms_list = [None, None, None]):
    # Read dataset.
    if config.dataset_name == 'retina_ucsf':
        dataset = RetinaUCSFSegDataset(base_path=config.dataset_path,
                                       image_folder=config.image_folder,
                                       mask_folder=config.mask_folder,
                                       target_dim=config.target_dim)
        Subset = RetinaUCSFSegSubset

    else:
        raise ValueError(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    num_image_channel = dataset.num_image_channel()

    # Load into DataLoader
    ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
    ratios = tuple([c / sum(ratios) for c in ratios])
    indices = list(range(len(dataset)))
    train_indices, val_indices, test_indices = \
        split_indices(indices=indices, splits=ratios, random_seed=config.random_seed)

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
    desired_len = max(len(train_set), min_sample_per_epoch)
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

    return train_set, val_set, test_set, num_image_channel
