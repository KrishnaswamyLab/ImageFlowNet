from data_utils.extend import ExtendedDataset
from data_utils.split import split_indices
from datasets.retina_areds import RetinaAREDSDataset, RetinaAREDSSubset
from datasets.retina_ucsf import RetinaUCSFDataset, RetinaUCSFSubset
from datasets.synthetic import SyntheticDataset, SyntheticSubset
from torch.utils.data import DataLoader
from utils.attribute_hashmap import AttributeHashmap


def prepare_dataset(config: AttributeHashmap):
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

    train_set = Subset(main_dataset=dataset,
                       subset_indices=train_indices,
                       return_format=config.return_format,
                       pos_neg_pairs=config.pos_neg_pairs)
    val_set = Subset(main_dataset=dataset,
                     subset_indices=val_indices,
                     return_format='all_pairs',
                     pos_neg_pairs=config.pos_neg_pairs)
    test_set = Subset(main_dataset=dataset,
                      subset_indices=test_indices,
                      return_format='all_pairs',
                      pos_neg_pairs=config.pos_neg_pairs)

    min_batch_per_epoch = 5
    desired_len = max(len(train_set), min_batch_per_epoch)
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
