from data_utils.extend import ExtendedDataset
from data_utils.split import split_dataset
from datasets.retina_GA import RetinaGA
from torch.utils.data import DataLoader
from utils.attribute_hashmap import AttributeHashmap


def prepare_dataset(config: AttributeHashmap, mode: str = 'train'):
    # Read dataset.
    if config.dataset_name == 'retina_GA':
        dataset = RetinaGA(base_path=config.dataset_path,
                           target_dim=config.target_dim,
                           sample_pairs=False,
                           train_identity=config.train_identity)
    else:
        raise Exception(
            'Dataset not found. Check `dataset_name` in config yaml file.')

    num_image_channel = dataset.num_image_channel()

    # Load into DataLoader
    if mode == 'train':
        ratios = [float(c) for c in config.train_val_test_ratio.split(':')]
        ratios = tuple([c / sum(ratios) for c in ratios])
        train_set, val_set, test_set = split_dataset(dataset=dataset,
                                                     splits=ratios,
                                                     random_seed=config.random_seed)

        min_batch_per_epoch = 5
        desired_len = max(len(train_set),
                          config.batch_size * min_batch_per_epoch)
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
