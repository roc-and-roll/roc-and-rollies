import argparse
import functools
import os
from collections import defaultdict
from pathlib import Path
from typing import Union, Dict, Iterable, Type, Callable, List

import torch
from PIL import Image
from pytorch_training.data.caching_loader import CachingLoader
from pytorch_training.data.json_dataset import JSONDataset
from pytorch_training.data.utils import default_loader
from pytorch_training.distributed import get_world_size, get_rank
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

import global_config
from data.dataset import AugmentedDataset


def resilient_loader(path):
    try:
        return default_loader(path)
    except Exception as e:
        print(f"Could not load {path} with expeption: {e}")
        return Image.new('RGB', (256, 256))

def get_transforms(image_size, input_dim):
    transform_list = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * input_dim, (0.5,) * input_dim)  # TODO: ok normalizazion
    ]
    transform_list = transforms.Compose(transform_list)
    return transform_list

def build_data_loader(image_path: Union[str, Path], config: dict, uses_absolute_paths: bool,
                      dataset_class: Type[JSONDataset], shuffle_off: bool = False,
                      loader_func: Callable = resilient_loader, drop_last: bool = True, collate_func: Callable = None) -> DataLoader:
    transform_list = get_transforms(config['image_size'], config['input_dim'])

    dataset = dataset_class(
        image_path,
        root=os.path.dirname(image_path) if not uses_absolute_paths else None,
        transforms=transform_list,
        loader=loader_func,
    )

    sampler = None
    if get_world_size() > 1:
        sampler = DistributedSampler(dataset, shuffle=not shuffle_off)
        sampler.set_epoch(get_rank())

    if shuffle_off:
        shuffle = False
    else:
        shuffle = sampler is None

    if not global_config.debug:
        data_loader_class = functools.partial(DataLoader, num_workers=config['num_workers'])
    else:
        # Specifying 'num_workers' may have a negative impact on debugging capabilities of PyCharm
        data_loader_class = DataLoader

    loader = data_loader_class(
        dataset,
        config['batch_size'],
        shuffle=shuffle,
        drop_last=drop_last,
        sampler=sampler,
        collate_fn=collate_func,
    )
    return loader

def get_data_loader(dataset_json_path: Path, dataset_name: str, args: argparse.Namespace, config: dict,
                    validation: bool = False) -> DataLoader:
    # TODO: refactor this function and called functions so that only one of args and config is needed
    if args.cache_root is not None:
        loader_func = CachingLoader(dataset_json_path.parent, args.cache_root, base_loader=resilient_loader)
    else:
        loader_func = resilient_loader

    if dataset_name == 'dice':
        dataset_class = functools.partial(AugmentedDataset, image_size=config['image_size'],
                                          num_augmentations=config['num_augmentations'])
        data_loader = build_data_loader(dataset_json_path, config, False, dataset_class=dataset_class,
                                        shuffle_off=validation, drop_last=(not validation), loader_func=loader_func)
    else:
        raise NotImplementedError

    return data_loader


def fill_plot_images(data_loader: Iterable, num_desired_images: int = 16) -> Dict[str, List[torch.Tensor]]:
    """
        Gathers images to be used with ImagePlotter
    """
    image_list = defaultdict(list)
    for batch in data_loader:
        for image_key, images in batch.items():
            num_images = 0
            for image in images:
                image_list[image_key].append(image)
                num_images += 1
                if num_images >= num_desired_images:
                    break
            if len(image_list.keys()) == len(batch.keys()) and \
                    all([len(v) >= num_desired_images for v in image_list.values()]):
                return image_list
    raise RuntimeError(f"Could not gather enough plot images for display size {num_desired_images}.")
