import os
from typing import Dict, Union
from torch.nn.functional import one_hot

import torch
from pytorch_training.data.json_dataset import JSONDataset

from utils.augment_dataset import augment_image


class BaseDataset(JSONDataset):

    def __init__(self, *args, image_size: int = None, num_classes: int = 20, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_size = image_size
        self.num_classes = num_classes

    def load_json_data(self, json_data: Union[dict, list]):
        self.image_data = json_data

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.image_data[index]['file']
        if self.root is not None:
            path = os.path.join(self.root, path)  # TODO

        input_image = self.loader(path)
        input_image = self.transforms(input_image)

        label = int(self.image_data[index]['chosen'])

        return {
            'images': input_image,
            'labels': label
        }


class AugmentedDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.num_augmentations = kwargs.pop("num_augmentations")
        assert isinstance(self.num_augmentations, int), "num_augmentations must be an Integer"
        super().__init__(*args, **kwargs)

    def __len__(self) -> int:
        return self.num_augmentations * super().__len__()

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # TODO: balance according to dice distribution
        # Dataset was artificially inflated, so we have to shift the index back
        original_dataset_length = super().__len__()
        actual_index = index % original_dataset_length

        path = self.image_data[actual_index]['file']
        if self.root is not None:
            path = os.path.join(self.root, path)  # TODO

        input_image = self.loader(path)
        if index // original_dataset_length != 0:
            # After first iteration through the original dataset, return augmented images
            input_image = augment_image(input_image, num_images=1)[0]
        input_image = self.transforms(input_image)

        label = int(self.image_data[actual_index]['chosen'])

        l = torch.zeros((1,), dtype=torch.long)
        l[0] = label - 1
        return {
            'images': input_image,
            'labels': one_hot(l, num_classes=self.num_classes)
        }
