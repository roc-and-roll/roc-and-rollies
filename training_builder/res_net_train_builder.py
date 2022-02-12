from typing import Dict

import torch
from pytorch_training import Updater
from pytorch_training.optimizer import GradientClipAdam
from torch.optim import Optimizer
from torchvision import models

import global_config
from training_builder.base_train_builder import BaseSingleNetworkTrainBuilder
from updater import StandardUpdater


class ResNet18(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.network = models.resnet18(pretrained=True)
        self.network.fc = torch.nn.Linear(in_features=512, out_features=num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network.forward(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.softmax(self.forward(x), dim=1)


class ResNetTrainBuilder(BaseSingleNetworkTrainBuilder):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_network()
        self.network = self._prepare_network(self.network)
        self.optimizer_opts = {
            'betas': (self.config['beta1'], self.config['beta2']),
            'weight_decay': self.config['weight_decay'],
            'lr': float(self.config['lr']),
        }

    def _initialize_network(self):
        self.network = ResNet18(self.config['num_classes'])

    def get_optimizers(self) -> Dict[str, Optimizer]:
        optimizer = GradientClipAdam(self.network.parameters(), **self.optimizer_opts)  # TODO: probs alright?
        return {'main': optimizer}

    def get_updater(self) -> Updater:
        updater = StandardUpdater(
            iterators={'images': self.train_data_loader},
            networks=self.get_networks_for_updater(),
            optimizers=self.get_optimizers(),
            device=global_config.device,
            copy_to_device=(self.world_size == 1)
        )
        return updater
