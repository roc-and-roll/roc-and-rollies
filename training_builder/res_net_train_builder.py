from typing import Dict

import torch
from pytorch_training import Updater
from pytorch_training.optimizer import GradientClipAdam
from torch.optim import Optimizer
from torchvision import models

import global_config
from training_builder.base_train_builder import BaseSingleNetworkTrainBuilder
from updater import StandardUpdater


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
        self.network = models.resnet18(pretrained=True)  # TODO: maybe move to BaseNetwork

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
