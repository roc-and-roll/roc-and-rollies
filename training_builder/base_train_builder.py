import functools
from typing import Dict, List, Union

from pytorch_training import Updater
from pytorch_training.distributed.utils import strip_parallel_module
from pytorch_training.extensions import ImagePlotter, Snapshotter, Evaluator, Logger
from pytorch_training.triggers import get_trigger
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import torch

import global_config
from networks import load_weights
from networks.base_network import BaseNetwork
from utils.data_loading import fill_plot_images
from visualization.plotter import DicePlotter


class BaseTrainBuilder:
    def __init__(self, config: dict, train_data_loader: Union[DataLoader, None] = None,
                 val_data_loader: Union[DataLoader, None] = None, rank: int = 0, world_size: int = 1):
        self.network = None
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.config = config
        self.fine_tune = config['fine_tune']
        self.rank = rank
        self.world_size = world_size

    def _prepare_network(self, network: BaseNetwork, network_name: str = 'network') -> BaseNetwork:
        assert network is not None, 'Network was not properly initialized!'
        network.to(global_config.device)
        if self.fine_tune is not None:
            load_weights(network, self.fine_tune, key=network_name)

        if self.world_size > 1:
            distributed = functools.partial(DDP, device_ids=[self.rank], find_unused_parameters=True,
                                            broadcast_buffers=False, output_device=self.rank)
            network = distributed(network)
        return network

    def _initialize_network(self):
        raise NotImplementedError

    def get_network(self) -> BaseNetwork:
        return self.network

    def get_networks_for_updater(self) -> Dict[str, BaseNetwork]:
        raise NotImplementedError

    def get_optimizers(self) -> Dict[str, Optimizer]:
        raise NotImplementedError

    def get_updater(self) -> Updater:
        raise NotImplementedError

    def get_snapshotter(self) -> Union[Snapshotter, None]:
        raise NotImplementedError

    def get_evaluator(self, logger: Logger) -> Union[Evaluator, None]:
        return None

    def get_image_plotter(self) -> Union[ImagePlotter, None]:
        if self.rank != 0:
            return None
        plot_data_loader = self.val_data_loader if self.val_data_loader is not None else self.train_data_loader
        predictions = fill_plot_images(plot_data_loader, num_desired_images=self.config['display_size'])
        image_plotter = DicePlotter(
            predictions['images'],
            [strip_parallel_module(self.network)],
            self.config['log_dir'],
            labels=predictions['labels'],
            trigger=get_trigger((self.config['image_save_iter'], 'iteration')),
            plot_to_logger=True,
        )
        return image_plotter


class BaseSingleNetworkTrainBuilder(BaseTrainBuilder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_networks_for_updater(self) -> Dict[str, BaseNetwork]:
        return {'network': self.network}

    def get_snapshotter(self) -> Union[Snapshotter, None]:
        if self.rank != 0:
            return None
        snapshotter = Snapshotter(
            {
                'network': strip_parallel_module(self.network),
                **self.get_optimizers(),
            },
            self.config['log_dir'],
            trigger=get_trigger((self.config['snapshot_save_iter'], 'iteration'))
        )
        return snapshotter
