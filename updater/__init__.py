import torch
from pytorch_training import Updater
from pytorch_training.reporter import get_current_reporter
from pytorch_training.updater import GradientApplier
from torch import Tensor
from torch import nn


# TODO: adapt and clean
class StandardUpdater(Updater):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def update_core(self):
        batch = next(self.iterators['images'])
        batch = {key: value.to(self.device) for key, value in batch.items()}
        images = batch['images'].to(self.device)
        labels = batch['labels']
        reporter = get_current_reporter()

        network = self.networks['network']

        with GradientApplier([network], [self.optimizers['main']]):
            prediction = network(images)
            loss = self.loss(prediction, labels)
            loss.backward()
        reporter.add_observation({"cross_entropy": loss}, 'loss')
