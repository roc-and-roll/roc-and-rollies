import torch
from pytorch_training import Updater
from pytorch_training.reporter import get_current_reporter
from pytorch_training.updater import GradientApplier
from torch import Tensor
from torch import nn


# TODO: adapt and clean
class StandardUpdater(Updater):
    """
    This updater is mainly used for the DocUFCN model
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = nn.CrossEntropyLoss()

    def update_core(self):
        batch = next(self.iterators['images'])
        batch = {key: value.to(self.device) for key, value in batch.items()}
        reporter = get_current_reporter()

        network = self.networks['segmentation']

        with GradientApplier([network], [self.optimizers['main']]):
            segmentation_prediction = network(batch['images'])
            batch_size, num_classes, height, width = segmentation_prediction.shape
            segmentation_prediction = segmentation_prediction.permute(0, 2, 3, 1)
            segmentation_prediction = torch.reshape(segmentation_prediction, (batch_size * height * width, num_classes))

            label_image = batch['segmented']
            label_image = label_image.permute(0, 2, 3, 1)
            label_image = label_image.reshape((-1,))

            loss = self.loss(segmentation_prediction, label_image)

            loss.backward()
        reporter.add_observation({"softmax": loss}, 'loss')


# class EMANetUpdater(Updater):
#     def __init__(self, *args, **kwargs):
#         self.em_mom = kwargs.pop('em_mom')
#         super().__init__(*args, **kwargs)
#
#     def update_core(self):
#         batch = next(self.iterators['images'])
#         batch = {key: value.to(self.device) for key, value in batch.items()}
#         reporter = get_current_reporter()
#
#         network = self.networks['segmentation']
#
#         loss, mu = network(batch['images'], torch.squeeze(batch['segmented'], dim=1))
#
#         with torch.no_grad():
#             mu = mu.mean(dim=0, keepdim=True)
#             network.emau.mu *= self.em_mom
#             network.emau.mu += mu * (1 - self.em_mom)
#
#         loss = loss.mean()
#         self.optimizers['main'].zero_grad()
#         loss.backward()
#         self.optimizers['main'].step()
#
#         reporter.add_observation({'softmax': loss}, 'loss')
#
#
# class TransUNetUpdater(Updater):
#     def __init__(self, *args, **kwargs):
#         num_classes = kwargs.pop('num_classes')
#         super().__init__(*args, **kwargs)
#         self.ce_loss = nn.CrossEntropyLoss()
#         self.dice_loss = DiceLoss(num_classes)
#
#     def update_core(self):
#         batch = next(self.iterators['images'])
#         batch = {key: value.to(self.device) for key, value in batch.items()}
#         reporter = get_current_reporter()
#
#         network = self.networks['segmentation']
#
#         with GradientApplier([network], [self.optimizers['main']]):
#             prediction = network(batch['images'])
#
#             ground_truth = torch.squeeze(batch['segmented'], dim=1)
#             loss_ce = self.ce_loss(prediction, ground_truth.long())
#             loss_dice = self.dice_loss(prediction, ground_truth, softmax=True)
#             loss = 0.5 * loss_ce + 0.5 * loss_dice
#
#             loss.backward()
#
#         reporter.add_observation(
#             {
#                 'combined': loss,
#                 'CE': loss_ce,
#                 'Dice': loss_dice,
#             }, 'loss'
#         )
