import argparse
import datetime
import multiprocessing
import os
from pathlib import Path

import torch

import global_config
from pytorch_training import Trainer
from pytorch_training.distributed.utils import synchronize, get_rank, get_world_size
from pytorch_training.extensions.logger import WandBLogger
from pytorch_training.extensions.lr_scheduler import LRScheduler
from pytorch_training.triggers import get_trigger
from training_builder.base_train_builder import BaseTrainBuilder
from training_builder.train_builder_selection import get_train_builder_class
from utils.clamped_cosine import ClampedCosineAnnealingLR
from utils.config import load_yaml_config, merge_config_and_args
from utils.data_loading import get_data_loader

if os.environ.get('REMOTE_PYCHARM_DEBUG_SESSION', False):
    import pydevd_pycharm
    pydevd_pycharm.settrace('localhost', port=int(os.environ.get('REMOTE_PYCHARM_DEBUG_PORT')),
                            stdoutToServer=True, stderrToServer=True, suspend=False)


def get_scheduler(config: dict, trainer: Trainer, training_builder: BaseTrainBuilder) -> LRScheduler:
    if 'cosine_max_update_epoch' in config:
        cosine_end_iteration = config['cosine_max_update_epoch'] * trainer.iterations_per_epoch
    elif 'cosine_max_update_iter' in config:
        cosine_end_iteration = config['cosine_max_update_iter']
    else:
        cosine_end_iteration = config['epochs']

    schedulers = {}
    for optimizer_name, optimizer in training_builder.get_optimizers().items():
        # TODO: Looks broken. At least for pixel ensembles it doesn't change the lr and always uses end_lr
        schedulers[optimizer_name] = ClampedCosineAnnealingLR(optimizer, cosine_end_iteration, eta_min=config['end_lr'])

    return LRScheduler(schedulers, trigger=get_trigger((1, 'iteration')))


def main(args: argparse.Namespace, rank: int, world_size: int):
    # TODO: clean up the messy mixture of using config and args - use config and only config everywhere
    config = load_yaml_config(args.config)
    config = merge_config_and_args(config, args)

    train_data_loader = get_data_loader(Path(config['train_json']), config['dataset'], args, config)
    if args.validation_json is not None:
        val_data_loader = get_data_loader(Path(config['validation_json']), config['dataset'], args, config,
                                          validation=True)
    else:
        val_data_loader = None

    train_builder_class = get_train_builder_class(config)
    training_builder = train_builder_class(config, train_data_loader, val_data_loader, rank=rank, world_size=world_size)

    if 'max_iter' in config:
        stop_trigger = (config['max_iter'], 'iteration')
    else:
        stop_trigger = (config['epochs'], 'epoch')

    trainer = Trainer(
        training_builder.get_updater(),
        stop_trigger=get_trigger(stop_trigger)
    )

    print("Initializing wandb... ", end='')
    logger = WandBLogger(
        args.log_dir,
        args,
        config,
        os.path.dirname(os.path.realpath(__file__)),
        trigger=get_trigger((config['log_iter'], 'iteration')),
        master=rank == 0,
        project_name=args.wandb_project_name,
        entity=args.wandb_entity,
        run_name=args.log_name,
        disabled=global_config.debug
    )
    print("done")
    # TODO: maybe make dummy Objects to circumvent all the ifs
    evaluator = training_builder.get_evaluator(logger)
    if evaluator is not None:
        # TODO: log confusion matrix: https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM
        trainer.extend(evaluator)

    snapshotter = training_builder.get_snapshotter()
    if snapshotter is not None:
        trainer.extend(snapshotter)

    image_plotter = training_builder.get_image_plotter()
    if image_plotter is not None:
        trainer.extend(image_plotter)

    lr_scheduler = get_scheduler(config, trainer, training_builder)
    trainer.extend(lr_scheduler)

    trainer.extend(logger)

    synchronize()
    print('Setup complete. Starting training...')
    trainer.train()


if __name__ == '__main__':
    print('Training script started')
    parser = argparse.ArgumentParser(description='Train a network for semantic segmentation of documents')
    parser.add_argument('config', help='path to config with common train settings, such as LR')
    parser.add_argument('--train', dest='train_json', required=True,
                        help='Path to json file with train images')
    parser.add_argument('--val', dest='validation_json', help='path to json file with validation images')
    parser.add_argument('--coco-gt', help='PAth to COCO GT required, if you set validation images')
    parser.add_argument('--fine-tune', help='Path to model to finetune from')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--mpi-backend', default='gloo', choices=['nccl', 'gloo'],
                        help='MPI backend to use for interprocess communication')
    parser.add_argument('--class-to-color-map', default='handwriting_colors.json',
                        help='path to json file with class color map')
    parser.add_argument('-c', '--cache-root',
                        help='path to a folder where you want to cache images on the local file system')
    parser.add_argument('-l', '--log-dir', default='training', help='outputs path')
    parser.add_argument('-ln', '--log-name', default='training', help='name of the train run')
    parser.add_argument('--warm-restarts', action='store_true', default=False,
                        help='If the scheduler should use warm restarts')
    parser.add_argument('--wandb-project-name', default='Debug', help='The project name of the WandB project')
    parser.add_argument('--wandb-entity', help='The name of the WandB entity')
    parser.add_argument('--debug', action='store_true', default=False, help='Special mode for faster debugging')

    parsed_args = parser.parse_args()
    parsed_args.log_dir = os.path.join('logs', parsed_args.log_dir, parsed_args.log_name,
                                       datetime.datetime.now().isoformat())
    global_config.debug = parsed_args.debug

    if torch.cuda.device_count() > 1:
        multiprocessing.set_start_method('forkserver')
        torch.cuda.set_device(parsed_args.local_rank)
        torch.distributed.init_process_group(backend=parsed_args.mpi_backend, init_method='env://')
        synchronize()

    main(parsed_args, get_rank(), get_world_size())
