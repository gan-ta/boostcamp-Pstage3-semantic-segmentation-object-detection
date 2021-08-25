import os
import argparse
import json
import yaml
import random

import numpy as np

import wandb
import torch
import torch.nn as nn

from mmcv import Config
os.sys.path.append(os.path.abspath(r'./mmdetection_trash/'))
# os.sys.path.append(os.path.abspath(r'./Swin-Transformer-Object-Detection/'))
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.models import build_detector
from mmdet.apis import train_detector


def set_random_seed(seed: int, deterministic: bool = False) -> None:
    """
    Set random seeds for reproducibility.
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(args: argparse.Namespace, wandb_init_cfg: dict) -> int:
    """
    Main
    """

    """ Configuration """
    # Random seed
    set_random_seed(args.seed)
    # MMDetection config
    cfg = Config.fromfile(args.config)

    # Paths and directories to save results
    if cfg.resume_from is not None:
        exp_dir = os.path.dirname(cfg.resume_from)
    elif wandb_init_cfg['name'] is None or wandb_init_cfg['use_sweep']:
        exp_dir = os.path.join(args.save_dir, r'_NEW_EXPERIMENT')
        os.makedirs(exp_dir, exist_ok=True)
    else:
        exp_dir = os.path.join(args.save_dir, wandb_init_cfg['name'])
        try:
            os.makedirs(exp_dir, exist_ok=False)
        except OSError as err:
            print(f'[ERROR: {err}] The directory exists. Check the directory to save checkpoints.')
            return 1

    """ MMDetection """
    cfg.seed = args.seed
    cfg.gpu_ids = [0]

    # W&B
    cfg.log_config.hooks[1].init_kwargs.project = wandb_init_cfg['project']
    cfg.log_config.hooks[1].init_kwargs.entity = wandb_init_cfg['entity']
    cfg.log_config.hooks[1].init_kwargs.group = wandb_init_cfg['group']
    cfg.log_config.hooks[1].init_kwargs.job_type = wandb_init_cfg['job_type']
    cfg.log_config.hooks[1].init_kwargs.tags = wandb_init_cfg['tags']
    cfg.log_config.hooks[1].init_kwargs.name = wandb_init_cfg['name']
    cfg.log_config.hooks[1].init_kwargs.notes = wandb_init_cfg['notes']
    if wandb_init_cfg['use_sweep']:
        cfg.checkpoint_config.max_keep_ckpts = 1
        cfg.checkpoint_config.interval = 99999999
        cfg.evaluation.save_best = None
    cfg.work_dir = exp_dir

    # DEBUG: Training for debugging
    if args.debug:
        path, ext = os.path.splitext(cfg.data.train.ann_file)
        cfg.data.train.ann_file = path + r'_dev' + ext

        path, ext = os.path.splitext(cfg.data.val.ann_file)
        cfg.data.val.ann_file = path + r'_dev' + ext
        
        path, ext = os.path.splitext(cfg.data.test.ann_file)
        cfg.data.test.ann_file = path + r'_dev' + ext

    # Experiments using a smaller dataset
    if cfg.n_train_data == 3272:
        cfg.checkpoint_config.max_keep_ckpts = 10
        cfg.checkpoint_config.interval = 1
        cfg.evaluation.save_best = None
    elif cfg.n_train_data != 2617:
        path, ext = os.path.splitext(cfg.data.train.ann_file)
        cfg.data.train.ann_file = path + f'_{cfg.n_train_data}' + ext
        if cfg.n_train_data == 1024:
            cfg.log_config.interval = cfg.steps_per_epoch // 4
        elif cfg.n_train_data == 512:
            cfg.log_config.interval = cfg.steps_per_epoch // 2
        cfg.checkpoint_config.max_keep_ckpts = 1
        cfg.checkpoint_config.interval = 99999999
        cfg.evaluation.save_best = None

    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        datasets.append(build_dataset(cfg.data.val))

    model = build_detector(cfg.model)

    train_detector(model, datasets, cfg, distributed=False, validate=True)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,
                        default=2020, help='Random seed')
    parser.add_argument('-c', '--config', type=str,
                        default=None, help='Configuration file path')
    parser.add_argument('-w', '--wandb', type=str,
                        default=r'./configs/wandb.yaml', help='Configuration file path to initialize wandb')
    parser.add_argument('-s', '--save_dir', type=str,
                        default=r'../results/', help='Directory to save')
    parser.add_argument('-d', '--debug', type=bool,
                        default=False, help='Debugging mode')
    args = parser.parse_args()

    return args


def load_wandb_init_cfg(path: str) -> dict:
    """
    Load configuration to initialize W&B.
    """

    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    return cfg


if __name__ == '__main__':
    args = parse_args()
    wandb_init_cfg = load_wandb_init_cfg(args.wandb)

    main(args, wandb_init_cfg)
