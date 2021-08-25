""" Base """
import os
import sys
import datetime
from time import time
from glob import glob
from pprint import pprint
from tqdm import tqdm
from sklearn.utils import Bunch
import random
import json
import argparse
import logging
import warnings
warnings.filterwarnings('ignore')

""" Data Science """
import numpy as np

""" Computer Vision """
# import cv2 as cv

""" AI """
import wandb
# PyTorch
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from adamp import AdamP

""" Custom Modules """
sys.path.append(os.path.abspath(r'./baseline_code/'))
sys.path.append(os.path.abspath(r'./'))
sys.path.append(os.path.abspath(r'../../'))

from dataloader.image import *
from model.deepLabV3_effib7_ver2 import *
from util.loss import *
from util.utils import *
from util.augmentation import *
from util.scheduler import *


from segmentation_models.segmentation_models_pytorch.losses import SoftCrossEntropyLoss, FocalLoss

# from RMI.losses.rmi.rmi import * # 외부 라이브러리 import
# from _hrnet.loss.rmi import RMILoss
# from rmi import RMILoss


""" Paths """
data_dir = r'/opt/ml/input/data/'
train_data_path = os.path.join(data_dir, r'train.json')
val_data_path = os.path.join(data_dir, r'val.json')
test_data_path = os.path.join(data_dir, r'test.json')


class ParameterError(Exception):
    def __init__(self):
        super().__init__('Enter essential parameters')


def __get_logger():
    """
    로거 인스턴스 반환
    """

    __logger = logging.getLogger('logger')

    # # 로그 포멧 정의
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # 스트림 핸들러 정의
    stream_handler = logging.StreamHandler()
    # 각 핸들러에 포멧 지정
    stream_handler.setFormatter(formatter)
    # 로거 인스턴스에 핸들러 삽입
    __logger.addHandler(stream_handler)
    # 로그 레벨 정의
    __logger.setLevel(logging.DEBUG)

    return __logger
logger = __get_logger()


def seed_everything(seed: int):
    """
    Set random seeds for reproducibility
    """

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    return tuple(zip(*batch))


def save_model(model_state_dict, save_dir, filename):
    # checkpoint = {'net': model_state_dict}
    torch.save(model_state_dict, os.path.join(save_dir, filename))


def train(epochs, model, data_loader, val_loader, criterion, optimizer, scheduler, val_every, device):
    """
    Training

    Params:
        epochs(int) : 학습 에폭 수
        model(nn.Module) : 모델
        data_loader(DataLoader) : 학습 데이터 로더
        val_loader(DataLoader) : validation 데이터 로더
        criterion(loss func) : loss function
        optimizer(optimizer func) : optimizer function
        save_dir(str) : 모델 저장 공간
        val_every(int) : validation 주기
        device(device) : 사용 device
    """

    logger.info('Start training..')

    best_epoch = 0
    best_val_loss = 99999
    best_val_miou = 0

    for epoch_idx in range(epochs):
        epoch = epoch_idx + 1
        model.train()
        if config.lr_scheduler == 'SGDR': scheduler.step()

        _cum_loss = 0
        _cum_hist = np.zeros((12, 12))

        for step_idx, (_, images, masks, _) in enumerate(tqdm(data_loader)):
            # DEBUG: Scheduler
            # scheduler.step()
            # continue
            # break

            step = step_idx + 1

            # Dataset
            images = torch.stack(images)         # (batch, channel, height, width)
            masks = torch.stack(masks).long()    # (batch, height, width)
            # DEBUG: Dataset
            # print(images.min(), images.max())
            # continue
            # 마지막 step 때 batch size가 1인 경우 건너뜀
            if images.shape[0] == 1:
                continue

            # Device 할당
            images = images.to(device)
            masks = masks.to(device)

            # Inference
            outputs = model(images)

            # Loss and optimization
            if isinstance(criterion, list):
                loss = 0
                for _criterion, weights in zip(criterion, config.loss_weights):
                    loss += weights * _criterion(outputs, masks)
            else:
                loss = criterion(outputs, masks)
            _cum_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Add histogram and calculate mIoU.
            # TODO: Loss는 현 step의 batch에 대해서 계산, mIoU는 현재까지 누적된 모든 batches 대해서 계산, 두 가지가 다르므로 수정 필요
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            hist = add_hist(np.zeros((12, 12)), masks.detach().cpu().numpy(), outputs, n_class=12)
            _, _, miou, _ = label_accuracy_score(hist)
            _cum_hist = add_hist(_cum_hist, masks.detach().cpu().numpy(), outputs, n_class=12)

            # [LOG] Training Step
            if step % (len(data_loader) // 10) == 0:
                print(f'Epoch [{epoch}/{epochs}] | Step [{step}/{len(data_loader)}] | Loss: {loss.item():.4f} | mIoU: {miou:.4f}')
        
        # Calculate loss at the current epoch.
        loss = _cum_loss / len(data_loader)
        acc, acc_cls, miou, fwavacc = label_accuracy_score(_cum_hist)
      
        # Validation
        if epoch % val_every == 0:
            val_loss, val_miou = eval(epoch, model, val_loader, criterion, device)

            # LOG: W&B
            if args.wandb_project is not None:
                wandb.log(
                    {'epoch': epoch,
                     'lr': optimizer.param_groups[0]['lr'],
                     'loss': loss,
                     'train_mIoU': miou,
                     'val_loss': val_loss,
                     'val_mIoU': val_miou}
                     )

            # Save the model at the best performance epoch.
            if val_miou > best_val_miou:
                # print(f'Best performance at epoch: {epoch}')
                # print('Save model in', save_dir)

                best_epoch = epoch
                best_val_loss = val_loss
                best_val_miou = val_miou

                # Save the model.
                save_model(model.state_dict(), args.save_dir, filename = args.experiment + r'.pt')
            
        # LOG: The best performance
        print(f'Best epoch #{best_epoch} | Loss: {best_val_loss:.4f} | mIoU: {best_val_miou:.4f}')


def eval(epoch, model, data_loader, criterion, device):
    """
    Validation

    Params:
        epochs(int) : 학습 에폭 수
        model(nn.Module) : 모델
        data_loader(DataLoader) : 학습 데이터 로더
        criterion(loss func) : loss function
        device(device) : 사용 device
    """

    logger.info(f'Start validation #{epoch}')

    model.eval()

    hist = np.zeros((12, 12))

    with torch.no_grad():
        _cum_loss = 0
        # miou_list = []

        for step_idx, (_, images, masks, _) in enumerate(tqdm(data_loader)):
            # DEBUG
            # continue
            # break

            # Dataset
            images = torch.stack(images)         # (batch, channel, height, width)
            masks = torch.stack(masks).long()    # (batch, height, width)

            # Device 할당
            images = images.to(device)
            masks = masks.to(device)

            # Inference
            outputs = model(images)

            # Loss
            if isinstance(criterion, list):
                loss = 0
                for _criterion, weights in zip(criterion, config.loss_weights):
                    loss += weights * _criterion(outputs, masks)
            else:
                loss = criterion(outputs, masks)
            _cum_loss += loss

            # Add histogram.
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            hist = add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
            # miou = label_accuracy_score(masks.detach().cpu().numpy(), outputs, n_class=12)[2]
            # miou_list.append(miou)
        
        # Calculate loss and mIoU at the current epoch.
        loss = _cum_loss / len(data_loader)
        acc, acc_cls, miou, fwavacc = label_accuracy_score(hist)
        # miou = np.mean(miou_list)

        # LOG: Validation
        print(f'Validation #{epoch} | Loss: {loss:.4f} | mIoU: {miou:.4f}')

    return loss, miou


def main(args, config):
    logger.info("-------------------- Hyperparameters --------------------")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Loss: {config.loss}")
    logger.info(f"Optimizer: {config.optimizer}")
    logger.info(f"Weight decay: {config.weight_decay}")
    logger.info(f"Epochs: {config.epochs}")
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"Learning rate: {config.learning_rate}")
    logger.info("--------------------------------------------------\n")

    logger.info("--------------------------------------------------")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logger.info(f'We will use the GPU:{torch.cuda.get_device_name(0)}')
    else:
        device = torch.device("cpu")
        logger.info('No GPU available, using the CPU instead.')
    logger.info("--------------------------------------------------\n")

    """ Dataset """
    # Augmentation
    train_transform, val_transform, _ = get_transforms(config.aug)

    # Train dataset
    train_dataset = CustomDataLoader(
        data_dir=train_data_path, mode='train', transform=train_transform
        )
    # Validation dataset
    val_dataset = CustomDataLoader(
        data_dir=val_data_path, mode='val', transform=val_transform
        )

    # DataLoader
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=1,
        collate_fn=collate_fn
        )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=1,
        collate_fn=collate_fn
        )

    """ Model """
    if args.api == 'smp':
        if config.model == 'DeepLabV3Plus':
            model = smp.DeepLabV3Plus(
                encoder_name=config.enc,
                encoder_weights=config.enc_weights,
                classes=12
                )
    else:
        if config.model == 'DeepLabV3EffiB7Timm':
            model = DeepLabV3EffiB7Timm(
                n_classes=12,
                n_blocks=[3, 4, 23, 3],
                atrous_rates=[6, 12, 18, 24]
                )

    logger.info("-------------------- Model Test --------------------")
    x = torch.randn([2, 3, 512, 512])
    logger.info(f"Input shape : {x.shape}")
    out = model(x).to(device)
    logger.info(f"Output shape: {out.shape}")
    logger.info("--------------------------------------------------\n")

    model = model.to(device)

    # Loss function
    if config.loss == 'CE':
        criterion = nn.CrossEntropyLoss()
    elif config.loss == 'SoftCE':
        criterion = SoftCrossEntropyLoss(smooth_factor=config.smooth_factor)
    elif config.loss == 'Focal':
        criterion = FocalLoss('multiclass', gamma=config.focal_gamma)
    elif config.loss == 'DiceCE':
        pass
    elif config.loss == 'RMI':
        # criterion = RMILoss(num_classes=12, loss_weight_lambda=config.RMI_weight)
        criterion = [
            SoftCrossEntropyLoss(smooth_factor=config.smooth_factor),
            FocalLoss('multiclass', gamma=config.focal_gamma),
            # RMILoss(num_classes=12, rmi_only=True)
            RMILoss(num_classes=12)
        ]
    # TODO: Split 및 getattr() 적용
    elif config.loss == 'SoftCE+Focal+RMI':
        criterion = [
            SoftCrossEntropyLoss(smooth_factor=config.smooth_factor),
            FocalLoss('multiclass', gamma=config.focal_gamma),
            # RMILoss(num_classes=12, rmi_only=True)
             RMILoss(num_classes=12)
            ]
    else:
        raise Exception('[ERROR] Invalid loss')

    # Optimizer
    learning_rate = config.lr_min if config.lr_scheduler == 'SGDR' else config.learning_rate
    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            weight_decay=config.weight_decay
            )
    elif config.optimizer == 'AdamP':
        optimizer = AdamP(
            params= model.parameters(),
            lr=learning_rate,
            weight_decay=config.weight_decay
            )
    else:
        raise Exception('[ERROR] Invalid optimizer')
    
    # Learning rate scheduler
    if config.lr_scheduler == 'no':
        scheduler = None
    elif config.lr_scheduler == 'SGDR':
        scheduler = CosineAnnealingWarmUpRestarts(
            optimizer,
            T_0=config.T,
            T_up=config.T_warmup,
            T_mult=config.T_mult,
            eta_max=config.lr_max,
            gamma=config.lr_max_decay
            )
    else:
        raise Exception('[ERROR] Invalid learning rate scheduler')

    """ Train the model """
    train(
        epochs=config.epochs,
        model=model,
        data_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        val_every=1,
        device=device)


if __name__ == '__main__':

    """ Arguments """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=21, help='Random seed')
    parser.add_argument('-e', '--experiment', type=str, default=None, help='Experiment ID or name (W&B name)')
    parser.add_argument('-w', '--wandb_project', type=str, default='bc_AI_p3_img_seg', help='W&B project (if None, W&B will be not used.')
    parser.add_argument('--wandb_group', type=str, default='=new_group', help='W&B group')
    parser.add_argument('--wandb_sweep', type=bool, default=False, help='W&B sweep (if False, Sweep will be not used.')
    parser.add_argument('-c', '--config', type=str, default=None, help='Configuration file path')
    parser.add_argument('--api', type=str, default='smp', help='API for segmentation model')
    parser.add_argument('-s', '--save_dir', type=str, default='../results/', help='Directory to save')
    args = parser.parse_args()

    seed_everything(args.seed)

    """ Configuration """
    with open(args.config) as f:
        hparams = json.load(f)
    if args.experiment is None: args.experiment = hparams['model']
    if args.wandb_project is None:
        config = Bunch()
        config.update(hparams)
    else:
        if args.wandb_sweep:
            wandb.init(project=args.wandb_project, config=hparams)
        else:
            wandb.init(project=args.wandb_project, group=args.wandb_group, name=args.experiment, config=hparams)
        config = wandb.config
    del hparams

    """ Directories to save models """
    try:
        if args.save_dir == r'/opt/ml/saved/':
            os.makedirs(args.save_dir, exist_ok=True)
        else:
            os.makedirs(args.save_dir)
    except FileExistsError as err:
        print(r'[Error] The directory exists. Check the directory to save the model.')

    main(args, config)
