import os
import sys
import argparse
import json
import warnings
import random
import logging
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from tqdm import tqdm


dataset_path = '/opt/ml/input/data'
train_path = dataset_path + '/train.json'
val_path = dataset_path + '/val.json'
test_path = dataset_path + '/test.json'

class CFG:
    epochs =  20
    seed = 21
    learning_rate = 0.0001
    batch_size = 16
    weight_decay = 1e-6
    save_path = "./saved"
    aug_type = "basic"
    train_all = False
    save_name= "default_model"
    encoder_name = None
    decoder_name = None
    pretrain_weight = None
    


class ParameterError(Exception):
    def __init__(self):
        super().__init__('Enter essential parameters')

def __get_logger():
    """로거 인스턴스 반환
    """

    __logger = logging.getLogger('logger')

    # # 로그 포멧 정의
    formatter = logging.Formatter(fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
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


def seed_everything(seed):
    """ seed설정 함수

    Args : 
        seed(int) : 시드값 설정 변수
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ********************************************
# val_every = 1 
# saved_dir = './saved'
    
def save_model(model, saved_dir, file_name=CFG.save_name):
    check_point = {'net': model.state_dict()}
    output_path = os.path.join(saved_dir, file_name)
    torch.save(model.state_dict(), output_path)

def collate_fn(batch):
    return tuple(zip(*batch))
# ********************************************

def train(num_epochs, model, data_loader, val_loader, criterion, optimizer, saved_dir, val_every, device):
    """모델 학습을 위한 함수
    
    Args:
        num_epochs(int) : 학습 에폭 수
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
    best_mIou = 0
    
    for epoch in range(num_epochs):
        model.train()
        mIoU_list=[]
        cnt=0
        hist=np.zeros((12,12))
        for step, (images, masks, _) in enumerate(tqdm(data_loader)):
            cnt+=1
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)
            
            # gpu 연산을 위해 device 할당
            images, masks = images.to(device), masks.to(device)
                  
            # inference
            outputs = model(images)
            
            # loss 계산 (cross entropy loss)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train_all 시, step마다 miou 누적계산
            if(CFG.train_all):
                outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
                hist=add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
                acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
                mIoU_list.append(mIoU)
            # step 주기에 따른 loss 출력
            if (step + 1) % 25 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch+1, num_epochs, step+1, len(data_loader), loss.item()))
        
        # validation 주기에 따른 loss 출력 및 best model 저장
        
        if (epoch + 1) % val_every == 0:
            # train_all 시, epoch마다 validation 대신 train 간 평균 mIoU best model 저장.
            if(CFG.train_all):
                train_epoch_miou=np.mean(mIoU_list)
                if train_epoch_miou > best_miou:
                    print('Best performance at epoch: {} , mIoU : {}'.format(epoch + 1,train_epoch_miou))
                    print('Save model in', saved_dir)
                    best_miou = train_epoch_miou
                    save_model(model, saved_dir)
            else:
            # validation 간 best mIoU 모델 저장
                avgr_loss,val_epoch_miou = validation(epoch + 1, model, val_loader, criterion, device)
                if val_epoch_miou > best_miou:
                    print('Best performance at epoch: {} , val_mIoU: {}'.format(epoch + 1,val_epoch_miou))
                    print('Save model in', saved_dir)
                    best_miou = val_epoch_miou
                    save_model(model, saved_dir)

def validation(epoch, model, data_loader, criterion, device):
    """validation평가를 위한 함수
    
    Args:
        epochs(int) : 학습 에폭 수
        model(nn.Module) : 모델
        data_loader(DataLoader) : 학습 데이터 로더
        criterion(loss func) : loss function
        device(device) : 사용 device
    """
    logger.info('Start validation #{}'.format(epoch))
    model.eval()
    hist=np.zeros((12,12))
    with torch.no_grad():
        total_loss = 0
        cnt = 0
        mIoU_list = []
        for step, (images, masks, _) in enumerate(tqdm(data_loader)):
            
            images = torch.stack(images)       # (batch, channel, height, width)
            masks = torch.stack(masks).long()  # (batch, channel, height, width)

            images, masks = images.to(device), masks.to(device)            

            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss
            cnt += 1
            
            outputs = torch.argmax(outputs.squeeze(), dim=1).detach().cpu().numpy()
            hist=add_hist(hist, masks.detach().cpu().numpy(), outputs, n_class=12)
            acc, acc_cls, mIoU, fwavacc = label_accuracy_score(hist)
            mIoU_list.append(mIoU)
            
        avrg_loss = total_loss / cnt
        print('Validation #{}  Average Loss: {:.4f}, mIoU: {:.4f}'.format(epoch, avrg_loss, mIoU))
    return avrg_loss,np.mean(mIoU_list)

def main():
    """train 메인 로직 수행
    """
    logger.info("*************SETTING************")
    logger.info(f"seed : {CFG.seed}")
    logger.info(f"epochs : {CFG.epochs}")
    logger.info(f"learning rate : {CFG.learning_rate}")
    logger.info(f"batch size : {CFG.batch_size}")
    logger.info(f"weight decay : {CFG.weight_decay}")
    logger.info(f"train all : {CFG.train_all}")
    logger.info("********************************\n")

    if torch.cuda.is_available():
        logger.info("*************************************")
        device = torch.device("cuda")
        logger.info(f'There are {torch.cuda.device_count()} GPU(s) available.')
        logger.info(f'We will use the GPU:{torch.cuda.get_device_name(0)}')
        logger.info("*************************************\n")
    else:
        logger.info("*************************************")
        logger.info('No GPU available, using the CPU instead.')
        device = torch.device("cpu")
        logger.info("*************************************\n")

    train_transform, val_transform, test_transform = get_transforms(CFG.aug_type)

    # train dataset
    if(CFG.train_all):
        #print("train_all_mode")
        train_path=dataset_path + '/train_all.json'
    else:
        train_path=dataset_path + '/train.json'

    train_dataset = CustomDataLoader(data_dir=train_path, mode='train', transform=train_transform)
    ###
    #print(len(train_dataset))
    ###
    
    # validation dataset
    if(CFG.train_all):
        val_loader=None
    else:
        val_dataset = CustomDataLoader(data_dir=val_path, mode='val', transform=val_transform)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, 
        batch_size= CFG.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
        )
        
        

    # DataLoader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
    batch_size=CFG.batch_size,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
    )
    
    
    if(CFG.encoder_name is not None and CFG.decoder_name is not None and CFG.pretrain_weight is not None):
        model = smpModel(encoder=CFG.encoder_name,pretrain_weight=CFG.pretrain_weight,decoder=CFG.decoder_name)
    else:
        model = testDeepLabV3()
    
#     logger.info("*************Model Formatting Test************")
#     x = torch.randn([1, 3, 512, 512])
#     logger.info(f"input shape : {x.shape}")
#     out = model(x).to(device)
#     logger.info(f"output shape :  {out.size()}")
#     logger.info("********************************\n")
    
    model = model.to(device)

    # Loss function 정의
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer 정의
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG.learning_rate, weight_decay = CFG.weight_decay)

    # 학습 수행
    train(
        num_epochs = CFG.epochs, 
        model = model, 
        data_loader = train_loader, 
        val_loader = val_loader, 
        criterion = criterion, 
        optimizer = optimizer,
        saved_dir = CFG.save_path, 
        val_every = 1 , 
        device = device)
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', '--config', default=None, type=str,help='config file path')
    parser.add_argument('-s', '--save', default=None, type=str,help='save path')
    
    
    args = parser.parse_args()

    sys.path.append("/opt/ml/pstage03")
    from dataloader.image import *
    from model.models import *
    from util.loss import *
    from util.utils import *
    from util.augmentation import *

    with open(args.config) as json_file:
        json_data = json.load(json_file)

    # parameter setting
    CFG.epochs = json_data['epochs']
    CFG.seed = json_data['seed']
    CFG.learning_rate = json_data['learning_rate']
    CFG.batch_size = json_data['batch_size']
    CFG.weight_decay = json_data['weight_decay']
    CFG.aug_type = json_data['aug_type']
    
    CFG.save_path = args.save
    CFG.save_name = json_data['save_name']
    CFG.encoder_name = json_data['encoder_name']
    CFG.decoder_name = json_data['decoder_name']
    CFG.pretrain_weight=json_data['pretrain_weight']
    
    
    seed_everything(CFG.seed)
    
    if json_data['train_all']=="True":
        CFG.train_all = True
        logger.info(f"train_all mode")
    
    else:
        CFG.train_all = False
    
    # save 디렉토리 생성
    if not os.path.isdir(CFG.save_path):
        os.mkdir(save_path)

    main()