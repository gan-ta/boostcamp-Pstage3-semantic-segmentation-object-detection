import os
import sys
from tqdm import tqdm
import argparse
import logging

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch

""" Custom Modules """
sys.path.append(os.path.abspath(r'./baseline_code/'))
sys.path.append(os.path.abspath(r'./'))
sys.path.append(os.path.abspath(r'../../'))

from dataloader.image import *
from model.FCN8s import *
from model.deeplabv3Plus_effib7 import *
from model.deeplabv3_ResNext import *
from model.efficientb7_DeepLabv3_timm import *
from utils.CRF import *

""" Paths """
data_dir = r'/opt/ml/input/data/'
train_config_path = os.path.join(data_dir, r'train.json')
val_config_path = os.path.join(data_dir, r'val.json')
test_config_path = os.path.join(data_dir, r'test.json')


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


def collate_fn(batch):
    return tuple(zip(*batch))

def test(model, test_loader, device):
    """저장된 model에 대한 prediction 수행

    Args:
        model(nn.Module) : 저장된 모델
        test_loader(Dataloader) : test데이터의 dataloader
        device

    테스트 loader는 배치 사이즈 4 이하이면 작동을 하지 않습니다.   
    """

    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()

    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)

    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))

            # crf적용
            outs =  dense_crf(imgs, outs.cpu().numpy())
            outs = torch.tensor(outs).to(device)

            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()

            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(imgs), oms):
                transformed = transform(image=img, mask=mask)
                mask = transformed['mask']
                temp_mask.append(mask)

            oms = np.array(temp_mask)

            oms = oms.reshape([oms.shape[0], size*size]).astype(int)
            preds_array = np.vstack((preds_array, oms))

            file_name_list.append([i['file_name'] for i in image_infos])
    print("End prediction.")
    file_names = [y for x in file_name_list for y in x]

    return file_names, preds_array


def main():
    """inference main logic수행
    """
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


    # model 설정
    model_path = '/opt/ml/saved/' + CFGInference.model_path

    model = DeepLabV3EffiB7Timm(n_classes=12, n_blocks=[3, 4, 23, 3], atrous_rates=[6, 12, 18, 24])
 
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    
    # default normalize
    norm_mean = (0, 0, 0)
    norm_std = (1, 1, 1)

    test_transform = A.Compose([
        A.Normalize(mean=norm_mean, std=norm_std),
        ToTensorV2()
    ])

    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_config_path, mode='test', transform=test_transform)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    batch_size= CFGInference.batch_size,
    num_workers=1,
    collate_fn=collate_fn
    )

    # sample_submisson.csv 열기
    submission = pd.read_csv('/opt/ml/submission/sample_submission.csv', index_col=None)

    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)

    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, ignore_index=True)

    # submission.csv로 저장
    submission.to_csv("/opt/ml/submission/" + CFGInference.model_path.split(".")[0] + ".csv", index=False)
    

class CFGInference:
  batch_size = 4
  model_path = None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-m', '--model', default=None, type=str, help='model path')
    # parser.add_argument('-c', '--config', default=None, type=str,help='config file path') # 훈련때 사용했던 config파일 사용
    parser.add_argument('-b' , '--batch', default = None, type = str, help = 'batch size')

    args = parser.parse_args()

    CFGInference.batch_size = int(args.batch)
    CFGInference.model_path = args.model

    main()