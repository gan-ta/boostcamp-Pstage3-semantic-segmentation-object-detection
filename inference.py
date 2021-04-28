import sys
import argparse

import torch

dataset_path = '/opt/ml/input/data'
train_path = dataset_path + '/train.json'
val_path = dataset_path + '/val.json'
test_path = dataset_path + '/test.json'

def collate_fn(batch):
    return tuple(zip(*batch))

def test(model, data_loader, device):
    size = 256
    transform = A.Compose([A.Resize(256, 256)])
    print('Start prediction.')
    model.eval()
    
    file_name_list = []
    preds_array = np.empty((0, size*size), dtype=np.long)
    
    with torch.no_grad():
        for step, (imgs, image_infos) in enumerate(tqdm(test_loader)):

            # inference (512 x 512)
            outs = model(torch.stack(imgs).to(device))['out']
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(temp_images), oms):
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


def test(model, data_loader, device):
    """저장된 model에 대한 prediction 수행

    Args:
        model(nn.Module) : 저장된 모델
        data_loader(Dataloader) : test데이터의 dataloader
        device
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
            outs = model(torch.stack(imgs).to(device))['out']
            oms = torch.argmax(outs.squeeze(), dim=1).detach().cpu().numpy()
            
            # resize (256 x 256)
            temp_mask = []
            for img, mask in zip(np.stack(temp_images), oms):
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

    # test dataset
    test_dataset = CustomDataLoader(data_dir=test_path, mode='test', transform=test_transform)

    # test dataloader
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
    batch_size= CFGInference.batch_size,
    num_workers=4,
    collate_fn=collate_fn
    )

    # sample_submisson.csv 열기
    submission = pd.read_csv('./submission/sample_submission.csv', index_col=None)
    
    # test set에 대한 prediction
    file_names, preds = test(model, test_loader, device)
    
    # PredictionString 대입
    for file_name, string in zip(file_names, preds):
        submission = submission.append({"image_id" : file_name, "PredictionString" : ' '.join(str(e) for e in string.tolist())}, ignore_index=True)
        
    # submission.csv로 저장
    submission.to_csv("./submission/Baseline_DeepLabv3(vgg16).csv", index=False)


class CFGInference:
  batch_size = 16

if name == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('-m', '--model', default=None, type=str, help='model path')
    parser.add_argument('-c', '--config', default=None, type=str,help='config file path') # 훈련때 사용했던 config파일 사용

    sys.path.append("/opt/ml/pstage03")
    from dataloader.image import *

    CFGInference.batch_size = json_data['batch_size']

    main()