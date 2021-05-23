from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector,init_detector
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)

import logging
import argparse

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


def get_cfg(config_file):
    """ config파일을 이용하여 Config객체 생성

    Args :
        config_file(str) : config파일 이름
    
    Returns:
        Config객체
    """
    classes = ("UNKNOWN", "General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    # config file 들고오기
    cfg = Config.fromfile(config_file)

    PREFIX = '/content/drive/MyDrive/object detection/'

    # dataset 바꾸기
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = PREFIX
    cfg.data.train.ann_file = PREFIX + 'train_mosaic.json'
    # cfg.data.train.pipeline[2]['img_scale'] = (512, 512)

    cfg.data.val.classes = classes
    cfg.data.val.img_prefix = PREFIX
    cfg.data.val.ann_file = PREFIX + 'val.json'
    # cfg.data.val.pipeline[1]['img_scale'] = (512, 512)

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = PREFIX
    cfg.data.test.ann_file = PREFIX + 'test.json'
    # cfg.data.test.pipeline[1]['img_scale'] = (512, 512)

    cfg.data.samples_per_gpu = 2

    cfg.seed=2020
    cfg.gpu_ids = [0]
    cfg.work_dir = './work_dirs/' + args.config.split("/")[-1].split(".")[0]


    # cfg.model.roi_head.bbox_head.num_classes = 11

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)

    return cfg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path')

    args = parser.parse_args()

    cfg = get_cfg(args.config)

    model = build_detector(cfg.model)

    datasets = [build_dataset(cfg.data.train)]

    logger.info("training start!")
    train_detector(model, datasets[0], cfg, distributed=False, validate=True)
    logger.info("training end!")

