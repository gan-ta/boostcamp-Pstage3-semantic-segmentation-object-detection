# Heesup Object Detection

mmdetection 라이브러리를 활용하기 위한 다양한 모델의 개인 config 구축 파일 입니다.

**`_base_`** : 기본 base가 되는 config파일

- datasets
    - coco_instance.py : swin detection model + default dataset
    - coco_instance-Aug.py : swin detection model + Augmentation version1 적용 및 튜닝을 한 dataset
    - coco_instance-Aug2.py : swin detection model + Augmentation version2 적용 및 튜닝을 한 dataset
    
- models
    - htc_swin_fpn-carafe.py: swin + fpn-carafe + htc default 모델 조합
        
 - schedules
    - schedule_tun.py : 최종 결정 스케줄러 config 설정(optimizer, 스케줄러 조정)
    
- default_runtime.py : runtime 시 사용한 설정

**`swin`**

- htc_fpn-carafe_swinS.py : swin small + fpn-carafe + htc default 모델 조합

- htc_fpn-carafe_swinB.py : swin base + fpn-carafe + htc default 모델 조합

