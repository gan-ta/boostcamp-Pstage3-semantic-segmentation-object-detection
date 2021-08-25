# Heesup Object Detection

mmdetection 라이브러리를 활용하기 위한 다양한 모델의 개인 config 구축 파일 입니다.

**`_base_`** : 기본 base가 되는 config파일

- datasets
    - coco_instance.py : 기본 detection 모델을 위한 dataset(segmentation 정보 포함)
    - coco_instance-Aug.py : detection model + Augmentation 적용 및 튜닝을 한 dataset(segmentation 정보 포함)
    - coco_detection.py : 기본 detection 모델을 위한 dataset
    - coco_detection-Aug.py : detection model + Augmentation 적용 및 튜닝을 한 dataset (faster rcnn으로 Augmentation 실험 시 사용)
    
- models
    - rpn_r50_fpn.py : resnet50 RPN 기본 base 모델 설정
    - retinanet_r50_fpn.py : resnet50 retinanet 기본 base 모델 설정
    - faster_rcnn_resnext.py : resnext faster rcnn 기본 base 모델 설정 config파일
    - faster_rcnn_r50_fpn.py : resnet50 faster rcnn 기본 base 모델 설정 config파일
    - casecade_mask_rcnn_swin_fpn.py : swin casecade mask rcnn 기본 base 모델 설정 config파일
    - cascade_rcnn_resnext_fpn.py : resnext casecade rcnn 기본 base 모델 설정 config파일
    - cascade_rcnn_r50_fpn.py : resnet50 casecade rcnn 기본 base 모델 설정 config파일
    
 - schedules
    - schedule_1x.py : shedule_1x 기본 스케줄러 config 설정
    - schedule_2x.py : shedule_2x 기본 스케줄러 config 설정
    - schedule_tun.py : 최종 결정 스케줄러 config 설정
    - shcedule_20e.py : shedule 20 epoch 기본 스케줄러 config 설정
    
- default_runtime.py : runtime 시 사용한 설정
 
**`bifpn`**

- htc_bifpn_ResNest.py : ResNest + BiFPN(neck 튜닝) + HTC 모델 조합

- faster_rcnn_bifpn_ResNest.py : ResNest + BiFPN(neck 튜닝) + fasterRCNN 모델 조합

**`detectors`** 

- detectors_resnext.py : ResNext detectors 모델 조합(RFP)

**`dynamic_rcnn`** 

- dynamic_rcnn_fpn-carafe_ResNest.py : ResNest + fpn-carafe + dynamic RCNN 모델 조합

**`faster_rcnn`**

- HRNET_faster_rcnn.py ; HRNet + HRFPN + fasterRCNN 조합의 모델

- RegNet_faster_rcnn.py : RegNetx + FPN + fasterRCNN 조합의 모델

- Res2Net_faster_rcnn.py : Res2Net + FPN + fasterRCNN 조합의 모델

- ResNest_faster_rcnn-Aug.py : ResNest + FPN + fasterRCNN 조합의 모델(최종 Augmentation 조합 사용)

- ResNest_faster_rcnn.py : ResNest + FPN + fasterRCNN 조합의 모델

- ResNet_faster_rcnn-Aug.py : ResNet + FPN + fasterRCNN 조합의 모델

- resnext_faster_rcnn.py : ResNext + FPN + fasterRCNN 조합의 모델

**`fpg`**

- faster_rcnn_fpg_ResNest.py : ResNest + fpg + faster RCNN 모델 조합

**`fpn-carafe`**

- faster_rcnn_fpn-carafe_ResNest.py : ResNest + fpn-carafe + faster RCNN 모델 조합

- htc_fpn-carafe_dc_ResNest.py : ResNest + fpn-carafe + htc(Deformable Convolution 적용)모델 조합

- htc_fpn-carafe_ResNest-Aug.py : ResNest + fpn-carafe + htc(Augmentation)모델 조합

- htc_fpn-carafe_ResNest.py: ResNest + fpn-carafe + htc(Augmentation) 모델 조합

- scnet_fpn-carafe_ResNest.py : ResNest + fpn-carafe + scnet 모델 조합

**`htc`**

- htc_dc_ResNest.py :ResNest + FPN + HTC(Deformable Convolution 적용) 모델 조합

- htc_ResNest-Aug.py : ResNest + FPN + HTC(Augmentation 적용) 모델 조합

- htc_ResNest.py : ResNest + FPN + HTC 모델 조합

**`nas-fpn`**

- nas-fpn_ResNest.py : ResNest + NASFPN + faster RCNN 모델 조합

**`nascos-fpn`**

- nascos-fpn_ResNest.py : ResNest + nasfcos + faster RCNN 모델 조합

**`pafpn`**

- faster_rcnn_pafpn_ResNest.py : ResNest + pafpn + faster RCNN 모델 조합

**`rpn`** 

- rpn_ResNest.py : ResNest + FPN + rpn 모델 조합

**`scnet`** 

- scnet_ResNest.py : ResNest + FPN + SCNet 모델 조합

**`swin`**

- casecade_rcnn_swin.py : swin + casecade RCNN 모델 조합(mmdetection 라이브러리에서는 사용하지 않고 swin 전용 저장소에서 사용)

