# 부스트캠프 AI Tech

### Image segmentation and Detection
본프로젝트는 NAVER AI BoostCamp에서 개최한 competition입니다

### 최종 결과
🧨 Public Score: 0.6520

🏆 Private Score: 0.6239

## File Structure

### Baseline Code
```
baseline_code/
├── config/ - model, hyper parameter config
│   ├── config.json
│   ├── config_basic.json
│   ├── config_dev.json
│   ├── train_Effib7_deeplabv3.json
│   ├── config_ResNext_deeplabv3.json
│   └── train_Effib7_deeplabv3.json
│
├── dataloader/ - data 
│   └── image.py
│
├── model / - models
│   ├── deconvNet.py
│   ├── deeplabV1.py
│   ├── deeplabV2.py
│   ├── deeplabV3.py
│   ├── deeplabV3_resnext_ver1.py
│   ├── deeplabV3_resnext_ver2.py
│   ├── deeplabV3_resnext_ver2.py
│   ├── dilated.py
│   ├── deepplabV3_effib7_ver1.py 
│   ├── deepplabV3_effib7_ver2.py 
│   ├── deepplabV3_effib7_ver3.py 
│   ├── deeplabV3Plus_effib7.py
│   ├── FCN8s.py
│   ├── MANet_effib7.py
│   ├── models.py - initial version
│   ├── segNet.py
│   └── unet_ver1_padding.py
│
├── util /
│   ├── augmentation.py
│   ├── CRF.py
│   ├── loss.py
│   ├── additional_loss.py
│   ├── scheduler.py
│   └── util.py
│
├── train.py
│
└── inference.py
```

### notebook
```
notebook/
├── heesup /
│   ├── dailymission_model_implementation
│   │   ├── DeconvNet.ipynb
│   │   ├── DeepLabV1_VGG16_imple.ipynb
│   │   ├── DeepLabv2_VGG16_imple.ipynb
│   │   ├── DeepLabv3_VGG16_imple.ipynb
│   │   ├── DilatedNet.ipynb
│   │   ├── SegNet.ipynb                          
│   │   ├── UNet.ipynb
│   │   └── utils.py
│   │ 
│   └── practice_analysis
│       ├── Augmentation_vis.ipynb
│       ├── cocoAPI_practice.ipynb
│       ├── crf_practice.ipynb
│       ├── EDA.ipynb
│       ├── efficient + DeepLabv3_practice.ipynb
│       ├── ensemble_voting.ipynb
│       ├── inference_CRF_apply.ipynb
│       ├── inference_filter_model_apply.ipynb
│       ├── inference_TTA.ipynb
│       ├── loss_analysis&practice.ipynb
│       ├── result_analysis.ipynb
│       └── utils.py
│
└── pyunghwa/
    ├── model/
    │   ├── DeepLab-V3.ipynb
    │   ├── DeepLabv3 baseline(efficinetNet-b7).ipynb
    │   └── efficientb7+v3.ipynb
    │ 
    └── loss/
        └── additional_loss.py

```

## Usage
### 1, Train
```
python train.py --experiment NEW_EXPERIMENT --config ../config/config.json
```
### 2, inference
```
python inference.py --model ../results/0001.pth  --batch 4
```

## 문제 해결 측면
- 각자의 솔루션에 대해서는 여기에서 확인 하실 수 있습니다.

**[김희섭](https://github.com/bcaitech1/p3-ims-obd-detecting-your-mind-ssap-possible/blob/master/sementic_segmentation/personal/heesup/README.md)**

