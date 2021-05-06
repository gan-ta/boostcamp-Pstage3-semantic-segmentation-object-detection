# 부스트캠프 AI Tech

### [P Stage 3] Image segmentation and Detection
### 14조 너의 마음 검출도 쌉가능


## 1. Usage
---

* Training: `python train.py --experiment NEW_EXPERIMENT --config ../config/config.json`
* Inference: 업데이트 중. 각자의 Jupyter 코드를 사용해주세요.


<!-- ## Directory 구조 -->


<!-- ## Command Line Arguments -->
<!-- --- -->

<!-- ### `train.py` -->

<!-- ### `inference.py` -->


## 2. Configuration (.json)
---

1. Base

    * `model`: str (default: "DeepLabV3Plus")
        * ["DeepLabV3Plus", "DeepLabV3EffiB7Timm"]
        * 직접 구현되어 있는 `DeepLabV3EffiB7Timm` 외에는 `smp` documents를 참고하여 `model`, `enc`, `enc_weights`를 설정해주세요.
    * `enc`: str (default: "timm-regnety_320")
        * ["timm-regnety_320", "timm-efficientnet-b0", "timm-efficientnet-b3"]
    * `enc_weights`: str (default: "imagenet")
        * ["imagenet", "noisy-student"]
    * `epochs`: int (default: 20)

2. Loss
 
    * `loss`: str (default: "CE")
        * ["CE", "SoftCE", "Focal", ~~"DiceCE"~~, "RMI"]
        * CE 외의 다른 loss 사용시 추가적인 코드가 필요합니다.
    * `loss_weights`: list[float] (default: None)
        * Weights of losses (multi-loss)
        * 합이 1이 되게끔 설정해주는 것을 권장합니다.

    * Soft CE
        * `smooth_factor`: float (default: 0.2)
    * Focal
        * `focal_gamma`: float (default: 2.0)
    * RMI 
        * `RMI_weight`: float (default: 0.5)
            * RMI loss 사용시 `loss = RMI_weight * BCE + (1 - RMI_weight) * RMI`

3. Optimizer

    * `optimizer`: str (default: "Adam")
        * ["Adam", "AdamP"]
    * `weight_decay`: float, (default: 1e-6)

4. Batch size and learning rate
 
    * `batch_size` : int (default: 8)
    * `learning_rate` : float (default: 1e-4)
    * `lr_scheduler`: str (default: "no")
        * ["no", "SGDR"]

    * Scheduler
        * SGDR
            * `lr_min`: float (default: 1e-6)
            * `lr_max`: float (default: 1e-4)
            * `lr_max_decay`: float (default: 0.5)
            * `T`: int (default: 4)
            * `T_warmup`: int (default: 2)
            * `T_mult`: int (default: 2)

5. Data augmentation (업데이트 중)

    * `aug`: str (default: "no)
