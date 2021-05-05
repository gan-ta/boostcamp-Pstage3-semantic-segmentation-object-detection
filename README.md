# 부스트캠프 AI Tech

### [P Stage 3] Image segmentation and Detection
### 14조 너의 마음 검출도 쌉가능


## 1. Usage
---

* Training: `python train.py --experiment new_experiment --config ../config/config.json`
<!-- * Inference: `` -->


<!-- ## Directory 구조 -->


<!-- ## Command Line Arguments -->
<!-- --- -->

<!-- ### `train.py` -->

<!-- ### `inference.py` -->


## 2. Configuration (.json)
---

1. Base

> * `model`: str (default: "DeepLabV3Plus")
>     * ["DeepLabV3Plus", "DeepLabV3EffiB7Timm"]
> * `enc`: str (default: "timm-regnety_320")
>     * ["timm-regnety_320", "timm-efficientnet-b0", "timm-efficientnet-b3"]
> * `enc_weights`: str (default: "imagenet")
>     * ["imagenet", "noisy-student"]
> * `epochs`: int (default: 20)

2. Loss
 
> * `loss`: str (default: "CE")
>     * ["CE", "DiceCE", "RMI"]
> * `loss_weight`: float (default: 0.5)
>     * Compound loss 사용시 `loss = loss_weight * distribution-based_loss + (1 - loss_weight) * region-based_loss`

3. Optimizer
 
> * `optimizer`: str (default: "Adam")
>     * ["Adam", "AdamP"]
> * `weight_decay`: float, (default: 1e-6)

4. Batch size and learning rate
 
> * `batch_size` : int (default: 8)
> * `learning_rate` : float (default: 1e-4)
> * `lr_scheduler`: str (default: "no")
>     * ["no", "SGDR"]
> * `lr_min`: float (default: 1e-6)
> * `lr_max`: float (default: 1e-4)
> * `lr_max_decay`: float (default: 0.8)
> * `T`: int (default: 4)
> * `T_warmup`: int (default: 2)
> * `T_mult`: int (default: 2)

1. Data augmentation (업데이트 중)

> * `aug`: str (default: "no)