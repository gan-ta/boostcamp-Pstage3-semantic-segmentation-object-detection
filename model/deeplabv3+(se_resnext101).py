import torch.nn as nn
import segmentation_models_pytorch as smp

class DeepLabV3Plus_se_resnext101(nn.Module):
    def __init__(self, in_channels = 3, classes = 12):
        super(DeepLabV3Plus_se_resnext101, self).__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="se_resnext101_32x4d",     
            encoder_weights="imagenet",
            in_channels=3,               
            classes=12
        )

    def forward(self, x):
        output = self.backbone(x)
        
        return output