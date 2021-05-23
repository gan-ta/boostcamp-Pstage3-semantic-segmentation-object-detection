import torch.nn as nn
import segmentation_models_pytorch as smp

class MANet_efficientNet_b7(nn.Module):
    def __init__(self, in_channels = 3, classes = 12):
        super(MANet_efficientNet_b7, self).__init__()
        self.backbone = smp.MAnet(
            encoder_name="efficientnet-b7",     
            encoder_weights="imagenet",
            in_channels=3,               
            classes=12
        )

    def forward(self, x):
        output = self.backbone(x)
        
        return output