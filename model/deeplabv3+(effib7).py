import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class Effib7DeepLabV3PlusAllTrain(nn.Module):
    def __init__(self, in_channels = 3, classes = 12):
        super(Effib7DeepLabV3PlusAllTrain, self).__init__()
        self.backbone = smp.DeepLabV3Plus(
            encoder_name="timm-efficientnet-b7",     
            encoder_weights= "noisy-student",  
            in_channels=in_channels,               
            classes=classes
        )

    def forward(self, x):
        output = self.backbone(x)
        
        return output

if __name__ == '__main__':
    model = Effib7DeepLabV3PlusAllTrain(3,12)
    x = torch.randn([2, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x)
    print("output shape : ", out.size())