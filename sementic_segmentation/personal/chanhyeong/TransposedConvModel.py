import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
class smpModel(nn.Module):
    def __init__(self,encoder_name="resnext101_32x8d",decoder_name="DeepLabV3Plus",pretrain_weight="imagenet",num_classes=12):
        super(smpModel,self).__init__()
        if(decoder_name=="DeepLabV3Plus"):
            self.backbone=smp.DeepLabV3Plus(encoder_name=encoder_name,classes=12)
            self.backbone.decoder.up=nn.ConvTranspose2d(256,256,4,stride=4)
            self.backbone.segmentation_head[1]=nn.ConvTranspose2d(12,12,4,stride=4)
        elif(decoder_name=="Unet"):
            self.backbone=smp.Unet(encoder_name=encoder_name,classes=12)
    def forward(self,x):
        x=self.backbone(x)
        return x
        
        