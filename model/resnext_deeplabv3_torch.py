import torchvision.models as models
import torch
import torch.nn as nn
from torch.nn import functional as F
import segmentation_models_pytorch as smp

def conv3x3_relu(in_ch, out_ch, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(in_ch, 
                                           out_ch,
                                           kernel_size=3, 
                                           stride=1,
                                           padding=rate,
                                           dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu

class ASPPConv(nn.Module):
    def __init__(self, inplanes, outplanes, kernel_size, padding, dilation):
        super(ASPPConv, self).__init__()
        self.atrous_conv = nn.Sequential(
            nn.Conv2d(inplanes, outplanes, kernel_size = kernel_size, 
                      stride = 1, padding = padding, dilation=dilation, bias = False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )
        

    def forward(self, x):
        out = self.atrous_conv(x)
        
        return out
    
    
class ASPPPooling(nn.Module):
    def __init__(self, inplanes, outplanes):
        super(ASPPPooling, self).__init__()
        self.image_pool = nn.Sequential(
            nn.AdaptiveMaxPool2d((1,1)),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, stride = 1, bias= False),
            nn.BatchNorm2d(outplanes),
            nn.ReLU()
        )
        

    def forward(self, x):
        out = self.image_pool(x)
        return out

    
class ASPP(nn.Module):
    def __init__(self, inplanes, outplanes,atrous_rates):
        super(ASPP, self).__init__()
        rates = atrous_rates
        
        self.aspp1 = ASPPConv(inplanes, outplanes, 1, padding=0, dilation=rates[0])
        self.aspp2 = ASPPConv(inplanes, outplanes, 3, padding=rates[1], dilation=rates[1])
        self.aspp3 = ASPPConv(inplanes, outplanes, 3, padding=rates[2], dilation=rates[2])
        self.aspp4 = ASPPConv(inplanes, outplanes, 3, padding=rates[3], dilation=rates[3])
        
        self.global_avg_pool = ASPPPooling(inplanes, outplanes)
        
        # concat후 다시 채널 수를 맞춰주기 위한 작업
        self.project = nn.Sequential(
            nn.Conv2d(outplanes*5, outplanes, 1, bias=False), 
            nn.BatchNorm2d(outplanes), 
            nn.ReLU(), 
            nn.Dropout(0.5)      
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True) # image pooling부분은 원본 크기만큼 upsampling
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        
        out = self.project(x)
        return out
    
class DeepLabHead(nn.Sequential):
    def __init__(self, in_ch, out_ch, n_classes, atrous_rates):
        super(DeepLabHead, self).__init__()
        self.add_module("0", ASPP(in_ch, out_ch,atrous_rates))
        self.add_module("1", nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1 , bias=False)) # passing convolution
        self.add_module("2", nn.BatchNorm2d(out_ch))
        self.add_module("3", nn.ReLU())
        self.add_module("4", nn.Conv2d(out_ch, n_classes, kernel_size=1, stride=1)) # classification

class ResNextDeepLabV3EncoderPretrain(nn.Sequential):
    """인코더 부분만 pretrain된 모델
    """
    def __init__(self, n_classes, atrous_rates):
        super(ResNextDeepLabV3EncoderPretrain, self).__init__()
        backbone = models.resnext101_32x8d(pretrained=True)
        self.encoder = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
            backbone.layer4
        )
        self.decoder = DeepLabHead(in_ch=2048, out_ch=256, n_classes=12,atrous_rates=atrous_rates)

    def forward(self, x):
        h = self.encoder(x)
        h = self.decoder(h)
        output = F.interpolate(h, size=x.shape[2:], mode="bilinear", align_corners=False)
        
        return output

class ResNextDeepLabV3AllTrain(nn.Module):
    def __init__(self, in_channels = 3, classes = 12):
        super(ResNextDeepLabV3AllTrain, self).__init__()
        self.backbone = smp.DeepLabV3(
            encoder_name="resnext101_32x8d",
            encoder_weights= "imagenet",
            in_channels=in_channels,
            classes=classes
            )

    def forward(self, x):
        output = self.backbone(x)
        
        return output

if __name__ == '__main__':
    model = ResNextDeepLabV3EncoderPretrain(12,[1, 12, 24, 36])
    x = torch.randn([2, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x)
    print("output shape : ", out.size())

    model = ResNextDeepLabV3AllTrain(3,12)
    x = torch.randn([2, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x)
    print("output shape : ", out.size())
