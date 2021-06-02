import torch
import torch.nn as nn
from torch.nn import functional as F

def conv3x3_relu(in_ch, out_ch, rate=1):
    conv3x3_relu = nn.Sequential(nn.Conv2d(in_ch, 
                                           out_ch, 
                                           kernel_size=3,
                                           stride=1,
                                           padding=rate,
                                           dilation=rate),
                                 nn.ReLU())
    return conv3x3_relu


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.conv1 = nn.Sequential(
            conv3x3_relu(3, 64),
            conv3x3_relu(64, 64),
            nn.MaxPool2d(3, stride=2, padding=1) # 1/2
        )
        
        self.conv2 = nn.Sequential(
            conv3x3_relu(64, 128),
            conv3x3_relu(128, 128),
            nn.MaxPool2d(3, stride=2, padding=1) # 1/2 
        )
        
        self.conv3 = nn.Sequential(
            conv3x3_relu(128, 256),
            conv3x3_relu(256, 256),
            conv3x3_relu(256, 256),
            nn.MaxPool2d(3, stride=2, padding=1) # 1/2
        )
        
        self.conv4 = nn.Sequential(
            conv3x3_relu(256, 512),
            conv3x3_relu(512, 512),
            conv3x3_relu(512, 512),
            nn.MaxPool2d(3, stride=1, padding=1)
        )
        
        self.conv5 = nn.Sequential(
            conv3x3_relu(512, 512, rate=2),
            conv3x3_relu(512, 512, rate=2),
            conv3x3_relu(512, 512, rate=2),
            nn.MaxPool2d(3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        
        return x

    
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=1024, num_classes=21):
        super(ASPP, self).__init__()
        
        # rate = 6
        self.conv_3x3_r6 =  nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size  = 3, stride = 1, padding = 6, dilation = 6),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        )
        
        # rate = 12
        self.conv_3x3_r12 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size  = 3, stride = 1, padding = 12, dilation = 12),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        )
        
        # rate = 18
        self.conv_3x3_r18 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size  = 3, stride = 1, padding = 18, dilation = 18),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
            
        )
        
        # rate = 24
        self.conv_3x3_r24 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size  = 3, stride = 1, padding = 24, dilation = 24),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(0.5),
            nn.Conv2d(out_channels, num_classes, kernel_size=1)
        )
        
        

    def forward(self, feature_map):
        out_img_r6 = self.conv_3x3_r6(feature_map)
        out_img_r12 = self.conv_3x3_r12(feature_map)
        out_img_r18 = self.conv_3x3_r18(feature_map)
        out_img_r24 = self.conv_3x3_r24(feature_map)
        
        out = sum([out_img_r6, out_img_r12, out_img_r18, out_img_r24])
        
        return out

class DeepLabV2(nn.Module):
    ## VGG 위에 ASPP 쌓기
    def __init__(self, backbone, classifier, upsampling=8):
        super(DeepLabV2, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.upsampling = upsampling

    def forward(self, x):
        x = self.backbone(x)
        _, _, feature_map_h, feature_map_w = x.size()
        x = self.classifier(x)
        out = F.interpolate(x, size=(feature_map_h * self.upsampling, feature_map_w * self.upsampling), mode="bilinear")
        return out

if __name__ == '__main__':
    backbone = VGG16()
    aspp_module = ASPP(in_channels=512, out_channels=256, num_classes = 12)
    model = DeepLabV2(backbone=backbone, classifier=aspp_module)
    
    x = torch.randn([1, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x)
    print("output shape : ", out.size())