import torch
import torch.nn as nn
from torch.nn import functional as F

def conv_relu(in_ch, out_ch, size=3, rate=1):
    conv_relu = nn.Sequential(nn.Conv2d(in_channels=in_ch, 
                                        out_channels=out_ch,
                                        kernel_size=size, 
                                        stride=1, 
                                        padding=rate, 
                                        dilation=rate),
                              nn.ReLU())
    return conv_relu


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features1 = nn.Sequential(conv_relu(3, 64, 3, 1),
                                       conv_relu(64, 64, 3, 1),
                                       nn.MaxPool2d(2, stride=2, padding=0))
        self.features2 = nn.Sequential(conv_relu(64, 128, 3, 1),
                                       conv_relu(128, 128, 3, 1),
                                       nn.MaxPool2d(2, stride=2, padding=0))
        self.features3 = nn.Sequential(conv_relu(128, 256, 3, 1),
                                       conv_relu(256, 256, 3, 1),
                                       conv_relu(256, 256, 3, 1),
                                       nn.MaxPool2d(2, stride=2, padding=0))
        self.features4 = nn.Sequential(conv_relu(256, 512, 3, 1),
                                       conv_relu(512, 512, 3, 1),
                                       conv_relu(512, 512, 3, 1))
        self.features5 = nn.Sequential(conv_relu(512, 512, 3, rate=2),
                                       conv_relu(512, 512, 3, rate=2),
                                       conv_relu(512, 512, 3, rate=2))
                                       
    def forward(self, x):
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)
        out = self.features4(out)
        out = self.features5(out)
        return out

class classifier(nn.Module):
    def __init__(self, num_classes): 
        super(classifier, self).__init__()
        self.classifier = nn.Sequential(nn.Conv2d(512, 4096, kernel_size=7, dilation=4, padding=12), 
                                        nn.ReLU(),
                                        nn.Dropout2d(0.5),
                                        nn.Conv2d(4096, 4096, kernel_size=1),
                                        nn.ReLU(),
                                        nn.Dropout2d(0.5),
                                        nn.Conv2d(4096, num_classes, kernel_size=1)
                                        )

    def forward(self, x): 
        out = self.classifier(x)
        return out

class BasicContextModule(nn.Module):
    def __init__(self, num_classes):
        super(BasicContextModule, self).__init__()
        
        self.layer1 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 1))
        self.layer2 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 1))
        self.layer3 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 2))
        self.layer4 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 4))
        self.layer5 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 8))
        self.layer6 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 16))
        self.layer7 = nn.Sequential(conv_relu(num_classes, num_classes, 3, 1))
        # No Truncation 
        self.layer8 = nn.Sequential(nn.Conv2d(num_classes, num_classes, 1, 1))
        
    def forward(self, x): 
        
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        
        return out
    
class DilatedNet(nn.Module):
    def __init__(self, backbone, classifier, context_module):
        super(DilatedNet, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.context_module = context_module
        
        self.deconv = nn.ConvTranspose2d(in_channels=12,
                                         out_channels=12,
                                         kernel_size=16,
                                         stride=8,
                                         padding=4)
        
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        x = self.context_module(x)
        out = self.deconv(x)
        return out

if __name__ == '__main__':
    backbone = VGG16()
    classifier = classifier(num_classes=12)
    context_module = BasicContextModule(num_classes=12)
    model = DilatedNet(backbone=backbone, classifier=classifier, context_module=context_module)
    
    x = torch.randn([1, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x)
    print("output shape : ", out.size())