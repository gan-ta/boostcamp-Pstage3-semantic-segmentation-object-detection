import torch
import torch.nn as nn

class SegNet(nn.Module):
    def __init__(self, num_classes=12, init_weights=True):
        super(SegNet, self).__init__()
        def CBR(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
            layers = []
            layers += [
                nn.Conv2d(in_channels = in_channels,
                         out_channels = out_channels,
                         kernel_size = kernel_size,
                         stride = stride,
                         padding = padding)
            ]
            
            layers += [nn.BatchNorm2d(num_features = out_channels)]
            layers += [nn.ReLU()]
            
            cbr = nn.Sequential(*layers)
            return cbr
        
        # ===========================Encoder===========================
        # conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True, return_indices = True) # ceil mode : 길이가 잘리면 그 안에서 pooling해옴
        
        # conv2
        self.conv2_1 = CBR(64, 128 ,3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 =  nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True, return_indices = True)
        
        # conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True, return_indices = True)
        
        # conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True, return_indices = True)
        
        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(kernel_size = 2, stride = 2, ceil_mode = True, return_indices = True)
        
        # ===========================Decoder===========================
        # deconv5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.dconv5_3 = CBR(512, 512, 3, 1, 1)
        self.dconv5_2 = CBR(512, 512, 3, 1, 1)
        self.dconv5_1 = CBR(512, 512, 3, 1, 1)

        # deconv4 
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.dconv4_3 = CBR(512, 512, 3, 1, 1)
        self.dconv4_2 = CBR(512, 512, 3, 1, 1)
        self.dconv4_1 = CBR(512, 256, 3, 1, 1)

        # deconv3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.dconv3_3 = CBR(256, 256, 3, 1, 1)
        self.dconv3_2 = CBR(256, 256, 3, 1, 1)
        self.dconv3_1 = CBR(256, 128, 3, 1, 1)

        # deconv2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.dconv2_2 = CBR(128, 128, 3, 1, 1)
        self.dconv2_1 = CBR(128, 64, 3, 1, 1)
        
        # deconv1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.dconv1_1 = CBR(64, 64, 3, 1, 1)
        
        
        self.score_fr = nn.Conv2d(64, num_classes, kernel_size = 3, padding=1)
        
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)

                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # ===========================Encoder===========================
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        dim1 = h.size()
        h,pool1_indices = self.pool1(h)
        
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        dim2 = h.size()
        h, pool2_indices = self.pool2(h)
        
        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)
        dim3 = h.size()
        h, pool3_indices = self.pool3(h)
        
        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)
        dim4 = h.size()
        h, pool4_indices = self.pool4(h)
        
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)
        dim5 = h.size()
        h, pool5_indices = self.pool5(h)
        
        # ===========================Decoder===========================
        h = self.unpool5(h, pool5_indices, output_size = dim5)
        h = self.dconv5_3(h)
        h = self.dconv5_2(h)
        h = self.dconv5_1(h)
        
        h = self.unpool4(h, pool4_indices, output_size = dim4)
        h = self.dconv4_3(h)
        h = self.dconv4_2(h)
        h = self.dconv4_1(h)
        
        h = self.unpool3(h, pool3_indices, output_size = dim3)
        h = self.dconv3_3(h)
        h = self.dconv3_2(h)
        h = self.dconv3_1(h)
        
        h = self.unpool2(h, pool2_indices, output_size = dim2)
        h = self.dconv2_2(h)
        h = self.dconv2_1(h)
        
        h = self.unpool1(h, pool1_indices, output_size = dim1)
        h = self.dconv1_1(h)
        out = self.score_fr(h) 
        
        
        return out

if __name__ == '__main__':
    model = SegNet(num_classes=12)
    x = torch.randn([1, 3, 512, 512])
    print("input shape : ", x.shape)
    out = model(x)
    print("output shape : ", out.size())