import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, downsample=False, initial =False):
        super().__init__()
        self.initial = initial
        stride = 2 if downsample else 1
        if initial:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding)
        if downsample:
            if in_channels != out_channels:
                # change channels + spatial
                self.downsample = nn.Conv2d(in_channels, out_channels,
                                            kernel_size=1, stride=stride)
            else:
                # same channels, just downsample spatially
                self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        else:
            # no downsampling at all
            self.downsample = nn.Identity()
        self.act = nn.LeakyReLU(0.2, inplace=True)
    def forward(self, x):
    
        if self.initial:
            x1 = self.conv1(x)
            x2 = self.leaky_relu(x1)
            p = self.downsample(x)
            out = self.act(x2 + p)
            return out
        
        else:
            x1 = self.conv1(x)
            x2 = self.leaky_relu(x1)
            x3 = self.conv2(x2)
            p = self.downsample(x)
            out = self.act(x3 + p)
            return out 

class Net(nn.Module):
    def __init__(self, img_size, in_ch = 1):
        super(Net, self).__init__()
        H = img_size
        W = img_size
        self.block1 = Block(in_ch, 16, kernel_size=3, padding=1, initial=True, downsample=False)
        self.block2 = Block(16, 16, downsample=False)
        self.block3 = Block(16, 32, downsample=True)
        self.block4 = Block(32, 32, downsample=True)
        self.block5 = Block(32, 32, downsample=True)
        self.block6 = Block(32, 32, downsample=True)

        factor = 2 ** 4 
        factor_H = H // factor
        factor_W = W // factor
        factor_ch = 32
        self.fc= nn.Linear(factor_H * factor_W * factor_ch, 1)

    def forward(self, x):
        x = self.block1(x)
        for block in [self.block2, self.block3, self.block4, self.block5, self.block6]:
            x = block(x)   
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x_out = self.fc(x)
        return x_out
