import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.LeakyReLU(0.2, inplace=True) 
    def forward(self, x):
        x = self.conv(x)
        # x = self.bn(x)
        x = self.act(x)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1):
        super(Encoder, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, stride, padding)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.downsample = ConvBlock(out_channels, out_channels, stride=2)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        p = self.downsample(x)
        return x, p

# Define the Decoder class
class Decoder(nn.Module):
    def __init__(self, skip_channels, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.skip_channels = skip_channels
        self.conv_up = ConvBlock(in_channels, out_channels)
        self.conv1 = ConvBlock(skip_channels + out_channels, out_channels)
    def forward(self, x, skip ,dropout = None):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv_up(x)
        if dropout:
            x = F.dropout(x, p=0.2, training=self.training)
        x = torch.cat((x, skip), dim=1)
        x = self.conv1(x)
        return x
    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()

        # Encoder
        self.encoder1 = Encoder(3, 32, kernel_size=5, stride=1, padding=2)
        self.encoder2 = Encoder(32, 64)
        self.encoder3 = Encoder(64, 128)
        self.encoder4 = Encoder(128, 128)
        self.encoder5 = Encoder(128, 128)
        
        # Bottleneck
        self.bottleneck = ConvBlock(128, 128)

        # Decoder
        self.decoder5 = Decoder(128 ,128, 128)
        self.decoder4 = Decoder(128 ,128, 128)
        self.decoder3 = Decoder(128 ,128, 128)
        self.decoder2 = Decoder(64, 128, 64)
        self.decoder1 = Decoder(32, 64, 32)


        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)
    def forward(self, x):
        # Encoder
        skip1, p1 = self.encoder1(x)
        skip2, p2 = self.encoder2(p1)
        skip3, p3 = self.encoder3(p2)
        skip4, p4 = self.encoder4(p3)
        skip5, p5 = self.encoder5(p4)
        
        # Bottleneck
        b = self.bottleneck(p5)

        # Decoder

        d5 = self.decoder5(b, skip5)
        d4 = self.decoder4(d5, skip4)
        d3 = self.decoder3(d4, skip3)
        d2 = self.decoder2(d3, skip2, dropout=True)
        d1 = self.decoder1(d2, skip1,dropout=True)
        # Final convolution
        out = self.final_conv(d1)
        return out