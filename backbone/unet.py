import torch
import torch.nn as nn
from .blocks import DoubleConv, Up, OutHead

class UNet(torch.nn.Module):
    def __init__(self, in_ch=3, base=32, head_out_ch=1):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*8, base*16))
        self.up1 = Up(base*16, base*8)
        self.up2 = Up(base*8, base*4)
        self.up3 = Up(base*4, base*2)
        self.up4 = Up(base*2, base)
        self.outc = OutHead(base, head_out_ch)
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
