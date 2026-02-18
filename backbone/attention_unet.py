import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import DoubleConv, OutHead

class AttentionGate(nn.Module):
    def __init__(self, g_ch, x_ch, inter_ch):
        super().__init__()
        self.Wg = nn.Sequential(nn.Conv2d(g_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.Wx = nn.Sequential(nn.Conv2d(x_ch, inter_ch, 1, bias=False), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.Conv2d(inter_ch, 1, 1, bias=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
    def forward(self, g, x):
        psi = self.relu(self.Wg(g) + self.Wx(x))
        a = self.psi(psi)
        return x * a

class AttnUp(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2)
        self.attn = AttentionGate(in_ch//2, skip_ch, max(8, skip_ch//2))
        self.conv = DoubleConv(in_ch//2 + skip_ch, out_ch)
    def forward(self, x, skip):
        x = self.up(x)
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        skip = self.attn(x, skip)
        return self.conv(torch.cat([skip, x], 1))

class AttentionUNet(nn.Module):
    def __init__(self, in_ch=3, base=32, head_out_ch=1):
        super().__init__()
        self.inc = DoubleConv(in_ch, base)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base, base*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*2, base*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*4, base*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base*8, base*16))
        self.up1 = AttnUp(base*16, base*8, base*8)
        self.up2 = AttnUp(base*8, base*4, base*4)
        self.up3 = AttnUp(base*4, base*2, base*2)
        self.up4 = AttnUp(base*2, base, base)
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
