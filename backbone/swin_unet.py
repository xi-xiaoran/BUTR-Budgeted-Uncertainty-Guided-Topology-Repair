import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvDown(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2,2)
    def forward(self, x):
        s = self.conv(x)
        return s, self.pool(s)

class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch//2, 2, 2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch//2+skip_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x, skip):
        x = self.up(x)
        dy = skip.size(2)-x.size(2); dx = skip.size(3)-x.size(3)
        x = F.pad(x, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([skip, x], 1))

class SwinUNetLite(nn.Module):
    def __init__(self, in_ch=3, base=32, head_out_ch=1):
        super().__init__()
        self.d1=ConvDown(in_ch, base)
        self.d2=ConvDown(base, base*2)
        self.d3=ConvDown(base*2, base*4)
        self.d4=ConvDown(base*4, base*8)
        self.b=nn.Sequential(nn.Conv2d(base*8, base*16, 3,1,1), nn.ReLU(inplace=True),
                             nn.Conv2d(base*16, base*16, 3,1,1), nn.ReLU(inplace=True))
        self.u1=Up(base*16, base*8, base*8)
        self.u2=Up(base*8, base*4, base*4)
        self.u3=Up(base*4, base*2, base*2)
        self.u4=Up(base*2, base, base)
        self.out=nn.Conv2d(base, head_out_ch, 1)
    def forward(self, x):
        s1,x=self.d1(x)
        s2,x=self.d2(x)
        s3,x=self.d3(x)
        s4,x=self.d4(x)
        x=self.b(x)
        x=self.u1(x,s4); x=self.u2(x,s3); x=self.u3(x,s2); x=self.u4(x,s1)
        return self.out(x)
