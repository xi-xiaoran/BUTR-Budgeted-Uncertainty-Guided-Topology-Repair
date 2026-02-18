import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import ConvBNReLU, OutHead


class VGGBlock(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, mid_ch)
        self.conv2 = ConvBNReLU(mid_ch, out_ch)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class UNetPP(nn.Module):
    """UNet++ (Nested U-Net) backbone.

    IMPORTANT: We always upsample to match the spatial size of the skip tensor
    we concatenate with (instead of scale_factor=2). This guarantees that the
    output stays aligned with the input even when H/W are not divisible by 2**depth.
    """

    def __init__(self, in_ch=3, base=32, head_out_ch=1):
        super().__init__()
        nb = [base, base * 2, base * 4, base * 8, base * 16]
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = VGGBlock(in_ch, nb[0], nb[0])
        self.conv1_0 = VGGBlock(nb[0], nb[1], nb[1])
        self.conv2_0 = VGGBlock(nb[1], nb[2], nb[2])
        self.conv3_0 = VGGBlock(nb[2], nb[3], nb[3])
        self.conv4_0 = VGGBlock(nb[3], nb[4], nb[4])

        self.conv0_1 = VGGBlock(nb[0] + nb[1], nb[0], nb[0])
        self.conv1_1 = VGGBlock(nb[1] + nb[2], nb[1], nb[1])
        self.conv2_1 = VGGBlock(nb[2] + nb[3], nb[2], nb[2])
        self.conv3_1 = VGGBlock(nb[3] + nb[4], nb[3], nb[3])

        self.conv0_2 = VGGBlock(nb[0] * 2 + nb[1], nb[0], nb[0])
        self.conv1_2 = VGGBlock(nb[1] * 2 + nb[2], nb[1], nb[1])
        self.conv2_2 = VGGBlock(nb[2] * 2 + nb[3], nb[2], nb[2])

        self.conv0_3 = VGGBlock(nb[0] * 3 + nb[1], nb[0], nb[0])
        self.conv1_3 = VGGBlock(nb[1] * 3 + nb[2], nb[1], nb[1])

        self.conv0_4 = VGGBlock(nb[0] * 4 + nb[1], nb[0], nb[0])

        self.outc = OutHead(nb[0], head_out_ch)

    @staticmethod
    def _up_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[2:], mode="bilinear", align_corners=False)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self._up_to(x1_0, x0_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self._up_to(x2_0, x1_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self._up_to(x1_1, x0_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self._up_to(x3_0, x2_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self._up_to(x2_1, x1_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self._up_to(x1_2, x0_2)], 1))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self._up_to(x4_0, x3_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self._up_to(x3_1, x2_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self._up_to(x2_2, x1_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self._up_to(x1_3, x0_3)], 1))

        return self.outc(x0_4)
