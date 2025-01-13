import math

import torch
import torch.nn as nn

__all__ = (
    "CBS",
    "LightConv",
    "DWConv",
    "Focus",
)

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class CBS(nn.Module):
    default_act = nn.SiLU()
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, kernel_size=k, stride=s, padding=autopad(k, p, d), dilation=d, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWConv(CBS):
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, act=True):
        super().__init__(c1, c2, k, s, p, d, g=math.gcd(c1, c2), act=act)

class LightConv(nn.Module):
    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        super().__init__()
        self.cv1 = CBS(c1, c2, 1, act=False)
        self.cv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        return self.cv2(self.cv1(x))

class Focus(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, d=1, g=1, act=True):
        super().__init__()
        self.cv = CBS(c1*4, c2, k, s, p ,d, g, act)

    def forward(self, x):
        return self.cv(torch.cat((x[..., ::2, ::2], x[..., ::2, 1::2],
                                 x[..., 1::2, ::2], x[..., 1::2, 1::2]), dim=1))