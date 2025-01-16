import torch
import torch.nn as nn

from .conv import CBS, DWConv
from .component import C3K2, C2PSA, SPPF

class YoloBackbone(nn.Module):
    def __init__(self, ch=(256, 512, 1024)):
        super().__init__()
        self.cv1 = CBS(3, 64, 3, 2)
        self.cv2 = CBS(64, 128, 3, 2)
        self.c3k21 = C3K2(128, ch[0], c3k=False, e=0.25)
        self.cv3 = CBS(ch[0], ch[0], 3, 2) #P3
        self.c3k22 = C3K2(ch[0], ch[1], c3k=False, e=0.25)
        self.cv4 = CBS(ch[1], ch[1], 3, 2) #P4
        self.c3k23 = C3K2(ch[1], ch[1], c3k=True)
        self.cv5 = CBS(ch[1], ch[2], 3, 2)
        self.c3k24 = C3K2(ch[2], ch[2], c3k=True)
        self.sppf = SPPF(ch[2],ch[2], 5)
        self.psa  = C2PSA(ch[2], ch[2])

    def forward(self, x):
        x_out1 = self.cv3(self.c3k21(self.cv2(self.cv1(x))))
        x_out2 = self.cv4(self.c3k22(x_out1))
        x_out3 = self.psa(self.sppf(self.c3k24(self.cv5(self.c3k23(x_out2)))))
        return [x_out1, x_out2, x_out3]


class YoloNeck(nn.Module):
    def __init__(self, ch=(256, 512, 1024)):
        super().__init__()
        self.up1 = nn.Upsample(None,2,'nearest')
        self.c3k21 = C3K2(ch[2]+ch[1], ch[1], False)
        self.up2 = nn.Upsample(None,2,'nearest')
        self.c3k22 = C3K2(ch[1]+ch[0], ch[0], False)
        self.cv1 = CBS(ch[0], ch[0], 3, 2)
        self.c3k23 = C3K2(ch[1]+ch[0], ch[1], False)
        self.cv2 = CBS(ch[1], ch[1], 3, 2)
        self.c3k24 = C3K2(ch[1]+ch[2], ch[2], True)

    def forward(self, x):
        x_out0_t = x[2]
        x_out1_t = self.c3k21(torch.cat((self.up1(x[2]), x[1]), dim=1))
        x_out2_t = self.c3k22(torch.cat((self.up2(x_out1_t), x[0]), dim=1))

        x_out0 = x_out2_t
        x_out1 = self.c3k23(torch.cat((self.cv1(x_out0), x_out1_t), dim=1))
        x_out2 = self.c3k24(torch.cat((self.cv2(x_out1), x_out0_t), dim=1))
        return [x_out0, x_out1, x_out2]


class YoloHead(nn.Module):
    def __init__(self, nc=80, ch=(256, 512, 1024)):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.no = nc + 5
        c2, c3 = 32, 128  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(CBS(x, c2, 3), CBS(c2, c2, 3), nn.Conv2d(c2, 5, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), CBS(x, c3//2, 1)),
                    nn.Sequential(DWConv(c3//2, c3//2, 3), CBS(c3//2, c3//4, 1)),
                    nn.Conv2d(c3//4, self.nc, 1),
                )
                for x in ch
            )
        )

    def forward(self, x):
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1)
        return x

class Hyp:
    def __init__(self, box=6, cls=2.5, conf=1):
        self.box = box
        self.cls = cls
        self.conf = conf

class Yolov11(nn.Module):
    def __init__(self, nc=80, ch=(256, 512, 1024), stride=(8, 16, 32)):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.no = nc + 5  # number of outputs per anchor
        self.stride = stride  # strides computed during build
        self.backbone = YoloBackbone(ch)
        self.neck = YoloNeck(ch)
        self.head = YoloHead(nc=self.nc, ch=ch)
        self.hyp = Hyp()

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))