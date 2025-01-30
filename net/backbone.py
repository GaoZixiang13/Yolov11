import torch
import torch.nn as nn
import math, copy

from .conv import CBS, DWConv
from .component import C3K2, C2PSA, SPPF, DFL
from utils.tal import dist2bbox, make_anchors, bbox2dist

class YoloBackbone(nn.Module):
    def __init__(self, ch=(128, 256, 512)):
        super().__init__()
        self.cv1 = CBS(3, 32, 3, 2)
        self.cv2 = CBS(32, 64, 3, 2)
        self.c3k21 = C3K2(64, ch[0], n=2, c3k=False)
        self.cv3 = CBS(ch[0], ch[0], 3, 2) #P3
        self.c3k22 = C3K2(ch[0], ch[1], n=2, c3k=False)
        self.cv4 = CBS(ch[1], ch[1], 3, 2) #P4
        self.c3k23 = C3K2(ch[1], ch[1], n=2, c3k=True)
        self.cv5 = CBS(ch[1], ch[2], 3, 2)
        self.c3k24 = C3K2(ch[2], ch[2], n=2, c3k=True)
        self.sppf = SPPF(ch[2],ch[2], 5)
        self.psa  = C2PSA(ch[2], ch[2])

    def forward(self, x):
        x_out1 = self.cv3(self.c3k21(self.cv2(self.cv1(x))))
        x_out2 = self.cv4(self.c3k22(x_out1))
        x_out3 = self.psa(self.sppf(self.c3k24(self.cv5(self.c3k23(x_out2)))))
        return [x_out1, x_out2, x_out3]


class YoloNeck(nn.Module):
    def __init__(self, ch=(128, 256, 512)):
        super().__init__()
        self.up1 = nn.Upsample(None,2,'nearest')
        self.c3k21 = C3K2(ch[2]+ch[1], ch[1], False, 2)
        self.up2 = nn.Upsample(None,2,'nearest')
        self.c3k22 = C3K2(ch[1]+ch[0], ch[0], False, 2)
        self.cv1 = CBS(ch[0], ch[0], 3, 2)
        self.c3k23 = C3K2(ch[1]+ch[0], ch[1], False, 2)
        self.cv2 = CBS(ch[1], ch[1], 3, 2)
        self.c3k24 = C3K2(ch[1]+ch[2], ch[2], True, 2)

    def forward(self, x):
        x_out0_t = x[2]
        x_out1_t = self.c3k21(torch.cat((self.up1(x[2]), x[1]), dim=1))
        x_out2_t = self.c3k22(torch.cat((self.up2(x_out1_t), x[0]), dim=1))

        x_out0 = x_out2_t
        x_out1 = self.c3k23(torch.cat((self.cv1(x_out0), x_out1_t), dim=1))
        x_out2 = self.c3k24(torch.cat((self.cv2(x_out1), x_out0_t), dim=1))
        return [x_out0, x_out1, x_out2]


class YoloHead(nn.Module):
    def __init__(self, nc=80, ch=(128, 256, 512)):
        super().__init__()
        self.nc = nc  # number of classes
        self.legacy = False
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = (8, 16, 32)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(CBS(x, c2, 3), CBS(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(CBS(x, c3, 3), CBS(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), CBS(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), CBS(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )

        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.one2one_cv2 = copy.deepcopy(self.cv2)
        self.one2one_cv3 = copy.deepcopy(self.cv3)

    def forward(self, x):
        one2one = self.forward_feat([xi.detach() for xi in x], self.one2one_cv2, self.one2one_cv3)
        one2many = self.forward_feat(x, self.cv2, self.cv3)
        return {'one2one':one2one, 'one2many':one2many}

    def forward_feat(self, x, cv2, cv3):
        y = []
        for i in range(self.nl):
            y.append(torch.cat((cv2[i](x[i]), cv3[i](x[i])), 1))
        return y

class Hyp:
    def __init__(self, box=15, cls=0.5, dfl=3):
        self.box = box
        self.cls = cls
        self.dfl = dfl

class Yolov11(nn.Module):
    def __init__(self, nc=80, ch=(128, 256, 512), stride=(8, 16, 32)):
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = 16  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = stride  # strides computed during build
        self.backbone = YoloBackbone(ch)
        self.neck = YoloNeck(ch)
        self.head = YoloHead(nc=self.nc, ch=ch)
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.hyp = Hyp()

    def forward(self, x):
        return self.head(self.neck(self.backbone(x)))