import torch
import torch.nn as nn

from .conv import CBS

__all__ = (
    "DFL",
    "C3K2",
    "C2PSA",
    "SPPF",
)
class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)

class BottleNeck(nn.Module):
    def __init__(self, c1, c2, e=0.5, g=1, k=(3,3), shortcut=True):
        super().__init__()
        c_ = int(c2*e)
        self.cv1 = CBS(c1, c_, k[0], 1)
        self.cv2 = CBS(c_, c2, k[1], 1, g=g)
        self.shortAdd = c1==c2 and shortcut

    def forward(self, x):
        x_out = self.cv2(self.cv1(x))
        return x + x_out if self.shortAdd else x_out

class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, g=1):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = CBS(c1, c_, 1, 1)
        self.cv2 = CBS(c1, c_, 1, 1)
        self.cv3 = CBS(c_*2, c2, 1, 1)
        self.m = nn.Sequential(
            *(BottleNeck(c_, c_, shortcut=shortcut, e=1.0, g=g, k=((1, 1), (3, 3))) for _ in range(n))
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.cv2(x), self.m(self.cv1(x))), dim=1))

class C3K(C3):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5, g=1, k=3):
        super().__init__(c1, c2, n, shortcut, e, g)
        c_ = int(c2 * e)
        self.m = nn.Sequential(
            *[BottleNeck(c_, c_, g=g, k=(k,k), e=1.0, shortcut=shortcut) for _ in range(n)]
        )

    def forward(self, x):
        return self.cv3(torch.cat((self.cv2(x), self.m(self.cv1(x))), dim=1))

class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.ce = int(c2*e)
        self.cv1 = CBS(c1, 2 * self.ce, 1, 1)
        self.cv2 = CBS((2 + n) * self.ce, c2, 1, 1)
        self.m = nn.ModuleList(
            BottleNeck(self.ce, self.ce, e=1.0, g=g, k=((3, 3), (3, 3)), shortcut=shortcut) for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3K2(C2f):
    def __init__(self, c1, c2, c3k=False, n=1, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2*e)
        self.m = nn.ModuleList(
            C3K(c_, c_, n=2, shortcut=shortcut, g=g) if c3k else BottleNeck(c_, c_, g=g, shortcut=shortcut) for _ in range(n)
        )

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = CBS(dim, h, 1, act=False)
        self.proj = CBS(dim, dim, 1, act=False)
        self.pe = CBS(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSABlock(nn.Module):
    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True):
        super().__init__()
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads= num_heads)
        self.ffn = nn.Sequential(
            CBS(c, c*2),
            CBS(c*2, c, act=False)
        )
        self.shortcut = shortcut

    def forward(self, x):
        x = x + self.attn(x) if self.shortcut else self.attn(x)
        return x + self.ffn(x) if self.shortcut else self.ffn(x)

class C2PSA(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1*e)
        self.cv1 = CBS(c1, 2*self.c, 1)
        self.cv2 = CBS(2*self.c, c2, 1)
        self.m = nn.Sequential(
            *(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))
        )

    def forward(self, x):
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), dim=1))

class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        self.c_ = c1 // 2
        self.cv1 = CBS(c1, self.c_, 1, 1)
        self.cv2 = CBS(self.c_*4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k//2)

    def forward(self, x):
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

