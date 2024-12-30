import torch


# 创建一个一维张量
tensor = torch.randn((1, 1, 4, 4))

y = [tensor for _ in range(4)]
print(y)

torch.cat(y, dim=1)