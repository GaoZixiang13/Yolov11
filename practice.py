import torch
import torch.nn.functional as F

a = torch.tensor([[1, 2, 3]]).float()/10
b = torch.tensor([[1, 0, 1]]).float()

c = F.binary_cross_entropy(a, b, reduction="none")

print(c)

