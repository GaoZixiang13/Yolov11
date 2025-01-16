import torch
import torch.nn.functional as F

a = torch.tensor([[0, 1, 1, 0, 2]]).float()
b = torch.tensor([0, 1, 2]).long()

print(a)
a[b] = 0
print(a)