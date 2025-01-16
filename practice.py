import torch
import torch.nn.functional as F

a = torch.tensor([[1], [2], [3], [4], [5], [6]])
b = a.squeeze(-1)

print(a, b)