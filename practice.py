import torch
import torch.nn.functional as F

a = torch.tensor([[0, 5, 1]]).float()
b = torch.tensor([[0, 5, 2]]).float()

print(a*b)