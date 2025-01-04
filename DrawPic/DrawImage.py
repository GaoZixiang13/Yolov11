import numpy as np
import matplotlib.pyplot as plt
import torch

def silu(x):
    return x * torch.sigmoid(x)

def drawLoss(x, y, train=True):
    plt.plot(x, y)
    plt.xlabel('epoch')
    if train:
        plt.ylabel('trainLoss')
        plt.title('train loss each epoch')
    else:
        plt.ylabel('ValLoss')
        plt.title('Val loss each epoch')
    plt.grid()
    plt.show()

x = np.linspace(-10, 10, 3000)
y = [silu(torch.tensor(i)) for i in x]
