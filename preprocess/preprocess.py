from random import random

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

class CarObjectDataset(Dataset):
    def __init__(self, images, targets, input_shape, num_classes, train=True, cuda=True):
        super(CarObjectDataset, self).__init__()
        self.images = images
        self.targets = targets
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.cuda = cuda
        self.eps = 1e-6
        self.train = train

    def __getitem__(self, index):
        tx = self.images[index]
        ty = self.targets[index]
        ty = np.array(ty)
        if self.train:
            x, y = self.image_preprocess(tx, ty)
        else:
            x, y = self.image_preprocess_test(tx, ty)

        return x, y

    def __len__(self):
        return len(self.images)

    def image_preprocess(self, img_path, labels):
        img = Image.open('../DataSet/' + img_path).convert('RGB')
        # --------------------------------------------------------
        # normal data augmentation
        img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img) #色域变换
        img = transforms.RandomHorizontalFlip(p=0.5)(img) #水平翻转
        # --------------------------------------------------------

        img = img.resize((self.input_shape, self.input_shape), Image.BICUBIC)
        img = np.transpose(np.array(img)/255., (2, 0, 1))
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

        y_labels = torch.zeros(self.num_classes)
        y_labels[labels] = 1

        return img_tensor, y_labels

    def image_preprocess_test(self, img_path, labels):
        img = Image.open('../DataSet/' + img_path).convert('RGB')

        img = img.resize((self.input_shape, self.input_shape), Image.BICUBIC)
        img = np.transpose(np.array(img) / 255., (2, 0, 1))
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

        y_labels = torch.zeros(self.num_classes)
        y_labels[labels] = 1

        return img_tensor, y_labels


