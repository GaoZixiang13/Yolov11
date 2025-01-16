import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

class CarObjectDataset(Dataset):
    def __init__(self, images, targets, input_shape, resize_shape, num_classes, M, train=True, cuda=True):
        super(CarObjectDataset, self).__init__()
        self.images = images
        self.targets = targets
        self.input_size = input_shape
        self.resize_shape = resize_shape
        self.num_classes = num_classes
        self.cuda = cuda
        self.eps = 1e-6
        self.M   = M
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

    def align(self, targets, M):
        nl = targets.shape[0]
        nums = M - nl
        tt = torch.zeros(nums, 5)
        targets = torch.cat((targets, tt), dim=0)
        return targets

    def image_preprocess(self, img_path, labels):
        img = Image.open('../DataSets/CarObject/data/training_images/'+img_path).convert('RGB')

        img = transforms.Pad((0,0,0,self.input_size[0]-self.input_size[1]), fill=0)(img)
        # --------------------------------------------------------
        # normal data augmentation
        img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img) #色域变换
        # --------------------------------------------------------
        img = img.resize((self.resize_shape, self.resize_shape), Image.BICUBIC)
        img = np.transpose(np.array(img)/255., (2, 0, 1)) #(c, w, h)
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

        if len(labels) > 0:
            y_t      = torch.tensor(labels).type(torch.FloatTensor)/self.input_size[0]*self.resize_shape
            y_cls    = torch.zeros(y_t.shape[0]).unsqueeze(-1)
            y_labels = torch.cat((y_cls, y_t), dim=-1)
        else:
            y_labels = torch.zeros(5)

        y_labels = self.align(y_labels, self.M)

        return img_tensor, y_labels

    def image_preprocess_test(self, img_path, labels):
        img = Image.open('../DataSets/CarObject/data/training_images/'+img_path).convert('RGB')
        img = transforms.Pad((0,0,0,self.input_size[0]-self.input_size[1]), fill=0)(img)

        img = img.resize((self.resize_shape, self.resize_shape), Image.BICUBIC)
        img = np.transpose(np.array(img) / 255., (2, 0, 1))
        img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

        y_labels = torch.tensor(labels).type(torch.FloatTensor)/self.input_size[0]*self.resize_shape

        return img_tensor, y_labels


