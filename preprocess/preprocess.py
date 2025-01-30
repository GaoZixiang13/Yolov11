import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from PIL import Image

class CarObjectDataset(Dataset):
    def __init__(self, images, targets, resize_shape, num_classes, max_pt, train=True, cuda=True):
        super(CarObjectDataset, self).__init__()
        self.images = images
        self.targets = targets
        self.resize_shape = resize_shape
        self.num_classes = num_classes
        self.cuda = cuda
        self.max_pt = max_pt
        self.max_pt_this_batch = 0
        self.eps = 1e-6
        self.train = train

    def __getitem__(self, index):
        x = self.images[index]
        y = self.targets[index]
        if self.train:
            x, y = self.image_preprocess(x, y)

        return x, y

    def __len__(self):
        return len(self.images)

    def align(self, targets, M): # 对 齐
        nl = targets.shape[0]
        nums = M - nl
        tt = torch.zeros(nums, 5)
        targets = torch.cat((targets, tt), dim=0)
        return targets

    def image_preprocess(self, img_path, labels):
        img = Image.open('D:/PyCharm项目/yolov11/DataSets/CarObject/data/training_images/'+img_path).convert('RGB')
        w, h = img.size

        img = transforms.Pad((0, 0, self.resize_shape - w, self.resize_shape - h), fill=0)(img)
        # --------------------------------------------------------
        # normal data augmentation
        img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img) #色域变换
        # --------------------------------------------------------
        img = np.transpose(np.array(img)/255., (2, 0, 1)) #(c, w, h)
        img_tensor = torch.from_numpy(img).float()

        if len(labels) > 0:
            y_label, y_cls = torch.tensor(labels).float().split((4, 1), dim=-1)
            x1y1, wh = y_label.chunk(2, dim=-1)
            x2y2 = x1y1 + wh
            y_label = torch.cat((x1y1, x2y2), dim=-1)/self.resize_shape
            y_labels  = torch.cat((y_cls, y_label), dim=-1)
        else:
            y_labels = torch.zeros(1, 5)

        y_labels = self.align(y_labels, self.max_pt)

        return img_tensor, y_labels


