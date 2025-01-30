import torch
import torch.nn.functional as F

import json
from PIL import Image, ImageDraw
from torchvision import transforms

def draw_bbox(image_path, box, xywh=False):
    image = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(image)

    w, h = image.size
    print(w, h)

    b = box
    if xywh:
        x1y1, wh = torch.tensor(box).chunk(2, -1)
        x2y2 = x1y1 + wh
        draw.rectangle([x1y1[0], x1y1[1], x2y2[0], x2y2[1]], outline='red', width=2)
    else:
        draw.rectangle([b[0], b[1], b[2], b[3]], outline='red', width=2)

    image.show()

def img_show(image_path, pad=True):
    img = Image.open(image_path).convert('RGB')

    if pad:
        w, h = img.size
        img = transforms.Pad((0, 0, 640 - w, 640 - h), fill=0)(img)

    img.show()

if __name__ == "__main__":
    img_show('D:/PyCharm项目/CocoDataSet/Images/train2017/000000000049.jpg')
    # draw_bbox('D:/PyCharm项目/CocoDataSet/Images/train2017/000000000049.jpg', [124.07, 126.8, 252.95, 297.19], xywh=True)