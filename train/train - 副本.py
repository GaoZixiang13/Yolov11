import torch, time
from torch.utils.data import Dataset, DataLoader, random_split

from preprocess.preprocess import CarObjectDataset
from utils.fit import fit_one_epoch
from net.backbone import Yolov11
from utils.loss11 import E2EDetectLoss
from DrawPic.DrawImage import drawLoss

print(torch.version.cuda)
print(torch.__file__)

# ---------------------------------------------------------------
# Hyper Parameters
BATCH_SIZE = 4
LR = 5e-4
weight_decay = LR/BATCH_SIZE
EPOCH = 60
# 网络输入图片size
RE_SIZE_shape = 640
# 总的类别数
num_classes = 1

trainSize = 0.9
# ---------------------------------------------------------------
num_workers = 0
# 初始学习率大小
warmup = False
warmup_lr = 1e-6
use_cosine = False
# 训练的世代数
warmup_epoch = 1

# 标签平滑
label_smoothing = 0
CUDA = True
# 是否载入预训练模型参数
use_pretrain = False
#有多个gpu才能为True
Parallel = False
# gpu
gpu_device_id = 0
# ---------------------------------------------------------------
import json
from PIL import Image

x_train_path, y_train = [], []
labels = {}

def read_coco_annotation(file_path):
    with open(file_path, 'r') as f:
        coco_data = json.load(f)
    # 打印数据集的基本信息
    data = coco_data.get("annotations")

    for i in range(len(data)):
        img_id = str(data[i]['image_id'])
        img_name = '0' * (12 - len(img_id)) + img_id + '.jpg'
        box = data[i]['bbox']
        box_c = data[i]['category_id']
        box.append(box_c - 1)
        if img_name not in labels:
            labels[img_name] = []
        labels[img_name].append(box)
        x_train_path.append(img_name)

# # 示例文件路径
file_path = 'D:/PyCharm项目/CocoDataSet/annotations/instances_train2017.json'
read_coco_annotation(file_path)

max_pt = 0
for i, name in enumerate(x_train_path):
    y_train.append(labels[name])
    max_pt = max(max_pt, len(labels[name]))

device = torch.device("cuda:%d" % gpu_device_id if torch.cuda.is_available() else "cpu")

# test
# for it, (x, y) in enumerate(train_loader):
#     if it > 10:
#         break
#     print(x.shape, y)

# 3. 定义 vit 模型
# 使用预训练的权重
# weights = ViT_B_16_Weights.DEFAULT
# model = vit_b_16(weights=weights)
#
# model.heads[0] = torch.nn.Linear(model.heads[0].in_features, num_classes)

#使用的网络
model = Yolov11(nc=num_classes)

#参数初始化
#weights_init(model)

if use_pretrain:
    LR = 7.61e-6
    model_path = '../model' \
                 '/Self_focalLoss_Pp_val_loss1.033-size256-lr0.00000148-ep040-train_loss1.388.pth'
    model.load_state_dict(torch.load(model_path))

# 冻结主干进行训练
# for param in model.backbone.parameters():
#     param.requires_grad = False
if Parallel:
    model = torch.nn.DataParallel(model)

if CUDA:
    model = model.to(device)

#优化器
optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=weight_decay)

if not use_cosine:
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
else:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5, eta_min=LR/100)

#损失函数
loss_func = E2EDetectLoss(model, device=device, re_shape=RE_SIZE_shape)

val_loss_save = 1e10

time = time.asctime(time.localtime(time.time()))

dataset = CarObjectDataset(x_train_path, y_train, RE_SIZE_shape, num_classes, max_pt, train=True)
trainDataLen = int(len(dataset) * trainSize)

print(torch.cuda.is_available())
print(f"Model is on {device.type} device")

# 正式训练
val_loss_save = 1e10
print('start Training!')
trainLoss = []
valLoss = []

train_dataset, val_dataset = random_split(dataset, [trainDataLen, len(dataset) - trainDataLen])

# 数据集读取
train_loader = DataLoader(
    dataset=train_dataset,
    shuffle=True,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    pin_memory=False,
    drop_last=False
)
val_loader = DataLoader(
    dataset=val_dataset,
    shuffle=False,
    batch_size=BATCH_SIZE,
    num_workers=num_workers,
    pin_memory=False,
    drop_last=False
)

for epoch in range(EPOCH):
    val_loss_save, val_loss = fit_one_epoch(model, optimizer, loss_func, lr_scheduler, EPOCH, epoch, train_loader, val_loader, RE_SIZE_shape, val_loss_save, time, CUDA, device)
    valLoss.append(val_loss)
drawLoss([i for i in range(epoch+1)], valLoss, False)



