import torch, time
from torch.utils.data import Dataset, DataLoader, random_split

from preprocess.preprocess import CarObjectDataset
from utils.fit import fit_one_epoch
from net.backbone import Yolov11
from utils.loss_t import v8DetectionLoss
from DrawPic.DrawImage import drawLoss

print(torch.version.cuda)
print(torch.__file__)

# ---------------------------------------------------------------
# Hyper Parameters
BATCH_SIZE = 4
num_workers = 0
# 初始学习率大小
warmup = False
warmup_lr = 1e-6
LR = 1e-4
use_cosine = False
# 训练的世代数
warmup_epoch = 1
EPOCH = 60
# 网络输入图片size
input_size = (676, 380)
RE_SIZE_shape = 640

# 总的类别数
num_classes = 1
# 标签平滑
label_smoothing = 0
CUDA = True
# 是否载入预训练模型参数
use_pretrain = False
#有多个gpu才能为True
Parallel = False
# gpu
gpu_device_id = 0

trainSize = 0.9
# ---------------------------------------------------------------

x_train_path, y_train = [], []
labels = {}
with open('../DataSets/CarObject/Data/train_solution_bounding_boxes.csv') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip('\n')
        data = line.split(',')
        img_name = data[0]
        box      = [float(data[1]), float(data[2]), float(data[3]), float(data[4])]
        if img_name not in labels:
            labels[img_name] = []
        labels[data[0]].append(box)
        x_train_path.append(img_name)

M = 0
for i, name in enumerate(x_train_path):
    y_train.append(labels[name])
    M = max(len(labels[name]), M)

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
print(model)

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
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=5e-4)

if not use_cosine:
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)
else:
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=5, eta_min=LR/100)

#损失函数
loss_func = v8DetectionLoss(model, device=device, re_shape=RE_SIZE_shape)

val_loss_save = 1e10

time = time.asctime(time.localtime(time.time()))

dataset = CarObjectDataset(x_train_path, y_train, input_size, RE_SIZE_shape, num_classes, M, train=True)
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
