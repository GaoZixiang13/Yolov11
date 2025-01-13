from net.backbone import Yolov11
from utils import fit
from torchvision import transforms

import torch, time
import numpy as np
import glob, tqdm
from PIL import Image, ImageDraw
from utils.loss_t import v8DetectionLoss, make_anchors

CUDA = True
input_size = (676, 380)
resize_shape = 640
num_classes = 1
gpu_device_id = 0
Parallel = False
single_img = True
reg_max = 16

# font = ImageFont.truetype(r'/home/b201/gzx/yolox_self/font/STSONG.TTF', 12)

# anchors_path = '/home/b201/gzx/yolov3_self/utils/yolo_wheat_anchors.txt'
# # 先验框的大小
# # 输入为416，anchor大小为
# anchors = load_model.load_anchors(anchors_path)
# anchors = torch.tensor(anchors)/pic_shape

device = torch.device("cuda:%d" % gpu_device_id if torch.cuda.is_available() else "cpu")
model = Yolov11(nc=1)
model_path = '../logs/' \
             'val_loss1006.815-size640-lr0.00000064-ep048-train_loss988.581.pth'
model.load_state_dict(torch.load(model_path))

if Parallel:
    model = torch.nn.DataParallel(model)

if CUDA:
    model = model.to(device)
    # anchors = anchors.cuda()

def img_preprocess(img_path):
    img = Image.open('../DataSets/CarObject/data/training_images/' + img_path).convert('RGB')
    img = transforms.Pad((0, 0, 0, input_size[0] - input_size[1]), fill=0)(img)
    # --------------------------------------------------------
    # normal data augmentation
    img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img)  # 色域变换
    # --------------------------------------------------------

    img = img.resize((resize_shape, resize_shape), Image.BICUBIC)
    img = np.transpose(np.array(img) / 255., (2, 0, 1))
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    return img_tensor

def boxes_nms(boxes):
    score = boxes[:, -1]
    pt_mask = torch.ones_like(score).bool()


model.eval()
if single_img:
    img_name = 'vid_4_700.jpg'
    img = img_preprocess(img_name).type(torch.FloatTensor).unsqueeze(0)
    loss = v8DetectionLoss(model, device=device)
    if CUDA:
        img = img.to(device)

    st = time.time()
    feats = model(img)
    ed = time.time()
    delay = ed - st

    pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], model.no, -1) for xi in feats], 2).split(
        (model.reg_max * 4, model.nc), 1
    )

    pred_scores = pred_scores.permute(0, 2, 1).contiguous()
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    anchor_points, stride_tensor = make_anchors(feats, model.stride, 0.5)

    print(anchor_points, stride_tensor)
    # Pboxes
    pred_bboxes = loss.bbox_decode(anchor_points, pred_distri)*stride_tensor  # xyxy, (b, h*w, 4)
    pred_bboxes = (pred_bboxes.squeeze(0)/resize_shape*input_size[0]).clamp_(0.0, input_size[0])
    print(pred_bboxes.shape)

    pred_scores = pred_scores.view(-1).detach().sigmoid()
    idx = pred_scores.argmax(-1)
    b = pred_bboxes[idx]/resize_shape*input_size[0]

    print(pred_scores[idx], b)
    '''
    最后得到的这个预测框可以用来进行绘制或是计算P、R等信息
    '''
    image = Image.open('../DataSets/CarObject/data/training_images/' + img_name).convert('RGB')
    image = transforms.Pad((0, 0, 0, input_size[0] - input_size[1]), fill=0)(image)
    draw = ImageDraw.Draw(image)

    draw.rectangle([b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2], outline='red', width=2)
    draw.text((b[0] - b[2] / 2, b[1] - b[3] / 2), 'Car {:.2f}'.format(pred_scores[idx] * 100), fill='red')
    # draw.text((b[0] - b[2] / 2, b[1] - b[2] / 2), '{:.2f}'.format(b[4] * b[5] * 100), fill='red', stroke_width=1)

    image.save('../predictImages/{}.jpg'.format(img_name.split('.')[0]))
    print('推理延迟为%.2fms' % delay)
