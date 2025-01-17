from net.backbone import Yolov11
from utils import fit
from torchvision import transforms

import torch, time
import numpy as np
import glob, tqdm
from PIL import Image, ImageDraw
from utils.loss_x import v8DetectionLoss, make_anchors, iou

CUDA = True
input_size = (676, 380)
resize_shape = 640
num_classes = 1
gpu_device_id = 0
Parallel = False

# font = ImageFont.truetype(r'/home/b201/gzx/yolox_self/font/STSONG.TTF', 12)

# anchors_path = '/home/b201/gzx/yolov3_self/utils/yolo_wheat_anchors.txt'
# # 先验框的大小
# # 输入为416，anchor大小为
# anchors = load_model.load_anchors(anchors_path)
# anchors = torch.tensor(anchors)/pic_shape

x_train_path = []
with open('../DataSets/CarObject/Data/sample_submission.csv') as f:
    lines = f.readlines()
    for line in lines[1:]:
        line = line.strip('\n')
        data = line.split(',')
        img_name = data[0]
        x_train_path.append(img_name)


device = torch.device("cuda:%d" % gpu_device_id if torch.cuda.is_available() else "cpu")
model = Yolov11(nc=1)
model_path = '../logs/' \
             'val_loss1745.381-size640-lr0.00000057-ep049-train_loss1843.478.pth'
model.load_state_dict(torch.load(model_path))

if Parallel:
    model = torch.nn.DataParallel(model)

if CUDA:
    model = model.to(device)
    # anchors = anchors.cuda()

def img_preprocess(img_path):
    img = Image.open('../DataSets/CarObject/data/testing_images/' + img_path).convert('RGB')
    img = transforms.Pad((0, 0, 0, input_size[0] - input_size[1]), fill=0)(img)
    # --------------------------------------------------------
    # normal data augmentation
    img = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)(img)  # 色域变换
    # --------------------------------------------------------

    img = img.resize((resize_shape, resize_shape), Image.BICUBIC)
    img = np.transpose(np.array(img) / 255., (2, 0, 1))
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)

    return img_tensor

def boxes_nms(boxes, score, pred_conf, cls_th=0.6, nms_th=0.3):
    '''
    :param boxes: (hw, 4)
    :param score: (hw, nc)
    :param pred_conf: (hw, 1)
    :return:
    '''
    pt_mask = ((score*pred_conf > cls_th).sum(-1) > 0) #(hw)
    unselected = pt_mask == True
    sorted  = (score*pred_conf).squeeze(-1).argsort() #(hw)
    while unselected.sum() > 0:
        cur_box_idx = sorted[pt_mask].argmax()
        box = boxes[pt_mask][cur_box_idx]
        iou_ = iou(boxes, box, ciou=False).squeeze(-1)
        ignore = (iou_ > nms_th) & (iou_ < 1)
        pt_mask[ignore] = False
        unselected[iou_ > nms_th] = False
        sorted[pt_mask & (unselected==False)] = -1

    return pt_mask

model.eval()
for i, img_name in enumerate(x_train_path):
    img = img_preprocess(img_name).type(torch.FloatTensor).unsqueeze(0)
    loss = v8DetectionLoss(model, re_shape=resize_shape, device=device)
    if CUDA:
        img = img.to(device)

    st = time.time()
    preds = model(img)
    ed = time.time()
    delay = ed - st

    pred_distri, pred_conf, pred_scores = torch.cat([xi.view(preds[0].shape[0], 5+num_classes, -1) for xi in preds], 2).split(
        (4, 1, num_classes), 1
    )

    pred_conf = pred_conf.permute(0, 2, 1).contiguous().sigmoid()  # (1, h*w, 1)
    pred_scores = pred_scores.permute(0, 2, 1).contiguous().sigmoid()  # (1, h*w, nc)
    pred_distri = pred_distri.permute(0, 2, 1).contiguous()

    anchor_points, stride_tensor = make_anchors(preds, model.stride, 0)

    # Pboxes
    pred_bboxes = loss.bbox_decode(anchor_points, pred_distri)*stride_tensor  # xyxy, (b, h*w, 4)
    pred_bboxes = (pred_bboxes.squeeze(0)/resize_shape*input_size[0])

    pred_scores = pred_scores.squeeze(0)
    pt_mask = boxes_nms(pred_bboxes, pred_scores, pred_conf.squeeze(0))
    boxes = pred_bboxes[pt_mask]

    '''
    最后得到的这个预测框可以用来进行绘制或是计算P、R等信息
    '''
    image = Image.open('../DataSets/CarObject/data/testing_images/' + img_name).convert('RGB')
    draw = ImageDraw.Draw(image)

    for i, b in enumerate(boxes):
        draw.rectangle([b[0], b[1], b[2], b[3]], outline='red', width=2)
        draw.text((b[0]+1, b[1]+1), 'Car {:.2f}'.format(pred_scores[pt_mask][i].item() * 100), fill='red')
        # draw.text((b[0] - b[2] / 2, b[1] - b[2] / 2), '{:.2f}'.format(b[4] * b[5] * 100), fill='red', stroke_width=1)

    image.save('../predictImages/{}.jpg'.format(img_name.split('.')[0]))
    print('推理时间：{:.2f}ms'.format(delay))

print('done!\n')