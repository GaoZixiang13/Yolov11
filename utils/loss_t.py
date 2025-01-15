import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from docutils.nodes import target
from sqlalchemy.util import clsname_as_plain_name

from train.val import pt_mask
from utils.metrics import bbox_iou, box_iou

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

class v8DetectionLoss:
    """Criterion class for computing training losses."""

    def __init__(self, m, device, re_shape, tal_topk=10):  # model must be de-paralleled
        """Initializes v8DetectionLoss with the model, defining model-related properties and BCE loss function."""
        self.bce = nn.CrossEntropyLoss()
        self.hyp = m.hyp
        self.stride = m.stride # model strides
        self.nc = m.nc  # number of classes
        self.no = m.no
        self.device = device
        self.reshape = re_shape

        self.assigner = simOTA(nc=self.nc, topk=tal_topk)

    def bbox_decode(self, anchor_points, pred_dist):
        lt, rb = pred_dist.chunk(2, dim=1)
        x1y1, x2y2 = anchor_points-lt, anchor_points+rb
        return torch.cat((x1y1, x2y2), dim=-1)

    def __call__(self, preds, targets):
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        loss = torch.zeros(3, device=self.device)  # conf, box, cls
        pred_conf, pred_distri, pred_scores = torch.cat([xi.view(preds[0].shape[0], self.no, -1) for xi in preds], 2).split(
            (1, 4, self.nc), 1
        )

        pred_conf   = pred_conf.permute(0, 2, 1).contiguous().sigmoid()
        pred_scores = pred_scores.permute(0, 2, 1).contiguous().sigmoid()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(preds, self.stride, 0.5)

        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        # Targets
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        # Pboxes
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        pt_mask = self.assigner(pred_bboxes, gt_bboxes, mask_gt, pred_scores*pred_conf, gt_labels)

        pred_bboxes = (pred_bboxes*stride_tensor).clamp(min=0, max=self.reshape)# xyxy, (b, h*w, 4)
        gt_scores = F.one_hot(gt_labels, num_classes=self.nc) #(bs, num_max_gt, nc)

        loss[0] = ciou(pred_bboxes[pt_mask], gt_bboxes[mask_gt])
        loss[1] = F.binary_cross_entropy(pred_scores[pt_mask], gt_scores[mask_gt])
        loss[2] = F.binary_cross_entropy(pred_conf, gt_scores[mask_gt])

        loss[0] *= self.hyp.box  # box gain
        loss[1] *= self.hyp.cls  # cls gain
        loss[2] *= self.hyp.dfl  # conf gain

        return loss.sum() * batch_size, loss.detach()  # loss(box, cls, conf)

def ciou(preds, boxes, eps=1e-9): #(bs, num_fg, 4), (bs, 1, 4)
    b1_x1, b1_y1, b1_x2, b1_y2 = preds.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = ((b1_x2.minimum(b2_x2, dim=-1) - b1_x1.maximum(b2_x1, dim=-1)).clamp_(0) *
             (b1_y2.minimum(b2_y2, dim=-1) - b1_y1.maximum(b2_y1, dim=-1)).clamp_(0))

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    cw = b1_x2.maximum(b2_x2, dim=-1) - b1_x1.minimum(b2_x1, dim=-1)  # convex (smallest enclosing box) width
    ch = b1_y2.maximum(b2_y2, dim=-1) - b1_y1.minimum(b2_y1, dim=-1)  # convex height
    c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
    d2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4  # center dist**2
    v = (4 / math.pi**2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (d2 / c2 + v * alpha)  # CIoU

class simOTA(nn.Module):
    def __init__(self, nc=80, topk=10, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.nc = nc
        self.eps = eps
        self.dis = 2.5

    def forward(self, pred_boxes, gt_boxes, mask_gt, pred_scores, gt_labels):
        '''
        :param pred_boxes: (bs, h*w, 4)
        :param gt_boxes: (bs, num_max_gt, 4)
        :param mask_gt: (bs, num_max_gt)
        :param pred_scores: (bs, h*w, nc)
        :param gt_labels: (bs, num_max_gt, 1)
        :return:
        '''
        bs = pred_boxes.shape[0]
        dtype = pred_boxes.dtype
        device = pred_boxes.device
        gt_num = mask_gt.sum()
        pred_boxes_center = (pred_boxes[..., :2] + pred_boxes[..., 2:])/2
        pred_boxes_center_epd = pred_boxes_center.expand(gt_num, dim=-1) #(bs, h*w, 4, gt_num)
        lt = pred_boxes_center_epd[..., :2] - gt_boxes[mask_gt][..., :2]
        rb = gt_boxes[mask_gt][..., 2:]     - pred_boxes_center_epd[..., 2:]
        in_boxes = ((lt > 0).sum(-1) == 2) & ((rb > 0).sum(-1) == 2) #(bs, h*w, 4)

        gt_boxes_center = (gt_boxes[..., :2] + gt_boxes[..., 2:])/2
        gt_boxes_lt, gt_boxes_rb = gt_boxes_center[...,:2]-self.dis, gt_boxes_center[...,2:]+self.dis
        gt_boxes_two = torch.cat((gt_boxes_lt, gt_boxes_rb), dim=-1)
        lt_two = pred_boxes_center_epd[..., :2] - gt_boxes_two[mask_gt][..., :2]
        rb_two = gt_boxes_two[mask_gt][..., 2:]     - pred_boxes_center_epd[..., 2:]
        in_centers = ((lt_two > 0).sum(-1) == 2) & ((rb_two > 0).sum(-1) == 2) #(bs, h*w, 4)

        fg_mask, is_in_boxes_and_centers = in_boxes | in_centers, in_boxes & in_centers

        gt_scores = F.one_hot(gt_labels, num_classes=self.nc) #(bs, num_max_gt, nc)

        pt_mask = torch.tensor([]).type(dtype).to(device) #(bs, gt_num, h*w)
        cost_mat = torch.tensor([]).type(dtype).to(device) #(bs, gt_num, h*w)
        for i in range(gt_num):
            ciou_loss = ciou(pred_boxes[fg_mask], gt_boxes[mask_gt][i]) #(bs, num_fg)
            cls_loss  = F.binary_cross_entropy(pred_scores[fg_mask], gt_scores[mask_gt][i].expand(fg_mask.sum(), dim=1)) #(bs, num_fg, nc)
            cost = (cls_loss + 3.0 * ciou_loss + 100000 * (~is_in_boxes_and_centers)) #(bs, num_fg)
            pt_num, _ = cost.topk(10, dim=-1).type(torch.LongTensor).clamp(min=1)
            _, pt_mask_i_idx = cost.topk(k=pt_num, dim=-1, largest=False)
            pt_mask_i = F.one_hot(pt_mask_i_idx, fg_mask.shape[1]).sum(-2).type(torch.BoolTensor)

            cost_mat = torch.stack((cost_mat, cost), dim=0)
            pt_mask = torch.stack((pt_mask, pt_mask_i), dim=0)

        overlap = pt_mask.sum(-2) > 1
        for b in range(bs):
            pt_mask[b, :, overlap[b]] = False
            deal_mat = cost_mat[b, :, overlap[b]] #(gt_num, overlap_num)
            pick_idx = deal_mat.argmin(dim=0)
            pt_mask[b, pick_idx, overlap[b]] = True

        return pt_mask




