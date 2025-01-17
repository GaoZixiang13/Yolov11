import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def make_anchors(feats, strides, grid_cell_offset=0.0):
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
        # (tx, ty, tw, th)
        txy, twh = pred_dist.chunk(2, dim=-1)
        center_points = anchor_points + txy.sigmoid()
        wh = torch.exp(twh)
        x1y1, x2y2 = center_points - wh/2, center_points + wh/2
        return torch.cat((x1y1, x2y2), dim=-1)

    def __call__(self, preds, targets):
        '''
        :param preds: bs, 5+nc, h, w
        :param targets:
        :return:
        '''
        """Calculate the sum of the loss for box, cls and dfl multiplied by batch size."""
        pred_distri, pred_conf, pred_scores = torch.cat([xi.view(preds[0].shape[0], self.no, -1) for xi in preds], 2).split(
            (4, 1, self.nc), 1
        )

        pred_conf   = pred_conf.permute(0, 2, 1).contiguous().sigmoid() # (bs, h*w, 1)
        pred_scores = pred_scores.permute(0, 2, 1).contiguous().sigmoid() # (bs, h*w, nc)
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        anchor_points, stride_tensor = make_anchors(preds, self.stride, 0)
        pred_bboxes = self.bbox_decode(anchor_points, pred_distri) #单位：格子

        # Targets
        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # cls, xyxy
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0).bool()

        loss_box, loss_cls, loss_conf = self.assigner(pred_bboxes*stride_tensor, gt_bboxes, mask_gt, pred_scores, pred_conf, gt_labels.long(), anchor_points, stride_tensor)
        loss = loss_box*self.hyp.box + loss_cls*self.hyp.cls + loss_conf*self.hyp.conf

        return loss

def iou(preds, boxes, ciou=False, eps=1e-9): #(bs, num, 4)
    b1_x1, b1_y1, b1_x2, b1_y2 = preds.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = ((b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) *
             (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0))

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union

    if ciou == True:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # convex height
        c2 = cw.pow(2) + ch.pow(2) + eps  # convex diagonal squared
        d2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).pow(2) + (b2_y1 + b2_y2 - b1_y1 - b1_y2).pow(2)) / 4  # center dist**2
        v = (4 / (math.pi**2)) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
        with torch.no_grad():
            alpha = v / (v - iou + (1 + eps))
        return iou - (d2 / c2 + v * alpha)  # CIoU
    return iou

class simOTA(nn.Module):
    def __init__(self, reshape=640, nc=80, topk=10, eps=1e-7):
        super().__init__()
        self.topk = topk
        self.nc = nc
        self.eps = eps
        self.reshape = reshape
        self.dis = 2.5

    def Focalloss(self, pred, target):
        alpha, gamma = .25, 2

        noobj_mask = (target == 0)
        pt = torch.clone(pred)
        pt[noobj_mask] = 1 - pred[noobj_mask]

        return -alpha * (1 - pt).pow(gamma) * torch.log(pt + self.eps)

    def BCEloss(self, pred, target):
        output = - target * torch.log(pred + self.eps) - (1.0 - target) * torch.log(1.0 - pred + self.eps)
        return output

    def forward(self, pred_boxes, gt_boxes, mask_gt, pred_scores, pred_conf, gt_labels, anchor_points, stride_tensor):
        '''
        :param pred_boxes: (bs, h*w, 4)
        :param gt_boxes: (bs, num_max_gt, 4)
        :param mask_gt: (bs, num_max_gt)
        :param pred_scores: (bs, h*w, nc)
        :param pred_conf: (bs, h*w, 1)
        :param gt_labels: (bs, num_max_gt, 1)
        :param anchor_points: (h*w, 2)
        :param stride_tensor: (h*w, 1)
        :return: pt_mask: (bs, num_gt, h*w)
        '''
        bs, hw, nc = pred_scores.shape[0], pred_scores.shape[1], pred_scores.shape[2]
        dtype = pred_boxes.dtype
        device = pred_boxes.device
        num_max_gt = mask_gt.shape[1]

        loss = torch.zeros(3, device=device).type(dtype)  # conf, box, cls

        for b in range(bs):
            pt_mask = torch.zeros(num_max_gt, hw).bool().to(device) #(num_max_gt, h*w)
            loss_t = torch.zeros(3, device=device).type(dtype)  # conf, box, cls
            cost_function = []

            for i in range(num_max_gt):
                if mask_gt[b][i] == False:
                    cost = 1000000 * torch.ones(hw).type(dtype).to(device)  # (h*w)
                    cost_function.append(cost)
                    continue

                gt_scores = F.one_hot(gt_labels[b][i], num_classes=self.nc).squeeze(0).type(dtype).to(device)  # (nc)

                anchor_points_pd = (anchor_points + 0.5) * stride_tensor #(hw, 2)
                gt_box = gt_boxes[b][i] #(4)
                lt = anchor_points_pd - gt_box[:2]
                rb = gt_box[2:] - anchor_points_pd
                in_boxes = ((lt > 0).sum(dim=-1) == 2) & ((rb > 0).sum(dim=-1) == 2)  # (hw)

                gt_boxes_center = (gt_box[:2] + gt_box[2:]) / 2
                dis = (self.dis * stride_tensor).expand(hw, 2)
                gt_boxes_lt, gt_boxes_rb = gt_boxes_center - dis, gt_boxes_center + dis
                lt_two = anchor_points_pd - gt_boxes_lt
                rb_two = gt_boxes_rb - anchor_points_pd  # (num_max_gt, h*w, 2)
                in_centers = ((lt_two > 0).sum(-1) == 2) & ((rb_two > 0).sum(-1) == 2)  # (h*w)

                fg_mask, is_in_boxes_and_centers = in_boxes | in_centers, in_boxes & in_centers  # (h*w)

                num_fg = fg_mask.sum().long()

                gt_score_pd = gt_scores.unsqueeze(0).repeat(hw, 1)
                gt_score_pd[~fg_mask] = 0.

                iou_val = iou(pred_boxes[b], gt_box, ciou=False).squeeze(-1) #(h*w, 1)
                iou_cost = -torch.log(iou_val + self.eps)
                cls_loss  = F.binary_cross_entropy(pred_scores[b], gt_score_pd, reduction="none").sum(-1) #(h*w)
                cost = (cls_loss + 3.0 * iou_cost + 100000 * (~is_in_boxes_and_centers)) #(h*w)

                if num_fg == 0:
                    print("error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                k = min(self.topk, num_fg)
                pt_num, _ = iou_val[fg_mask].topk(k=k, dim=-1) #(10)
                pt_num = pt_num.sum().clamp(min=1).type(torch.LongTensor) #(0)

                _, pt_mask_i_idx = cost.topk(k=pt_num, dim=-1, largest=False) #(pt_num[b])
                mask_t = F.one_hot(pt_mask_i_idx, num_classes=hw).sum(0).bool().squeeze(0).to(device)

                if len(mask_t.shape) != 1:
                    print(mask_t.shape)
                pt_mask[i][mask_t] = True
                cost_function.append(cost)

            cost_function = torch.stack(cost_function, dim=0)
            # cost_function (num_max_gt, h*w)
            # 去除重叠样本
            overlap_mask = pt_mask.sum(dim=0) > 1 #(h*w)
            cost_min_gt_indices = torch.argmin(cost_function[:, overlap_mask], dim=0) #(num_overlap)
            pt_mask[:, overlap_mask] = False
            pt_mask[cost_min_gt_indices, overlap_mask] = True

            for i in range(num_max_gt):
                if mask_gt[b][i] == False:
                    continue

                gt_box = gt_boxes[b][i]
                num_pt = pt_mask[i].sum().long().item()
                if num_pt == 0:
                    continue

                gt_scores = F.one_hot(gt_labels[b][i], num_classes=self.nc).squeeze(0).type(dtype).to(device)  # (nc)
                gt_score_pd = gt_scores.unsqueeze(0).expand(num_pt, self.nc)

                loss_t[0] += (1 - iou(pred_boxes[b][pt_mask[i]], gt_box, ciou=True)).sum()
                loss_t[1] += self.BCEloss(pred_scores[b][pt_mask[i]], gt_score_pd).sum()

            num_pts = pt_mask.sum().long() #这张图片被选为正样本的预测点数量

            gt_conf = pt_mask.sum(dim=0) > 0 #(h*w)
            loss[2] += self.BCEloss(pred_conf[b].squeeze(-1), gt_conf.type(dtype)).sum()/num_pts #conf

            loss[0] += loss_t[0]/num_pts #box
            loss[1] += loss_t[1]/num_pts #cls

        return loss[0], loss[1], loss[2]



