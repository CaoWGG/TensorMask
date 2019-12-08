import torch
import torch.nn as nn
import torch.nn.functional as F
from .ops.sigmoid_focal_loss.modules.sigmoid_focal_loss import SigmoidFocalLoss
from lib.lovasz_losses import lovasz_hinge
import math
def _sigmoid(x):
  y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
  return y

def diou(bboxes1, bboxes2):
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[..., 2] + bboxes1[..., 0]) / 2
    center_y1 = (bboxes1[..., 3] + bboxes1[..., 1]) / 2
    center_x2 = (bboxes2[..., 2] + bboxes2[..., 0]) / 2
    center_y2 = (bboxes2[..., 3] + bboxes2[..., 1]) / 2

    inter_max_xy = torch.min(bboxes1[..., 2:],bboxes2[..., 2:])
    inter_min_xy = torch.max(bboxes1[..., :2],bboxes2[..., :2])
    out_max_xy = torch.max(bboxes1[..., 2:],bboxes2[..., 2:])
    out_min_xy = torch.min(bboxes1[..., :2],bboxes2[..., :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[..., 0] * inter[..., 1]
    inter_diag = (center_x2 - center_x1)**2 + (center_y2 - center_y1)**2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[..., 0] ** 2) + (outer[..., 1] ** 2)
    union = area1+area2-inter_area
    u = (inter_diag) / (outer_diag + 1e-7 )
    iou = inter_area / (union + 1e-7)
    dious = iou - u
    return dious


def iou(bboxes1, bboxes2):
    w1 = bboxes1[..., 2] - bboxes1[..., 0]
    h1 = bboxes1[..., 3] - bboxes1[..., 1]
    w2 = bboxes2[..., 2] - bboxes2[..., 0]
    h2 = bboxes2[..., 3] - bboxes2[..., 1]
    area1 = w1 * h1
    area2 = w2 * h2
    inter_max_xy = torch.min(bboxes1[..., 2:],bboxes2[..., 2:])
    inter_min_xy = torch.max(bboxes1[..., :2],bboxes2[..., :2])
    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[..., 0] * inter[..., 1]
    union = area1+area2-inter_area
    iou = inter_area / (union + 1e-7)
    return iou

def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat

def _tranpose_and_gather_feat(feat, ind):
    feat = _gather_feat(feat, ind)
    return feat

class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def forward(self, output, mask, ind, target):
        pred = _tranpose_and_gather_feat(output, ind)
        mask = mask.unsqueeze(2).expand_as(pred).float()
        loss = (F.binary_cross_entropy_with_logits(pred, target, reduction='none') * mask).sum()
        loss = loss / (mask.sum() + 1e-4)
        return loss

class CIOULoss(nn.Module):
    def __init__(self):
        super(CIOULoss, self).__init__()

    def forward(self, output, mask, ind, target):
        mask = mask.float()
        pred = _tranpose_and_gather_feat(output, ind)
        right_offset,left_offset = torch.split(pred,[2,2],dim=-1)
        x1y1x2y2,ct,stride = torch.split(target,[4,2,1],dim=-1)
        stride = stride.expand_as(right_offset).float()
        predx1y1 = (ct + 0.5 - right_offset)*stride
        predx2y2 = (ct + 0.5 + left_offset )*stride
        predx1y1x2y2 = torch.cat([predx1y1,predx2y2],dim = -1)
        diou_loss = (1. - diou(predx1y1x2y2,x1y1x2y2)) * mask
        loss = diou_loss.sum() / ( mask.sum() + 1e-4)
        return loss

class MaskLoss(nn.Module):
    def __init__(self):
        super(MaskLoss, self).__init__()


    def forward(self, output, mask, ind, target):
        # B,N,window=target.size(0),target.size(1),target.size(-1)
        # loss = 0
        # num_smaple = mask.sum()
        # if num_smaple > 0:
        #     output = output.view(B,-1,window*window)
        #     pred = _tranpose_and_gather_feat(output, ind).view(B,N,window,window)
        #     mask = mask.unsqueeze(2).unsqueeze(2).expand_as(pred)
        #     pred = pred[mask]
        #     target = target[mask]
        #     loss = lovasz_hinge(pred,target)
        B,N,window=target.size(0),target.size(1),target.size(-1)
        output = output.view(B,-1,window*window)
        pred = _tranpose_and_gather_feat(output, ind).view(B,N,window,window)
        mask = mask.unsqueeze(2).unsqueeze(2).expand_as(pred).float()
        pos_ind = target.eq(1).float()
        neg_ind = (1-pos_ind)
        pos_loss = (torch.log(pred)*mask*pos_ind).sum()
        neg_loss = (torch.log(1-pred)*mask*neg_ind).sum()
        num_smaple = mask.sum()
        loss = 0 - (1.5*pos_loss + neg_loss)   # 1.5 from paper
        if num_smaple > 0:
            loss /= num_smaple
        return loss

class TensorMaskLoss(nn.Module):
    def __init__(self,opt):
        super(TensorMaskLoss,self).__init__()
        self.cls_loss = SigmoidFocalLoss(gamma=3,alpha=0.3)
        self.box_loss = CIOULoss()
        self.mask_loss = MaskLoss()
        self.opt = opt

    def forward(self, ouput,batch):
        opt = self.opt
        mask_loss = 0
        num_sample = batch['reg_mask'].sum()
        cls_loss = self.cls_loss(ouput['cls'].view([-1,opt.num_class]),batch['cls'].view([-1]))
        box_loss = self.box_loss(ouput['box'],batch['reg_mask'],batch['ind'],batch['xywh'])
        for i in range(6):
            mask_loss += self.mask_loss(_sigmoid(ouput['%d'%i]),batch['seg_mask_%d'%i],batch['seg_ind_%d'%i],batch['seg_%d'%i])
        mask_loss /= 6
        if num_sample > 0:
            cls_loss /= num_sample
        loss = opt.cls_weights * cls_loss + opt.xywh_weights * box_loss + opt.mask_weights * mask_loss
        loss_stats = {'loss': loss, 'cls_loss': cls_loss,
                      'diou_loss': box_loss, 'mask_loss': mask_loss}
        return loss,loss_stats