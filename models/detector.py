from models.tensormask import TensorMask
from config import cfg as opt
from lib.utils import load_model,save_model
from lib.coco import COCO
import numpy as np
import torch
import os
import cv2

def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

def cal_iou_np(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

class Detector():
    def __init__(self,opt):
        self.model = TensorMask(backbone=opt.backbone, num_cls=opt.num_class,
                           base_window=opt.base_window,
                           freezeBN=opt.frezeBN, freezeLayers=opt.frezeLayer,
                           align_corners=opt.align_corners)
        self.model = load_model(self.model, opt.weights)
        self.model.eval()
        self.model.cuda()
        self.mean = COCO.mean
        self.std = COCO.std
        self.opt = opt

        self.strides = np.array([self.opt.base_stride * 2 ** i for i in range(self.opt.k + 1)])
        self.windows = np.array([self.opt.base_window * lamda for lamda in self.strides], np.int32)

        self.output_size = np.array(list(zip(self.opt.input_w // self.strides, self.opt.input_h // self.strides)))
        self.num_det = [output_w * output_h for output_w, output_h in self.output_size]
        self.det_offset = np.cumsum(self.num_det)

    def run(self,image,vis=True):
        if isinstance(image,str):
            image = cv2.imread(image)
        show = image.copy()
        image,trans_output = self.prepare_image(image)
        input = torch.from_numpy(image).cuda()
        output = self.model(input)
        box,mask = self.decode(output,show.shape[:2],trans_output)
        if vis:
            self.show_img(show,box,mask)
        return box,mask

    def prepare_image(self,image):
        height, width = image.shape[0], image.shape[1]
        ar = width/height
        new_h,new_w = (self.opt.input_h,ar*self.opt.input_h) if ar < 1 else (self.opt.input_w/ar,self.opt.input_w)
        dx, dy = (self.opt.input_w - new_w) / 2, (self.opt.input_h - new_h) / 2
        src = np.array([[0, 0], [0, height], [width, 0]], dtype=np.float32)
        dst = np.array([[dx, dy], [dx, new_h + dy], [new_w + dx, dy]], dtype=np.float32)
        trans_input = cv2.getAffineTransform(src, dst)
        trans_output = cv2.getAffineTransform(dst, src)
        image = cv2.warpAffine(image, trans_input, (self.opt.input_w, self.opt.input_h),
                               flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        image = (image.astype(np.float32) / 255.)
        image = (image- self.mean) / self.std
        image = image.transpose(2, 0, 1)
        image = np.expand_dims(image,0).astype(np.float32)
        return image,trans_output

    def decode(self,output,img_hw,trans_ouput,method = 'nms',iou_threshold=0.45,sigma=0.3):
        socres,cls = torch.max(output['cls'].sigmoid_(),dim=-1)
        socres = socres.detach().cpu().numpy()
        cls = cls.detach().cpu().numpy()
        box = output['box'].detach().cpu().numpy()
        seg = [output['%d' % i].sigmoid_().detach().cpu().numpy() for i in range(self.opt.k + 1)]
        topk_inds = np.where(socres > self.opt.vis_thresh)
        result = []
        for det_num in topk_inds[1]:
            p = socres[0, det_num]
            cls_index = cls[0,det_num]
            b = box[0, det_num, :]
            for id, num in enumerate(self.det_offset):
                if num > det_num:
                    break
            offset = det_num - self.det_offset[id - 1] if id > 0 else det_num
            width, hight = self.output_size[id]

            ### ct_int_feat
            y = int(offset / width)
            x = int(offset % width)

            b[0:2] = (x + 0.5 - b[0] )* self.strides[id],( y + 0.5 - b[1] )* self.strides[id]
            b[2:4] = (x + 0.5 + b[2] )* self.strides[id],( y + 0.5 + b[3] )* self.strides[id]
            b[0:2] = affine_transform(b[0:2],trans_ouput).astype(int)
            b[2:4] = affine_transform(b[2:4], trans_ouput).astype(int)
            result.append([*b,p,cls_index,x,y,id])

        result = np.array(result) ## x1 y1 x2 y2 p cls ct_feat_x ct_feat_y feat_id

        ### use box to nms
        class_index = result[:,5] if len(result) > 0 else []
        classes_in_img = list(set(class_index))
        best_bboxes = []
        for cls in classes_in_img:
            cls_mask = (class_index == cls)
            cls_bboxes = result[cls_mask]
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 4])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = cal_iou_np(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                assert method in ['nms', 'soft-nms']
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
                score_mask = cls_bboxes[:, 4] >  self.opt.vis_thresh
                cls_bboxes = cls_bboxes[score_mask]
        mask_res= []
        for det in best_bboxes:
            mask = np.zeros([self.opt.input_h,self.opt.input_w],np.uint8)
            ct_feat_x,ct_feat_y,feat_id = int(det[-3]),int(det[-2]),int(det[-1])
            x, y = int((ct_feat_x + 0.5) * self.strides[feat_id]), int((ct_feat_y + 0.5) * self.strides[feat_id])
            window_seg = seg[feat_id][0, ct_feat_y, ct_feat_x, :, :]
            paste_x, paste_y, paste_x1, paste_y1 = x - self.windows[feat_id] // 2,\
                                                   y - self.windows[feat_id] // 2, \
                                                   x + self.windows[feat_id] // 2,\
                                                   y + self.windows[feat_id] // 2

            window_x, window_y, window_x1, window_y1 = max(-paste_x, 0), max(-paste_y, 0), \
                                                       self.windows[feat_id] - max(0, paste_x1 - self.opt.input_w), \
                                                       self.windows[feat_id] - max(0, paste_y1 - self.opt.input_h)
            paste_x, paste_y, paste_x1, paste_y1 = max(paste_x, 0), max(paste_y, 0), \
                                                   min(paste_x1, self.opt.input_w), \
                                                   min(paste_y1,self.opt.input_h)
            window_seg = cv2.resize(window_seg, (self.windows[feat_id],self. windows[feat_id]))
            window_seg = (window_seg > 0.5).astype(np.uint8)
            mask[paste_y:paste_y1, paste_x:paste_x1] = window_seg[window_y:window_y1, window_x:window_x1]
            mask = cv2.warpAffine(mask, trans_ouput,
                                     (img_hw[1], img_hw[0]),
                                     flags=cv2.INTER_LINEAR)

            mask_res.append(mask)
        return best_bboxes,mask_res

    def show_img(self,img,box,mask):
        for i in range(len(box)):
            det = box[i].astype(np.int)
            if self.opt.show_box:
                cv2.rectangle(img, (det[0], det[1]), (det[2], det[3]), (255, 0, 0), 2)
            color = np.array([[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]])
            seg = mask[i]==1
            img[seg] = img[seg] * 0.2 + color * 0.8

        cv2.imshow('result',img)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    opt.weights = '/data/yoloCao/pycharmProjects/tensormask/exp/coco_person/model_last.pth'
    detector = Detector(opt)
    img = '/data/yoloCao/DataSet/VOC2007/JPEGImages/2007_000027.jpg'
    opt.vis_thresh = 0.5
    detector.run(img)

