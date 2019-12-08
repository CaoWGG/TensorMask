import pycocotools.coco as coco
import numpy as np
import os
import cv2
from torch.utils.data import Dataset


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.], dtype=np.float32).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]

class COCO(Dataset):
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)
    def __init__(self, cfg, split = 'train',augment = True):
        super(COCO, self).__init__()
        self.data_dir = cfg.data_dir
        self.img_dir = os.path.join(self.data_dir, '{}2017'.format(split))
        self.annot_path = os.path.join(
            self.data_dir, 'annotations',
            'instances_{}2017.json').format(split)
        self.split = split
        print('==> initializing coco 2017 {} data.'.format(split))
        self.coco = coco.COCO(self.annot_path)
        self.images = self.coco.getImgIds()
        self.num_samples = len(self.images)

        self.class_name = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
            'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
            'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
            'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
            'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
            'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        self._valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]
        if cfg.class_name != '*' :
            self._valid_ids = [self.class_name.index(cfg.class_name)]
            self.class_name = [cfg.class_name]
            catIds = self.coco.getCatIds(self.class_name[-1])
            assert catIds == self._valid_ids
            self.images = self.coco.getImgIds(self.images, catIds)
            self.num_samples = len(self.images)
        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
        self.input_w = cfg.input_w
        self.input_h = cfg.input_h
        self.base_stride = cfg.base_stride
        self.base_window = cfg.base_window
        self.k = cfg.k
        self.num_class = len(self.class_name)

        self.augment=augment
        self.max_objs = 45
        self.jitter = cfg.jitter
        self.cfg = cfg
        if not self.augment:
            self.jitter = 0
        print('Loaded {} {} samples'.format(split, self.num_samples))

    def __len__(self):
        return self.num_samples


    def _coco_box_to_bbox(self, box):
        bbox = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]],dtype=np.float32)
        return bbox

    def get_image_name(self,img_id):
        return os.path.join(self.img_dir,self.coco.loadImgs(ids=[self.images[img_id]])[0]['file_name']).strip()

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def __getitem__(self, index):

        img_id = self.images[index]
        file_name = self.coco.loadImgs(ids=[img_id])[0]['file_name']
        img_path = os.path.join(self.img_dir, file_name)
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        anns = self.coco.loadAnns(ids=ann_ids)
        anns = list(filter(lambda x: x['category_id'] in self._valid_ids and x['iscrowd'] != 1, anns))
        image = cv2.imread(img_path)

        ## augment
        height, width = image.shape[0], image.shape[1]
        dw, dh = self.jitter * width, self.jitter * height
        new_ar = (width + np.random.uniform(-dw, dw)) / (height + np.random.uniform(-dh, dh))
        sclae = 1
        if new_ar < 1:
            new_h = sclae * self.input_h
            new_w = new_ar * new_h
        else:
            new_w = sclae * self.input_w
            new_h = new_w / new_ar

        dx, dy = (np.random.uniform(0, self.input_w - new_w), np.random.uniform(0, self.input_h - new_h)) \
            if self.augment else ((self.input_w - new_w) / 2, (self.input_h - new_h) / 2)

        flipped = False
        if np.random.random() < 0.5  and self.augment:
            image = np.copy(image[:, ::-1, :])
            flipped = True

        src = np.array([[0, 0], [0, height], [width, 0]], dtype=np.float32)
        dst = np.array([[dx, dy], [dx, new_h + dy], [new_w + dx, dy]], dtype=np.float32)
        trans_input = cv2.getAffineTransform(src, dst)
        image = cv2.warpAffine(image, trans_input, (self.input_w, self.input_h),
                               flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        show = image.copy()


        image = (image.astype(np.float32) / 255.)
        image = (image- self.mean) / self.std
        image = image.transpose(2, 0, 1)

        strides = np.array([self.base_stride*2**i for i in range(self.k+1)])
        windows = np.array([self.base_window*lamda for lamda in strides],np.int32)

        output_size = np.array(list(zip(self.input_w // strides, self.input_h // strides)))
        num_det = [output_w*output_h for output_w, output_h in output_size]
        det_offset = np.cumsum(num_det)
        label_conf = np.zeros((sum(num_det)),dtype=np.int64)
        xywh = np.zeros((self.max_objs, 7), dtype=np.float32) # x1 y1 x2 y2 ct_x ct_y stride
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)

        seg = [np.zeros((self.max_objs,window//self.base_stride,window//self.base_stride),dtype=np.float32) for window in windows]
        seg_ind = [np.zeros((self.max_objs),dtype=np.int64) for _ in windows]
        seg_mask = [np.zeros((self.max_objs),dtype=np.uint8) for _ in windows]
        num_objs = min(len(anns),self.max_objs)

        if num_objs > 0 :
            np.random.shuffle(anns)
        for k in range(num_objs):
            ann = anns[k]
            bbox = self._coco_box_to_bbox(ann['bbox'])
            cls_id = int(self.cat_ids[ann['category_id']])
            segment = self.coco.annToMask(ann)
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
                segment = segment[:, ::-1]
            bbox[:2] = affine_transform(bbox[:2], trans_input)
            bbox[2:] = affine_transform(bbox[2:], trans_input)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.input_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.input_h - 1)

            w , h = bbox[2:] - bbox[:2]
            max_edge = max(w,h)
            min_edge = min(w,h)
            ratio = max_edge / windows
            window_mask=(ratio > 0.5) * (ratio < 1.)  ## window > max(w,h) > window/2
            best_window = windows[window_mask]
            if len(best_window)==1 and min_edge > 0 :  ## min_edge must > 0
                segment= cv2.warpAffine(segment, trans_input,
                                     (self.input_w, self.input_h),
                                     flags=cv2.INTER_LINEAR)

                stride = strides[window_mask][0]
                best_window = best_window[0]
                feat_w, feat_h = output_size[window_mask][0]
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)

                xx , yy = np.arange(0,feat_w),np.arange(0,feat_h)
                xx , yy = (xx + 0.5) *stride,(yy + 0.5) * stride
                ct_feat_x,ct_feat_y = np.argmin(np.abs(ct[0] - xx)),np.argmin(np.abs(ct[1] - yy))  ## window ct close to box ct
                ct_x , ct_y  = int(xx[ct_feat_x]),int(yy[ct_feat_y])

                paded_segmnet = np.pad(segment, ((best_window//2, best_window//2), (best_window//2, best_window//2)), 'constant',
                                       constant_values=0)
                window_segment = paded_segmnet[ct_y : ct_y + best_window,ct_x : ct_x + best_window]

                feat_offset = det_offset[window_mask][0] - feat_w*feat_h
                output_offset =ct_feat_y * feat_w + ct_feat_x
                label_conf[feat_offset + output_offset] = (cls_id+1)

                xywh[k,0:4] =  bbox[0:4]
                xywh[k,4:6] =  ct_feat_x,ct_feat_y
                xywh[k, 6]  =  stride

                ind[k] = feat_offset + output_offset
                reg_mask[k] = 1

                window_segment = cv2.resize(window_segment,(best_window//self.base_stride,best_window//self.base_stride))
                window_index = windows.tolist().index(best_window)
                seg[window_index][k] = window_segment.astype(np.float32).copy()

                seg_ind[window_index][k] = output_offset
                seg_mask[window_index][k] = 1

                # cv2.imshow('',window_segment*255)
                # cv2.waitKey(0)

        ret = {'input':image ,'cls':label_conf,'ind': ind, 'xywh':xywh ,'reg_mask':reg_mask}
        for i in range(len(windows)):
            ret['seg_%d'%i] = seg[i]
            ret['seg_ind_%d' % i] = seg_ind[i]
            ret['seg_mask_%d' % i] = seg_mask[i]

        if self.cfg.test :
            ret['img'] = show

        return ret

if __name__ == '__main__':
    from config import cfg

    data = COCO(cfg,split='val',augment=False)
    print(len(data))
    for t in data:
        pass
