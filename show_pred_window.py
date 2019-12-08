from models.tensormask import TensorMask
from config import cfg as opt
from lib.utils import load_model,save_model
from lib.coco import COCO
import numpy as np
import torch
import os
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
model = TensorMask(backbone=opt.backbone , num_cls=opt.num_class ,
                   base_window= opt.base_window ,
                   freezeBN=opt.frezeBN,freezeLayers=opt.frezeLayer,
                   align_corners=opt.align_corners)

opt.test = True
opt.weights = 'exp/coco_person/model_last.pth'
model = load_model(model, opt.weights)
model.eval()
model.cuda()
val_loader = torch.utils.data.DataLoader(
    COCO(cfg=opt, split='val',augment=False),
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True
)
strides = np.array([opt.base_stride * 2 ** i for i in range(opt.k + 1)])
windows = np.array([opt.base_window * lamda for lamda in strides], np.int32)

output_size = np.array(list(zip(opt.input_w // strides, opt.input_h // strides)))
num_det = [output_w * output_h for output_w, output_h in output_size]
det_offset = np.cumsum(num_det)
for batch in val_loader:
    image= batch['img'].numpy()[0]
    input = batch['input'].cuda()
    output= model(input)

    socres, cls = torch.max(output['cls'].sigmoid_(), dim=-1)
    socres = socres.detach().cpu().numpy()
    cls = cls.detach().cpu().numpy()
    box= output['box'].detach().cpu().numpy()
    seg = [output['%d'%i].sigmoid_().detach().cpu().numpy() for i in range(opt.k+1)]
    topk_inds = np.where(socres > 0.4)

    for det_num in topk_inds[1]:
        p = socres[0,det_num]
        b = box[0,det_num,:]
        for id,num in enumerate(det_offset):
            if num > det_num:
                break
        offset = det_num-det_offset[id-1]if id > 0 else det_num
        width,hight = output_size[id]

        ### ct_int_feat
        y = int(offset/width)
        x = int(offset%width)

        window_seg = seg[id][0,y,x,:,:]

        ### ct_int
        x ,y = int((x + 0.5) * strides[id]),int((y + 0.5) * strides[id])
        ### show box
        b[0:2] = x - b[0]*strides[id] ,y - b[1]*strides[id]
        b[2:4] = x + b[2]*strides[id] ,y + b[3]*strides[id]
        b = b.astype(np.int)
        cv2.rectangle(image,(b[0],b[1]),(b[2],b[3]),(255,0,0),2)


        ### show mask
        img_h,img_w  = image.shape[:2]
        paste_x,paste_y,paste_x1,paste_y1= x - windows[id]//2, y- windows[id]//2,x + windows[id]//2,y + windows[id]//2

        window_x,window_y,window_x1,window_y1 = max(-paste_x,0),max(-paste_y,0), \
                                                windows[id]-max(0,paste_x1-img_w), \
                                                windows[id]-max(0,paste_y1-img_h)

        paste_x, paste_y, paste_x1, paste_y1 = max(paste_x, 0), max(paste_y, 0), min(paste_x1, img_w), min(paste_y1,
                                                                                                           img_h)
        window_seg = cv2.resize(window_seg,(windows[id],windows[id]))
        window_seg = (window_seg>0.5)

        ### paste to img
        window_seg_paste = window_seg[window_y:window_y1,window_x:window_x1]
        color = np.array([[np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]])
        image[paste_y:paste_y1,paste_x:paste_x1][window_seg_paste] = image[paste_y:paste_y1,paste_x:paste_x1][window_seg_paste]*0.2 + color*0.8

        ### show
        cv2.imshow('window',(window_seg).astype(np.uint8)*255)
        cv2.imshow('',image)
        cv2.waitKey(0)
