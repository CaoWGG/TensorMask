import os
from models.detector import Detector
from pycocotools.cocoeval import COCOeval
import pycocotools.coco as coco
import pycocotools.mask as mask_util
import numpy as np
from tqdm import tqdm
from config import cfg as opt
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
valid_ids = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13,
            14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
            24, 25, 27, 28, 31, 32, 33, 34, 35, 36,
            37, 38, 39, 40, 41, 42, 43, 44, 46, 47,
            48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
            58, 59, 60, 61, 62, 63, 64, 65, 67, 70,
            72, 73, 74, 75, 76, 77, 78, 79, 80, 81,
            82, 84, 85, 86, 87, 88, 89, 90]

## config recover weights
opt.weights = 'exp/coco_person/model_last.pth'
opt.vis_trehs = 0.01
split = 'val'

detector = Detector(opt)
data = coco.COCO(os.path.join(
            opt.data_dir, 'annotations',
            'instances_{}2017.json').format(split))

if opt.class_name!='*' :  ## for one class
    catIds = data.getCatIds(opt.class_name)
    imgIds = data.getImgIds(catIds=catIds)
    valid_ids = catIds

detections = []
for img_id in tqdm(data.getImgIds()):
    img_name = os.path.join(os.path.join(opt.data_dir, '{}2017'.format(split)),
                            data.loadImgs(ids=[img_id])[0]['file_name']).strip()
    boxs,masks = detector.run(img_name,vis=False)
    for i,det in enumerate(boxs):
        x, y, x1, y1, conf, cls = det[:6]
        detection = {
            "image_id": img_id,
            "category_id": int(valid_ids[int(cls)]),
            'segmentation':mask_util.encode(np.asfortranarray(masks[i])),
            #"bbox": [x, y, x1 - x, y1 - y],
            "score": float("{:.2f}".format(conf))
        }
        detections.append(detection)
coco_dets = data.loadRes(detections)
coco_eval = COCOeval(data, coco_dets, "segm")

if opt.class_name!='*':  ## for one class
    coco_eval.params.imgIds = imgIds
    coco_eval.params.catIds = catIds

coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
