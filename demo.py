import os
import cv2
from config import cfg as opt
from models.detector import Detector
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv', 'h264']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

opt.demo = '/data/yoloCao/DataSet/coco/val2017'
opt.weights = 'exp/coco_person/model_last.pth'
opt.vis_trehs = 0.4
detector = Detector(opt)
cv2.namedWindow('result', cv2.WINDOW_NORMAL)
cv2.resizeWindow('result', 1024, 768)
if opt.demo == 'webcam' or \
    opt.demo[opt.demo.rfind('.') + 1:].lower() in video_ext:
    cam = cv2.VideoCapture(0 if opt.demo == 'webcam' else opt.demo)

    while True:
        _, img = cam.read()
        ret = detector.run(img)
        if cv2.waitKey(1) == 27:
            break
else:
    if os.path.isdir(opt.demo):
        image_names = []
        ls = os.listdir(opt.demo)
        for file_name in sorted(ls):
            ext = file_name[file_name.rfind('.') + 1:].lower()
            if ext in image_ext:
                image_names.append(os.path.join(opt.demo, file_name))
    elif opt.demo.endswith('.txt'):
        image_names = []
        with open(opt.demo) as f:
            lines = f.readlines()
        for file_name in sorted(lines):
            file_name = file_name.strip()
            if file_name.split('.')[-1] in image_ext:
                image_names.append(file_name)
    else:
        image_names = [opt.demo]

    for (image_name) in image_names:
        ret = detector.run(image_name)
        if cv2.waitKey(0) == 27:
            break

