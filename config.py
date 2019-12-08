from easydict import EasyDict

cfg = EasyDict()

cfg.backbone = 'resnet50'
cfg.frezeBN = False
cfg.frezeLayer = False
cfg.align_corners = False   ## ref torch.nn.functional.interpolate /// when align_corners==False : [Follow Opencv resize logic]
cfg.resume = True  ## resume an experiment.Reloaded the optimizer parameter . make sure cfg.weights!="".
cfg.weights = ''
cfg.device = 'cuda'

cfg.cls_weights = 1.
cfg.xywh_weights = 1/4.
cfg.mask_weights = 2.   ## from paper

cfg.data_dir = '/data/yoloCao/DataSet/coco'
cfg.num_class = 1
cfg.class_name = 'person'   ## [person , *]
cfg.input_h = 512   ## 512 % 128 = 0
cfg.input_w = 640   ## 640 % 128 = 0
cfg.base_window = 12   ## base_window%2==0 in this impl..|| max window = 12* base_stride * 2^5 / 2  =  768  > 640
cfg.base_stride = 4    ## feat_2 --> strideHW=4
cfg.k = 5 # 0 1 2 3 4 5
cfg.jitter = 0.3


cfg.lr = 0.02           ## from paper
cfg.num_epochs = 72     ## from paper
cfg.lr_step = [64,70]   ## from paper
cfg.warm_up = 1000
cfg.batch_size = 6

cfg.gpus_str = '0,1,2'

cfg.save_dir = 'exp'
cfg.exp_id = 'coco_person'
cfg.print_iter = 1
cfg.test = False
cfg.vis_thresh = 0.3
cfg.show_box =  False
cfg.demo = ''
