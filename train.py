from models.tensormask import TensorMask
from lib.trainer import  Trainer
from lib.utils import load_model,save_model,Logger
from lib.coco import COCO
from lib  import optimer
from config import cfg as opt
import torch
import os


torch.backends.cudnn.benchmark= True  ## input size is not fixed
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
opt.gpus = [int(i) for i in opt.gpus_str.split(',')]
opt.gpus = list(range(len(opt.gpus)))
opt.batch_size = opt.batch_size * len(opt.gpus)
opt.save_dir = os.path.join(opt.save_dir,opt.exp_id)
logger = Logger(opt)


model = TensorMask(backbone=opt.backbone , num_cls=opt.num_class ,
                   base_window= opt.base_window ,
                   freezeBN=opt.frezeBN,freezeLayers=opt.frezeLayer,
                   align_corners= opt.align_corners)

optimizer = optimer.SGD([{'params':filter(lambda x:len(x.size()) == 4 ,model.parameters()),'weight_decay':0.0001 },
                            {'params': filter(lambda x:len(x.size()) <4,model.parameters())}],
                     lr=opt.lr,warm_up=1000,momentum=0.9,nesterov=True)
start_epoch = 0
if opt.weights != '' :
    model, optimizer, start_epoch = load_model(
      model, opt.weights, optimizer, opt.resume, opt.lr, opt.lr_step)
trainer = Trainer(opt,model,optimizer)
trainer.set_device(opt.gpus,opt.device)

print('Setting up data...')
val_loader = torch.utils.data.DataLoader(
    COCO(cfg=opt, split='val',augment=False),
    batch_size=8,
    shuffle=False,
    num_workers=8,
    pin_memory=True
)
train_loader = torch.utils.data.DataLoader(
    COCO(cfg=opt, split='train',augment=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=8,
    pin_memory=True
)

print('Starting training...')
best = 1e10
for epoch in range(start_epoch + 1, opt.num_epochs + 1):
    log_dict_train, _ = trainer.train(epoch, train_loader)
    logger.write('epoch: {} |'.format(epoch))
    for k, v in log_dict_train.items():
        logger.scalar_summary('train_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
    with torch.no_grad():
        log_dict_val, preds = trainer.val(epoch, val_loader)
    for k, v in log_dict_val.items():
        logger.scalar_summary('val_{}'.format(k), v, epoch)
        logger.write('{} {:8f} | '.format(k, v))
    if log_dict_val['loss'] < best:
        best = log_dict_val['loss']
        save_model(os.path.join(opt.save_dir, 'model_best.pth'),
               epoch, model)
    save_model(os.path.join(opt.save_dir, 'model_last.pth'),
             epoch, model, optimizer)
    logger.write('\n')
    if epoch in opt.lr_step:
        save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)),
             epoch, model, optimizer)
        lr = opt.lr * (0.1 ** (opt.lr_step.index(epoch) + 1))
        print('Drop LR to', lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr