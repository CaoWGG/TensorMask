from models.ops.align2nat.functions.swap_align2nat import swap_align2nat
from models.res_fpn import resnet_fpn_backbone
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

class Subnet(nn.Module):
    def __init__(self, in_channels = 256,mid_channels = 256 ,num_cls = -1):
        super(Subnet, self).__init__()

        self.conv = nn.Sequential(nn.Conv2d(in_channels, mid_channels, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(mid_channels, mid_channels, 3, padding=1),
                                  nn.ReLU(inplace=True))
        self.num_cls = num_cls
        if num_cls > 0:
            self.fc = nn.Conv2d(mid_channels, num_cls, 3, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.num_cls > 0:
            x = self.fc(x)
            x = x.permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.num_cls)
        return x

class TensorMask(nn.Module):
    def __init__(self,backbone = 'resnet50',num_cls = 80,base_window = 12,
                 freezeBN = True,freezeLayers = False ,align_corners = True):
        super(TensorMask,self).__init__()
        self.align_corners = align_corners
        self.base_fpn = resnet_fpn_backbone(backbone,pretrained=True,freezeBN=freezeBN,freezeLayers=freezeLayers,align_corners=align_corners)

        self.cls_subnet = Subnet(in_channels=256,mid_channels=256,num_cls = num_cls)

        self.box_subnet = Subnet(in_channels=256,mid_channels=128,num_cls = 4)

        self.mask_subnet = Subnet(in_channels=256,mid_channels=128)

        self.mask_fuse = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),nn.ReLU(inplace=True))
        self.mask_head = nn.Conv2d(128, base_window**2 , kernel_size=1, padding=0)

        self.base_window = base_window


        nn.init.constant_(self.box_subnet.fc.bias, 1)   ###  training box start with a little box not a point(its hard).
        nn.init.kaiming_uniform_(self.mask_fuse[0].weight, a=1)
        nn.init.constant_(self.mask_fuse[0].bias, 0)
        nn.init.kaiming_uniform_(self.mask_head.weight, a=1)
        nn.init.constant_(self.mask_head.bias, 0)

        nn.init.constant_(self.cls_subnet.fc.bias,-math.log((1-0.01)/0.01))

    def forward(self, x):
        x = self.base_fpn(x)
        cls_branch = torch.cat([self.cls_subnet(feat) for feat in x],dim = 1)
        box_branch = torch.cat([self.box_subnet(feat) for feat in x],dim = 1)
        mask_branch = [self.mask_subnet(feat) for feat in x]

        ret = {'cls':cls_branch,'box':box_branch}

        finest_feat = mask_branch[0]
        ##  tensor bipyamid
        for i in range(len(mask_branch)):
            x  = mask_branch[i]
            if i > 0:
                x = F.interpolate(x, scale_factor=2**i, mode="bilinear" ,align_corners=self.align_corners )
            x = self.mask_fuse(x + finest_feat)
            x = self.mask_head(x)
            x = x.view(x.size(0), self.base_window, self.base_window, x.size(2), x.size(3))
            x = swap_align2nat(x, 1 , 2**i ,-6., self.align_corners)
            ret['%d'%i]= x.permute(0, 3, 4, 1 , 2).contiguous()

        return ret

if __name__ == '__main__':
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES','1')
    import torch
    model = TensorMask(num_cls=1,base_window=10)
    model.cuda()
    input = torch.zeros([1,3,512,512]).cuda()
    out = model(input)
    pass
