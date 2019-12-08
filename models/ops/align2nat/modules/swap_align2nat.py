from torch.nn.modules.module import Module
from ..functions.swap_align2nat import SwapAlign2NatFunction


class SwapAlign2Nat(Module):
    def __init__(self, alpha=1 ,lamda = 1, pad_val = -9.0 ,align_corners=True ):
        super(SwapAlign2Nat, self).__init__()
        self.alpha = alpha
        self.lamda = lamda
        self.align_corners = align_corners
        self.pad_val = pad_val

    def forward(self, features):
        return SwapAlign2NatFunction.apply(features , self.alpha,self.lamda,self.pad_val,self.align_corners)
