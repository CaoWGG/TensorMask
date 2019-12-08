from torch.autograd import Function

from .. import swap_align2nat_cuda

class SwapAlign2NatFunction(Function):

    @staticmethod
    def forward(ctx, features,alpha,lamda,pad_val,align_corners):
        ctx.feature_size = features.size()
        ctx.alpha = alpha
        ctx.lamda = lamda
        ctx.align_corners = align_corners
        if features.is_cuda:
            output=swap_align2nat_cuda.forward(features,alpha,lamda,align_corners,pad_val)
        else:
            raise NotImplementedError
        return output

    @staticmethod
    def backward(ctx, grad_output):

        feature_size = ctx.feature_size
        alpha = ctx.alpha
        lamda = ctx.lamda
        align_corners = ctx.align_corners
        assert (feature_size is not None and grad_output.is_cuda)
        grad_input =  swap_align2nat_cuda.backward(grad_output.contiguous(),alpha,lamda,align_corners)
        return grad_input,None,None,None,None

swap_align2nat = SwapAlign2NatFunction.apply