#include <torch/extension.h>

#include <cmath>
#include <vector>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int SwapAlign2NatForwardLaucher(const at::Tensor& input,at::Tensor& output,
                           const int alpha,const bool align_corners,const float pad_val);
int SwapAlign2NatBackwardLaucher(const at::Tensor& grad_output,at::Tensor& grad_input,
                           const int alpha,const bool align_corners);

at::Tensor swap_align2nat_forward_cuda(const at::Tensor& input , const int alpha, const int lamda,const bool align_corners,const float pad_val) {
  CHECK_INPUT(input);
  int B = input.size(0);
  int V = input.size(1);
  int U = input.size(2);
  int H = input.size(3);
  int W = input.size(4);
  auto output = torch::zeros_like(input);
  output.resize_({B, lamda*V, lamda*U, H/lamda,W/lamda});
  output.contiguous();
  CHECK_INPUT(output);
  SwapAlign2NatForwardLaucher(input,output,alpha,align_corners,pad_val);
  return output;
}

at::Tensor swap_align2nat_backward_cuda(const at::Tensor& grad_output,const int alpha,const int lamda,const bool align_corners) {
  CHECK_INPUT(grad_output);
  int B = grad_output.size(0);
  int V = grad_output.size(1);
  int U = grad_output.size(2);
  int H = grad_output.size(3);
  int W = grad_output.size(4);
  auto grad_input = torch::zeros_like(grad_output);
  grad_input.resize_({B, V/lamda, U/lamda, H*lamda,W*lamda});
  grad_input.contiguous();
  CHECK_INPUT(grad_input);
  SwapAlign2NatBackwardLaucher(grad_output,grad_input,alpha,align_corners);
  return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &swap_align2nat_forward_cuda, "SwapAlign2Nat forward (CUDA)");
  m.def("backward", &swap_align2nat_backward_cuda, "SwapAlign2Nat backward (CUDA)");
}
