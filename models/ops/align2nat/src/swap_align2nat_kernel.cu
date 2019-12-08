#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
#include <ATen/AccumulateType.h>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t area_pixel_compute_source_index(
    accscalar_t scale,
    int dst_index,
    bool align_corners) {
  if (align_corners) {
    return scale * dst_index;
  } else {
    accscalar_t src_idx = scale * (dst_index + static_cast<accscalar_t>(0.5)) -
        static_cast<accscalar_t>(0.5);
    // See Note[Follow Opencv resize logic]
    return (src_idx < static_cast<accscalar_t>(0))
        ? static_cast<accscalar_t>(0)
        : src_idx;
  }
}

__device__ __forceinline__ size_t
loaction(const size_t n,const size_t v,const size_t u,const size_t y,const size_t x,
    const size_t V,const size_t U,const size_t H,const size_t W) {
  return x + W*(y + H*(u + U*(v + V*n)));
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t get_val(const scalar_t*data,
const size_t n,const size_t v,const size_t u,const size_t y,const size_t x,
               const size_t V,const size_t U,const size_t H,const size_t W,const scalar_t pad_val) {
    if (x <0 || x >= W || y < 0 || y >= H ){
        return pad_val;
    }else{
        return data[x + W*(y + H*(u + U*(v + V*n)))];
    }
}

template <typename scalar_t,typename accscalar_t>
__global__ void SwapAlign2NatForward(const int nthreads, const scalar_t *bottom_data,scalar_t *top_data,
                                const accscalar_t scaleV,const accscalar_t scaleU,
                                const int newV,const int newU,
                                const int newH,const int newW,
                                const accscalar_t scaleH,const accscalar_t scaleW,
                                const int orgV,const int orgU,
                                const int orgH,const int orgW,
                                const int alpha,const bool align_corners,const scalar_t pad_val
                                ) {
    const float v_offset =  -newV/2;
    const float u_offset =  -newU/2;
    int n,ov,ou,oh,ow,bottom_h,bottom_w;
    CUDA_1D_KERNEL_LOOP(index,nthreads){
        // (n, ov, ou, oh, ow) is an element in the top_data
        ow = index % newW;
        oh = (index / newW) % newH;
        ou = (index / newW / newH) % newU;
        ov = (index / newW / newH / newU) % newV;
        n  =  index / newW / newH / newU / newV;
        if (newV==orgV && newU==orgU && newW==orgW && newH==orgH){
            bottom_h =  oh + alpha * (ov + v_offset);
            bottom_w =  ow + alpha * (ou + u_offset);
            top_data[index] = get_val(bottom_data,n,ov,ou,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val);

        } else {
            //  h,w

            const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
                scaleH, oh, align_corners);
            const int h1 = h1r;
            const int h1p = (h1 < orgH - 1) ? 1 : 0;
            const accscalar_t h1lambda = h1r - h1;
            const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
            //
            const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
                scaleW, ow, align_corners);
            const int w1 = w1r;
            const int w1p = (w1 < orgW - 1) ? 1 : 0;
            const accscalar_t w1lambda = w1r - w1;
            const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;


            // v,u

            const accscalar_t v1r = area_pixel_compute_source_index<accscalar_t>(
                 scaleV, ov, align_corners);
            const int v1 = v1r;
            const int v1p = (v1 < orgV - 1) ? 1 : 0;
            const accscalar_t v1lambda = v1r - v1;
            const accscalar_t v0lambda = static_cast<accscalar_t>(1) - v1lambda;


            const accscalar_t u1r = area_pixel_compute_source_index<accscalar_t>(
                scaleU, ou, align_corners);
            const int u1 = u1r;
            const int u1p = (u1 < orgU - 1) ? 1 : 0;
            const accscalar_t u1lambda = u1r - u1;
            const accscalar_t u0lambda = static_cast<accscalar_t>(1) - u1lambda;

            accscalar_t h0w0,h0w1,h1w0,h1w1;

            bottom_h =  h1 + alpha * (ov + v_offset);
            bottom_w =  w1 + alpha * (ou + u_offset);
            h0w0 = v0lambda * u0lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1,u1,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val))+
                   v0lambda * u1lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1,u1+u1p,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val)) +
                   v1lambda * u0lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1+v1p,u1,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val)) +
                   v1lambda * u1lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1+v1p,u1+u1p,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val));

            bottom_h =  h1 + alpha * (ov + v_offset);
            bottom_w =  w1 + w1p + alpha * (ou + u_offset);
            h0w1 = v0lambda * u0lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1,u1,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val))+
                   v0lambda * u1lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1,u1+u1p,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val)) +
                   v1lambda * u0lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1+v1p,u1,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val)) +
                   v1lambda * u1lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1+v1p,u1+u1p,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val));

            bottom_h =  h1 + h1p + alpha * (ov + v_offset);
            bottom_w =  w1  + alpha * (ou + u_offset);
            h1w0 = v0lambda * u0lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1,u1,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val))+
                   v0lambda * u1lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1,u1+u1p,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val)) +
                   v1lambda * u0lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1+v1p,u1,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val)) +
                   v1lambda * u1lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1+v1p,u1+u1p,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val));

            bottom_h =  h1  + h1p  + alpha * (ov + v_offset);
            bottom_w =  w1  + w1p  + alpha * (ou + u_offset);
            h1w1 = v0lambda * u0lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1,u1,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val))+
                   v0lambda * u1lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1,u1+u1p,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val)) +
                   v1lambda * u0lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1+v1p,u1,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val)) +
                   v1lambda * u1lambda * static_cast<accscalar_t>(get_val(bottom_data,n,v1+v1p,u1+u1p,bottom_h,bottom_w,orgV,orgU,orgH,orgW,pad_val));

            const accscalar_t val = h0lambda * w0lambda * h0w0 +
                                    h0lambda * w1lambda * h0w1 +
                                    h1lambda * w0lambda * h1w0 +
                                    h1lambda * w1lambda * h1w1 ;

            top_data[index] = static_cast<scalar_t>(val);
        }

    }
}


template <typename scalar_t,typename accscalar_t>
__global__ void SwapAlign2NatBackward(const int nthreads, const scalar_t *bottom_data,scalar_t *top_data,
                                const accscalar_t scaleV,const accscalar_t scaleU,
                                const int newV,const int newU,
                                const int newH,const int newW,
                                const accscalar_t scaleH,const accscalar_t scaleW,
                                const int orgV,const int orgU,
                                const int orgH,const int orgW,
                                const int alpha,const bool align_corners
                                ) {
    const int v_offset =  -newV/2;
    const int u_offset =  -newU/2;
    int n,ov,ou,oh,ow,top_h,top_w;
    size_t top_offset ;
    CUDA_1D_KERNEL_LOOP(index,nthreads){
        // (n, ov, ou, oh, ow) is an element in the bottom_data
        ow = index % newW;
        oh = (index / newW) % newH;
        ou = (index / newW / newH) % newU;
        ov = (index / newW / newH / newU) % newV;
        n  =  index / newW / newH / newU / newV;
        if (newV==orgV && newU==orgU && newW==orgW && newH==orgH){
            top_h =  oh + alpha * (ov + v_offset);
            top_w =  ow + alpha * (ou + u_offset);
            if (!(top_w <0 || top_w >= orgW || top_h < 0 || top_h >= orgH))
            {
                top_offset =  loaction(n,ov,ou,top_h,top_w,newV,newU,newH,newW);
                top_data[top_offset] = bottom_data[index];
            }
        } else {
            //  h,w

            const accscalar_t h1r = area_pixel_compute_source_index<accscalar_t>(
                scaleH, oh, align_corners);
            const int h1 = h1r;
            const int h1p = (h1 < orgH - 1) ? 1 : 0;
            const accscalar_t h1lambda = h1r - h1;
            const accscalar_t h0lambda = static_cast<accscalar_t>(1) - h1lambda;
            //
            const accscalar_t w1r = area_pixel_compute_source_index<accscalar_t>(
                scaleW, ow, align_corners);
            const int w1 = w1r;
            const int w1p = (w1 < orgW - 1) ? 1 : 0;
            const accscalar_t w1lambda = w1r - w1;
            const accscalar_t w0lambda = static_cast<accscalar_t>(1) - w1lambda;


            // v,u

            const accscalar_t v1r = area_pixel_compute_source_index<accscalar_t>(
                 scaleV, ov, align_corners);
            const int v1 = v1r;
            const int v1p = (v1 < orgV - 1) ? 1 : 0;
            const accscalar_t v1lambda = v1r - v1;
            const accscalar_t v0lambda = static_cast<accscalar_t>(1) - v1lambda;


            const accscalar_t u1r = area_pixel_compute_source_index<accscalar_t>(
                scaleU, ou, align_corners);
            const int u1 = u1r;
            const int u1p = (u1 < orgU - 1) ? 1 : 0;
            const accscalar_t u1lambda = u1r - u1;
            const accscalar_t u0lambda = static_cast<accscalar_t>(1) - u1lambda;

            const accscalar_t d2val = static_cast<accscalar_t>(bottom_data[index]);

            top_h =  h1 + alpha * (ov + v_offset);
            top_w =  w1 + alpha * (ou + u_offset);
            if (!(top_w <0 || top_w >= orgW || top_h < 0 || top_h >= orgH) ){
               top_offset = loaction(n,v1,u1,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h0lambda * w0lambda * v0lambda * u0lambda * d2val));
               top_offset = loaction(n,v1,u1 + u1p,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h0lambda * w0lambda * v0lambda * u1lambda * d2val));
               top_offset = loaction(n,v1+v1p,u1,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h0lambda * w0lambda * v1lambda * u0lambda * d2val));
               top_offset = loaction(n,v1+v1p,u1+u1p,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h0lambda * w0lambda * v1lambda * u1lambda * d2val));
            }

            top_h =  h1 + alpha * (ov + v_offset);
            top_w =  w1 + w1p + alpha * (ou + u_offset);
            if (!(top_w <0 || top_w >= orgW || top_h < 0 || top_h >= orgH) ){
               top_offset = loaction(n,v1,u1,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h0lambda * w1lambda * v0lambda * u0lambda * d2val));
               top_offset = loaction(n,v1,u1 + u1p,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h0lambda * w1lambda * v0lambda * u1lambda * d2val));
               top_offset = loaction(n,v1+v1p,u1,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h0lambda * w1lambda * v1lambda * u0lambda * d2val));
               top_offset = loaction(n,v1+v1p,u1+u1p,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h0lambda * w1lambda * v1lambda * u1lambda * d2val));
            }

            top_h =  h1 + h1p + alpha * (ov + v_offset);
            top_w =  w1  + alpha * (ou + u_offset);
            if (!(top_w <0 || top_w >= orgW || top_h < 0 || top_h >= orgH) ){
               top_offset = loaction(n,v1,u1,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h1lambda * w0lambda * v0lambda * u0lambda * d2val));
               top_offset = loaction(n,v1,u1 + u1p,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h1lambda * w0lambda * v0lambda * u1lambda * d2val));
               top_offset = loaction(n,v1+v1p,u1,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h1lambda * w0lambda * v1lambda * u0lambda * d2val));
               top_offset = loaction(n,v1+v1p,u1+u1p,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h1lambda * w0lambda * v1lambda * u1lambda * d2val));
            }

            top_h =  h1  + h1p  + alpha * (ov + v_offset);
            top_w =  w1  + w1p  + alpha * (ou + u_offset);
            if (!(top_w <0 || top_w >= orgW || top_h < 0 || top_h >= orgH) ){
               top_offset = loaction(n,v1,u1,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h1lambda * w1lambda * v0lambda * u0lambda * d2val));
               top_offset = loaction(n,v1,u1 + u1p,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h1lambda * w1lambda * v0lambda * u1lambda * d2val));
               top_offset = loaction(n,v1+v1p,u1,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h1lambda * w1lambda * v1lambda * u0lambda * d2val));
               top_offset = loaction(n,v1+v1p,u1+u1p,top_h,top_w,orgV,orgU,orgH,orgW);
               atomicAdd(top_data + top_offset, static_cast<scalar_t>(h1lambda * w1lambda * v1lambda * u1lambda * d2val));
            }
        }

    }
}


template <typename scalar_t>
static inline scalar_t area_pixel_compute_scale(
    int64_t input_size,
    int64_t output_size,
    bool align_corners) {

  if (output_size > 1) {
    return align_corners
        ? static_cast<scalar_t>(input_size - 1) / (output_size - 1)
        : static_cast<scalar_t>(input_size) / output_size;
  } else {
    return scalar_t(0);
  }
}
int SwapAlign2NatForwardLaucher(const at::Tensor& input,at::Tensor& output,
                           const int alpha,const bool align_corners,const float pad_val){
  const int B = output.size(0);
  const int newV = output.size(1);
  const int newU = output.size(2);
  const int newH = output.size(3);
  const int newW = output.size(4);
  const int orgV = input.size(1);
  const int orgU = input.size(2);
  const int orgH = input.size(3);
  const int orgW = input.size(4);
  const int output_size = B*newV*newU*newH*newW;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.type(), "SwapAlign2NatForwardLaucher", ([&] {
        const scalar_t *bottom_data = input.data<scalar_t>();
        scalar_t *top_data = output.data<scalar_t>();
        using accscalar_t = at::acc_type<scalar_t, true>;
        const accscalar_t rV = area_pixel_compute_scale<accscalar_t>(
            orgV, newV, align_corners);
        const accscalar_t rU = area_pixel_compute_scale<accscalar_t>(
            orgU, newV, align_corners);
        const accscalar_t rH = area_pixel_compute_scale<accscalar_t>(
            orgH, newH, align_corners);
        const accscalar_t rW = area_pixel_compute_scale<accscalar_t>(
            orgW, newW, align_corners);
        SwapAlign2NatForward<scalar_t,accscalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(output_size,bottom_data,top_data,
                            rV,rU,newV,newU,newH,newW,
                            rH,rW,orgV,orgU,orgH,orgW,
                            alpha,align_corners,static_cast<scalar_t>(pad_val)
                            );
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}


int SwapAlign2NatBackwardLaucher(const at::Tensor& grad_output,at::Tensor& grad_input,
                           const int alpha,const bool align_corners){
  int B = grad_output.size(0);
  int newV = grad_output.size(1);
  int newU = grad_output.size(2);
  int newH = grad_output.size(3);
  int newW = grad_output.size(4);
  int orgV = grad_input.size(1);
  int orgU = grad_input.size(2);
  int orgH = grad_input.size(3);
  int orgW = grad_input.size(4);
  const int output_size = B*newV*newU*newH*newW;
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_output.type(), "SwapAlign2NatBackwardLaucher", ([&] {
        const scalar_t *bottom_data = grad_output.data<scalar_t>();
        scalar_t *top_data = grad_input.data<scalar_t>();
        using accscalar_t = at::acc_type<scalar_t, true>;
        const accscalar_t rV = area_pixel_compute_scale<accscalar_t>(
            orgV, newV, align_corners);
        const accscalar_t rU = area_pixel_compute_scale<accscalar_t>(
            orgU, newV, align_corners);
        const accscalar_t rH = area_pixel_compute_scale<accscalar_t>(
            orgH, newH, align_corners);
        const accscalar_t rW = area_pixel_compute_scale<accscalar_t>(
            orgW, newW, align_corners);
        SwapAlign2NatBackward<scalar_t,accscalar_t>
            <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(output_size,bottom_data,top_data,
                            rV,rU,newV,newU,newH,newW,
                            rH,rW,orgV,orgU,orgH,orgW,
                            alpha,align_corners
                            );
      }));
  THCudaCheck(cudaGetLastError());
  return 1;
}