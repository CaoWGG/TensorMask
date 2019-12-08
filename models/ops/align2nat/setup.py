from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='align2nat_cuda',
    ext_modules=[
        CUDAExtension('swap_align2nat_cuda', [
            'src/swap_align2nat_cuda.cpp',
            'src/swap_align2nat_kernel.cu',
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
