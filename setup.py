from setuptools import setup, find_packages
from torch.utils import cpp_extension

setup(
    name="extension",
    packages=find_packages(include=['extension']),
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="extension_cpp",
            sources=["csrc/muladd.cc"],
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
)
