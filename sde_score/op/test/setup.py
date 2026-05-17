from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="upfirdn2d",
    ext_modules=[
        CUDAExtension(
            "upfirdn2d",
            ["upfirdn2d.cpp"],
            include_dirs=[],  # 自动包含PyTorch路径，无需手动指定
        )
    ],
    cmdclass={"build_ext": BuildExtension}
)