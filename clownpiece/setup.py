import os

from setuptools import setup, Extension
import pybind11

parallel_modes = {
    "serial": 0,
    "stdthread": 1,
    "threadpool": 3,
}

parallel_mode = os.environ.get("CP_PARALLEL_MODE", "serial").lower()
if parallel_mode not in parallel_modes:
    choices = ", ".join(parallel_modes)
    raise RuntimeError(
        f"Unsupported CP_PARALLEL_MODE={parallel_mode!r}; choose one of: {choices}"
    )

extra_compile_args = ['-std=c++17', '-O3', '-Wall']
extra_link_args = []
if parallel_mode in ("stdthread", "threadpool"):
    extra_compile_args.append('-pthread')
    extra_link_args.append('-pthread')

ext_modules = [
    Extension(
        'tensor_impl',
        sources=[
            "tensor/meta.cc",
            "tensor/parallel.cc",
            "tensor/tensor.cc",
            "tensor/tensor_pybind.cc"
        ],
        include_dirs=[
            pybind11.get_include(),
            "tensor/",
        ],
        language='c++',
        define_macros=[('CP_PARALLEL_BACKEND', str(parallel_modes[parallel_mode]))],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name='tensor_impl',
    version='0.0.1',
    ext_modules=ext_modules,
)
