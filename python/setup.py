# Copyright (c) 2020, NVIDIA CORPORATION.
import os
import shutil
import sysconfig
from distutils.sysconfig import get_python_lib

import numpy as np
import versioneer
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension

install_requires = ["numba", "cython"]
cython_files = ["customcuda/**/*.pyx"]

CUDA_HOME = os.environ.get("CUDA_HOME", False)
if not CUDA_HOME:
    path_to_cuda_gdb = shutil.which("cuda-gdb")
    if path_to_cuda_gdb is None:
        raise OSError(
            "Could not locate CUDA. "
            "Please set the environment variable "
            "CUDA_HOME to the path to the CUDA installation "
            "and try again."
        )
    CUDA_HOME = os.path.dirname(os.path.dirname(path_to_cuda_gdb))

if not os.path.isdir(CUDA_HOME):
    raise OSError(
        f"Invalid CUDA_HOME: " "directory does not exist: {CUDA_HOME}"
    )

cuda_include_dir = os.path.join(CUDA_HOME, "include")

extensions = [
    Extension(
        "*",
        sources=cython_files,
        include_dirs=[
            "../cpp/include",
            "../cpp/build/include",
            os.path.dirname(sysconfig.get_path("include")),
            np.get_include(),
            cuda_include_dir,
        ],
        library_dirs=[
            get_python_lib(),
            os.path.join(os.sys.prefix, "lib"),
        ],
        libraries=["cudf"],
        language="c++",
        extra_compile_args=["-std=c++14"],
    ),
]

setup(
    name="customcuda",
    version=versioneer.get_version(),
    description="customcuda - Use existing CUDA Kernels with RapidsAI",
    url="https://github.com/jdye64/libcudf-examples",
    author="Jeremy Dyer",
    license="Apache 2.0",
    classifiers=[
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: C++",
        "Programming Language :: CUDA",
        "Programming Language :: Python",
    ],
    # Include the separately-compiled shared library
    setup_requires=["cython"],
    ext_modules=cythonize(extensions),
    packages=find_packages(include=["customcuda", "customcuda.*"]),
    package_data={
        "customcuda._libxx": ["*.pxd"],
        "customcuda._libxx.includes": ["*.pxd"],
    },
    cmdclass=versioneer.get_cmdclass(),
    install_requires=install_requires,
    zip_safe=False,
)