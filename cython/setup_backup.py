import os
from os.path import join as pjoin
from setuptools import setup
from Cython.Distutils import build_ext

import shutil
import sysconfig
from distutils.sysconfig import get_python_lib

import numpy as np
import versioneer
from Cython.Build import cythonize
from setuptools import find_packages, setup
from setuptools.extension import Extension



def find_in_path(name, path):
    """Find a file in a search path"""

    # Adapted fom http://code.activestate.com/recipes/52224
    for dir in path.split(os.pathsep):
        binpath = pjoin(dir, name)
        if os.path.exists(binpath):
            return os.path.abspath(binpath)
    return None


def locate_cuda():
    """Locate the CUDA environment on the system
    Returns a dict with keys 'home', 'nvcc', 'include', and 'lib64'
    and values giving the absolute path to each directory.
    Starts by looking for the CUDAHOME env variable. If not found,
    everything is based on finding 'nvcc' in the PATH.
    """

    # First check if the CUDAHOME env variable is in use
    if 'CUDAHOME' in os.environ:
        home = os.environ['CUDAHOME']
        nvcc = pjoin(home, 'bin', 'nvcc')
    else:
        # Otherwise, search the PATH for NVCC
        nvcc = find_in_path('nvcc', os.environ['PATH'])
        if nvcc is None:
            raise EnvironmentError('The nvcc binary could not be '
                'located in your $PATH. Either add it to your path, '
                'or set $CUDAHOME')
        home = os.path.dirname(os.path.dirname(nvcc))

    cudaconfig = {'home': home, 'nvcc': nvcc,
                  'include': pjoin(home, 'include'),
                  'lib64': pjoin(home, 'lib64')}
    for k, v in iter(cudaconfig.items()):
        if not os.path.exists(v):
            raise EnvironmentError('The CUDA %s path could not be '
                                   'located in %s' % (k, v))

    return cudaconfig


def customize_compiler_for_nvcc(self):
    """Inject deep into distutils to customize how the dispatch
    to gcc/nvcc works.
    If you subclass UnixCCompiler, it's not trivial to get your subclass
    injected in, and still have the right customizations (i.e.
    distutils.sysconfig.customize_compiler) run on it. So instead of going
    the OO route, I have this. Note, it's kindof like a wierd functional
    subclassing going on.
    """

    # Tell the compiler it can processes .cu
    self.src_extensions.append('.cu')
    self.src_extensions.append('.cuh')

    # Save references to the default compiler_so and _comple methods
    default_compiler_so = self.compiler_so
    super = self._compile

    print("here")

    # Now redefine the _compile method. This gets executed for each
    # object but distutils doesn't have the ability to change compilers
    # based on source extension: we add it.
    def _compile(obj, src, ext, cc_args, extra_postargs, pp_opts):
        print("src: " + str(src))
        if os.path.splitext(src)[1] == '.cu':
            # use the cuda for .cu files
            self.set_executable('compiler_so', CUDA['nvcc'])
            # use only a subset of the extra_postargs, which are 1-1
            # translated from the extra_compile_args in the Extension class
            postargs = extra_postargs['nvcc']
        elif os.path.splitext(src)[1] == '.pyx':
            print("Cythonize this file glob pattern")
            cythonize(src)
        else:
            postargs = extra_postargs['gcc']

        super(obj, src, ext, cc_args, postargs, pp_opts)
        # Reset the default compiler_so, which we might have changed for cuda
        self.compiler_so = default_compiler_so

    # Inject our redefined _compile method into the class
    self._compile = _compile



# Run the customize_compiler
class custom_build_ext(build_ext):
    def build_extensions(self):
        customize_compiler_for_nvcc(self.compiler)
        build_ext.build_extensions(self)

CUDA = locate_cuda()
cython_files = ["customcuda/**/*.pyx"]

ext = Extension(
        "*",
        sources=cython_files,
        include_dirs=[
            "../../cpp/include",
            "../../cpp/build/include",
            os.path.dirname(sysconfig.get_path("include")),
            np.get_include(),
            CUDA['include'],
        ],
        library_dirs=[
            get_python_lib(),
            os.path.join(os.sys.prefix, "lib"),
            CUDA['lib64'],
        ],
        runtime_library_dirs = [CUDA['lib64']],
        libraries=["cudf", "cudart"],
        language="c++",
        extra_compile_args= {
            'gcc': ["-std=c++14"],
            'nvcc': [
                '-arch=sm_30', '--ptxas-options=-v', '-c',
                '--compiler-options', "'-fPIC'"
            ]
        },
    )

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
    #ext_modules=cythonize(extensions),
    ext_modules=[ext],
    packages=find_packages(include=["customcuda", "customcuda.*"]),
    package_data={
        "customcuda._lib": ["*.pxd"],
        "customcuda._lib.includes": ["*.pxd"],
    },
    #cmdclass=versioneer.get_cmdclass(),
    cmdclass = {'build_ext': custom_build_ext},
    install_requires=["numba", "cython"],
    zip_safe=False
)



