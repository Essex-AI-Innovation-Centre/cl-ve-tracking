from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from glob import glob
import os
import numpy
import subprocess

# Change this if you target different host/device
nvcc_machine_code = '' #'-m64 -arch=compute_61 -code=sm_61'

gpu_sources_cpp = ' '.join(glob('../../gpu-kernels/*.cpp'))
gpu_sources_cu = ' '.join(glob('../../gpu-kernels/*.cu'))

gpu_kernel_build_cmd = f'/usr/local/cuda/bin/nvcc --compiler-options "-shared -fPIC" {gpu_sources_cpp} {gpu_sources_cu} -lib -o libgpu-kernels.so -O3 {nvcc_machine_code}'
os.system(gpu_kernel_build_cmd)

opencv_libs = subprocess.check_output('pkg-config --libs opencv4'.split())
opencv_libs = [name[2:] for name in str(opencv_libs, 'utf-8').split()][1:]

ext = Extension('pyvoldor_vo',
    sources = ['pyvoldor_vo.pyx'] + \
            [x for x in glob('../../voldor/*.cpp') if 'main.cpp' not in x],
    language = 'c++',
    library_dirs = ['.', '/usr/local/lib', '/usr/local/cuda/lib64'],
    libraries = ['gpu-kernels', 'cudart'] + opencv_libs,
    include_dirs = [numpy.get_include()]
)

setup(
    name='pyvoldor_vo',
    description='voldor visual odometry',
    author='Zhixiang Min',
    ext_modules=cythonize([ext])
)

# home = os.path.expanduser("~")
# os.system(f'yes y | mv libgpu-kernels.so {home}/palpable-ml/Tracking_and_localization/online -i')
# os.system(f'yes y | cp {home}/palpable-ml/Tracking_and_localization/online/libgpu-kernels.so {home}/palpable-ml/Tracking_and_localization/VOLDOR/demo -i')
# os.system(f'yes y | mv pyvoldor_vo.cpython-310-x86_64-linux-gnu.so {home}/palpable-ml/Tracking_and_localization/online -i')
# os.system(f'yes y | cp {home}/palpable-ml/Tracking_and_localization/online/pyvoldor_vo.cpython-310-x86_64-linux-gnu.so {home}/palpable-ml/Tracking_and_localization/VOLDOR/demo -i')
