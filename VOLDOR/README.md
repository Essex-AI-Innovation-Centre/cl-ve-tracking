# VOLDOR

VOLDOR is a real-time dense-indirect system that takes dense optical flows as input.

## Installation

```shell
cd slam_py/install
sudo apt install libopencv-dev python-opengl
python setup_linux_vo.py build_ext -i
```

If succeeded, `libgpu-kernels.so` and `pyvoldor_vo.xx.so` will appear in the `install` folder.

**Common Build Errors**

1.
    ```bash
    /home/user/anaconda3/envs/palenv/compiler_compat/ld: cannot find -lopencv_contrib: No such file or directory
    /home/user/anaconda3/envs/palenv/compiler_compat/ld: cannot find -lopencv_legacy: No such file or directory
    collect2: error: ld returned 1 exit status
    error: command '/usr/bin/g++' failed with exit code 1
    ```
    **Solution**: In [setup_linux_vo.py](slam_py/install/setup_linux_vo.py), modify line 18 as follows (```opencv4``` instead of ```opencv```):
    ```python
    opencv_libs = subprocess.check_output('pkg-config --libs opencv4'.split())
    ```
<br>

2.
    ```bash
    ImportError: /lib/x86_64-linux-gnu/libp11-kit.so.0: undefined symbol: ffi_type_pointer, version LIBFFI_BASE_7.0
    ```
    **Solution**: Create symbolic link in your Python environment:
    ```bash
    ln -sf /usr/lib/x86_64-linux-gnu/libffi.so.7 ~/anaconda3/envs/palenv/lib/libffi.so.7
    ```
<br>

3.
    ```bash
    In file included from ../../voldor/geometry.h:2,
                     from ../../voldor/geometry.cpp:1:
    ../../voldor/utils.h:15:10: fatal error: opencv2/highgui.hpp: No such file or directory
         15 | #include <opencv2/highgui.hpp>
            |          ^~~~~~~~~~~~~~~~~~~~~
    compilation terminated.
    error: command '/usr/bin/g++' failed with exit code 1
    ```
    **Solution**: Create symbolic link for opencv2:
    ```bash
    sudo ln -s /usr/include/opencv4/opencv2 /usr/include/opencv2
    ```
<br>

4.
    ```bash
    sh: 1: /usr/local/cuda/bin/nvcc: not found
    ```
    **Solution**: Fix nvcc path in [setup_linux_vo.py](slam_py/install/setup_linux_vo.py). In line 15, replace `/usr/local/cuda/bin/nvcc` with the correct path to `nvcc` on your system.
