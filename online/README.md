# Online Pipeline

## Preparation

1. Create a folder named `shared` and copy the shared object files `libgpu-kernels.so` and `pyvoldor_vo.xx.so` from VOLDOR.

    ```bash
    $ cd ~/cl-vae-tracking/online
    $ mkdir shared
    $ cp ../VOLDOR/slam_py/install/libgpu-kernels.so ../VOLDOR/slam_py/install/pyvoldor_vo.*.so shared/
    ```

2. Create a folder ```SAMPLE_NAME```, containing a video file named ```video.mp4``` as your input, along with the a file named ```calibration.yaml``` with the calibration parameters of the video's camera.

3. (Optional) If you want to perform evaluation, also include a file named ```trajectory_bboxes.pkl``` in the folder, containing a dict with key: 'bounding_boxes' and value: list of tuples (x, y, w, h) or None

4. (Optional) In case of real-time feed, create calibration file:
    ```bash
    $ cd calibration
    $ touch CAMERA_calibration.yaml
    ```
    For RealSense camera, we have created [realsense_calib.yaml](./calibration/realsense_calib.yaml)

5. Check the yaml files in the [configs](./configs/) folder for parameterization:

    ```yaml
    video_cfg.yaml

    # for saved video input
    camera:    
        video_path: /path/to/SAMPLE_NAME
    ```

    ```yaml
    realsense_cfg.yaml

    # for real-time feed from connected camera
    camera:
        video_path: null
        name: 'RealSense'
    ```

## Testing:

- For manual selection of tracking area:

    ```sh
    python run_pipeline.py -c ../configs/video_cfg.yaml
    ```

- For evaluation with ground truth:

    ```sh
    python run_evaluation.py -c ../configs/video_cfg.yaml
    ```

**Note**: You may encounter the following error if you are using a very high resolution video input (e.g. 1280 × 2048):
```
malloc(): corrupted top size
Aborted (core dumped)
```
