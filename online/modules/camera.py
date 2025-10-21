import cv2
import yaml
import os
import numpy as np
import time
import pickle
from online.utils.camera import find_camera_index, extract_calibration_matrix

class CameraSetup:
    def __init__(self, config, rescale_factor=None):
        self.cfg = config
        video_path = self.cfg['video_path']
        self.eval_mode = self.cfg['eval_mode']
        self.res_width = self.cfg['resolution']['width']
        self.res_height = self.cfg['resolution']['height']
        self.rescale = rescale_factor if rescale_factor is not None else 1.0

        if video_path is None:
            self._setup_camera_from_feed(self.cfg)
        else:
            self._setup_camera_from_video(video_path)
            if self.eval_mode:
                self._setup_evaluation(video_path)

        if not self.cap.isOpened():
            print(f"Error opening video stream!")
            return
        
        self._setup_intrinsics()

    def _set_camera_resolution(self):
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.res_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.res_height)
    
    def _sleep(self, duration):
        open_cap = time.time()
        while True:
            _ = self.cap.read()
            if time.time() - open_cap > duration:
                break

    def _setup_camera_from_feed(self, cfg):
        calib_path = os.path.join('../calibration', f"{cfg['name'].lower()}_calib.yaml")
        with open(calib_path) as f:
            calib_params = yaml.load(f, Loader=yaml.FullLoader)
            
        if cfg['name'].lower() == 'realsense':
            self.input_source = 69
        else:
            self.input_source = find_camera_index(cfg['name'], calib_params['source'])
            self.cap = cv2.VideoCapture(self.input_source)
            self._set_camera_resolution()
        
        self._sleep(3)
            
        self.fx, self.fy, self.cx, self.cy, self.bf = [calib_params['calibration'][key] for key in ['fx', 'fy', 'cx', 'cy', 'bf']]

    def _setup_camera_from_video(self, video_path):
        self.input_source = os.path.join(video_path, 'video.mp4')
        calib_path = os.path.join(video_path, 'calibration.yaml')
        self.fx, self.fy, self.cx, self.cy, self.bf = extract_calibration_matrix(calib_path)
        
        self.cap = cv2.VideoCapture(self.input_source)
        self._set_camera_resolution()

    def _setup_evaluation(self, video_path):
        self.gt_path = os.path.join(video_path, 'trajectory_bboxes.pkl')
        with open(self.gt_path, 'rb') as f:
            self.gt_boxes = pickle.load(f)['bounding_boxes']

    def _setup_intrinsics(self):
        self.basefocal = (self.fx + self.fy) * 0.25 * self.rescale if self.bf == 'auto' or self.bf<=0 else self.bf * self.rescale
        self.K = np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float32)
        self.K_inv = np.linalg.inv(self.K)
        self.fx *= self.rescale
        self.fy *= self.rescale
        self.cx *= self.rescale
        self.cy *= self.rescale
