import cv2
import yaml
import numpy as np
import argparse

import sys
import pathlib
path = str(pathlib.Path.absolute(pathlib.Path((__file__))).parent.parent.parent)
sys.path.append(path)

from online.utils.voldor import rescale_flow
from online.modules.logger import Logger
from online.modules.camera import CameraSetup
from online.modules.optical_flow import FlowModelSetup
from online.modules.voldor import VoldorSetup
from online.modules.tracking import TrackingHandler
from online.utils.tracking import *


class DemoPipeline:
    def __init__(self, config):
        self.logger = Logger(config['feature_matching']['feature_extractor'])

        # Set camera resolution
        video_path = os.path.join(config['camera']['video_path'], 'video.mp4')
        cap = cv2.VideoCapture(video_path)
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        config['camera']['resolution'] = {'width': W, 'height': H}

        self.camera = CameraSetup(config['camera'], config['VOLDOR']['rescale'])
        self.flow_model = FlowModelSetup(config['flow'])
        self.voldor = VoldorSetup(config['VOLDOR'], self.camera)       
        self.tracking_handler = TrackingHandler(config['feature_matching'], self.camera, rescale_factor=self.voldor.rescale)

    def run_loop(self):
        ret, prev_frame = self.camera.cap.read()
        frame_batch = [prev_frame.copy()] # for display only
        self.logger.add_frame(prev_frame.copy())

        prev_frame = prev_frame.astype(np.float32)
                
        vo_of_batch = [] # voldor_step input
        hr_of_batch = [] # OF_step input
        
        lost = False
        step = 0
        while True:
            if len(self.voldor.frames) > 0 and not lost and len(self.tracking_handler.tracked_areas) == 0:
                key = cv2.waitKey(1)
                if key & 0xFF == ord('r'): # press R to register area
                    keyframe = self.voldor.fid_cur - self.voldor.n_frames_registered
                    depth_map = self.voldor.frames[keyframe].get_scaled_depth()
                    T_kf = np.linalg.inv(self.voldor.frames[keyframe].Tcw)
                    idx = len(self.tracking_handler.tracked_areas)
                    self.tracking_handler.register_tracking_area(frame, idx, flows, depth_map, T_kf)
                elif key & 0xFF == ord('q'): # press Q to quit pipeline
                    break

            new_np_frames = [prev_frame]
            for _ in range(self.voldor.n_frames_registered):
                ret, frame = self.camera.cap.read()
                if not ret:
                    self.camera.cap.release()
                    return step
                frame_batch.append(frame.copy())

                frame = frame.astype(np.float32)
                new_np_frames.append(frame)
                prev_frame = frame.copy()

            np_flows = self.flow_model.flow_step([(new_np_frames[i], new_np_frames[i+1]) for i in range(len(new_np_frames)-1)])
            for i in range(len(np_flows)):
                hr_of_batch.append(np_flows[i]) # full-res for OF_bbox_step
                flow = rescale_flow(np_flows[i], self.voldor.rescale) # downscaled for voldor_step
                vo_of_batch.append(flow)
            
            lost = self.voldor.voldor_step(vo_of_batch)

            # For handling 'Tracking lost' case, skip a frame
            if lost:
                frame_batch.pop(0)
                vo_of_batch.pop(0)
                hr_of_batch.pop(0)
                if len(self.voldor.frames) == 0:
                    print('Handling the first step...')
                continue
            
            step += 1
            T = np.linalg.inv(self.voldor.Twc_cur)
            for area in self.tracking_handler.tracked_areas.values():
                area.track(T)

            frame = frame_batch[self.voldor.n_frames_registered].copy()
            flows = hr_of_batch[:self.voldor.n_frames_registered]

            for area in self.tracking_handler.tracked_areas.values():
                if self.tracking_handler.matching_test_mode:
                    area.use_flow = False

                if area.use_flow and not self.voldor.test_mode:
                    self.tracking_handler.OF_bbox_step(frame, area, flows)
                else:
                    self.tracking_handler.LM_bbox_step(frame, area, flows, depth_map, T_kf, self.voldor.test_mode)

            for _ in range(self.voldor.n_frames_registered):
                frame_batch.pop(0)
                vo_of_batch.pop(0)
                hr_of_batch.pop(0)         
            
            display_frame = self.tracking_handler.visualize_tracking_areas(frame, self.voldor.test_mode)
            self.logger.add_frame(display_frame)

            print(f"-----\nStep: {step}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, help='Provide path for configuration file')
    parser.add_argument('-sv', '--save', action='store_true', help='Save video')
        
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader) 

    demo_pipeline = DemoPipeline(config)
    try:    
        demo_pipeline.run_loop()
    except KeyboardInterrupt:
        print('Exception occured: KeyboardInterrupt')
    
    demo_pipeline.camera.cap.release()
    if args.save:
        demo_pipeline.logger.save_video(config['camera']['video_path'].split('/')[-1])
