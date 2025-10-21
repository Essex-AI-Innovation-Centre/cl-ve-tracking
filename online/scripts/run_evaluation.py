import os
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
        
        self.logger.init_metrics(['search_error_2d', 'prediction_error_2d', 'iou'])

        lost = False
        step = 0
        while True:
            if len(self.voldor.frames) > 0 and not lost:
                keyframe = self.voldor.fid_cur - self.voldor.n_frames_registered
                depth_map = self.voldor.frames[keyframe].get_scaled_depth()
                T_kf = np.linalg.inv(self.voldor.frames[keyframe].Tcw)
                if len(self.tracking_handler.tracked_areas) == 0:
                    gt_bbox_first = self.camera.gt_boxes[gt_idx] # TODO: for multiple areas
                    if gt_bbox_first:
                        gt_bbox_first = list(map(lambda x : max(0, x), gt_bbox_first))
                        idx = len(self.tracking_handler.tracked_areas)
                        self.tracking_handler.register_ground_truth_area(gt_bbox_first, frame, idx, flows, depth_map, T_kf)

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

            # find frame index from camera capture for ground truth bounding box
            frame_count = int(self.camera.cap.get(cv2.CAP_PROP_POS_FRAMES))
            register_dif = (self.voldor.vo_win_size - self.voldor.n_frames_registered)
            gt_idx = frame_count - 1 - register_dif - 1

            for area in self.tracking_handler.tracked_areas.values():
                if self.tracking_handler.matching_test_mode:
                    area.use_flow = False

                if area.use_flow and not self.voldor.test_mode:
                    self.tracking_handler.OF_bbox_step(frame, area, flows)
                else:
                    self.tracking_handler.LM_bbox_step(frame, area, flows, depth_map, T_kf, self.voldor.test_mode)

                # evaluation
                gt_bbox = self.camera.gt_boxes[gt_idx] # TODO: for multiple areas
                if gt_bbox:
                    pred_bbox = (area.x, area.y, area.w, area.h)
                    search_bbox = area.search
                    if (search_bbox or area.use_flow):
                        if search_bbox:
                            search_error_2d = get_l2_norm(get_bbox_centroid(gt_bbox), get_bbox_centroid(search_bbox))
                            self.logger.add_metric('search_error_2d', search_error_2d)
                        error_2d = get_l2_norm(get_bbox_centroid(gt_bbox), get_bbox_centroid(pred_bbox))
                        self.logger.add_metric('prediction_error_2d', error_2d)
                        iou = get_iou(gt_bbox, pred_bbox)
                        self.logger.add_metric('iou', iou)

            for _ in range(self.voldor.n_frames_registered):
                frame_batch.pop(0)
                vo_of_batch.pop(0)
                hr_of_batch.pop(0)         
            
            display_frame = self.tracking_handler.visualize_tracking_areas(frame, self.voldor.test_mode)
            gt_bbox = self.camera.gt_boxes[gt_idx] # TODO: for multiple areas
            if gt_bbox:
                gt_bbox = list(map(lambda x : max(0, x), gt_bbox))
                display_frame = self.tracking_handler.visualize_ground_truth(display_frame, gt_bbox)

            self.logger.add_frame(display_frame)

            print(f"-----\nStep: {step}")
            self.logger.report_metrics(frame.shape[0], frame.shape[1])


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
    matching_time = np.mean(demo_pipeline.tracking_handler.matching_times)
    if args.save:
        demo_pipeline.logger.save_video(config['camera']['video_path'].split('/')[-1], matching_time)
        demo_pipeline.logger.save_config(config)

    demo_pipeline.logger.report_metrics(demo_pipeline.camera.res_height, demo_pipeline.camera.res_width)
