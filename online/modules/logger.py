import time
import cv2
import os
import numpy as np
import yaml


class Logger():
    def __init__(self, model=None):
        self.avg_inference = None
        self.frames = []
        self.model = model

    def init_metrics(self, metric_names):
        self.metrics = {}
        for metric_name in metric_names:
            self.metrics[metric_name] = []

    def add_frame(self, frame):
        self.frames.append(frame)

    def add_metric(self, metric_name, metric):
        self.metrics[metric_name].append(metric)

    def report_metrics(self, frame_height, frame_width):
        num_diag_pixels = np.sqrt(frame_height**2 + frame_width**2)
        mean_search_error_2d = np.mean(self.metrics['search_error_2d'])
        mean_search_error_prc = mean_search_error_2d / num_diag_pixels * 100
        mean_prediction_error_2d = np.mean(self.metrics['prediction_error_2d'])
        mean_prediction_error_prc = mean_prediction_error_2d / num_diag_pixels * 100
        mean_iou = np.mean(self.metrics['iou'])

        results_messages = [f"Mean 2D localization error: {mean_search_error_2d:.2f} ({mean_search_error_prc:.1f}%)",
                            f"Mean 2D prediction error: {mean_prediction_error_2d:.2f} ({mean_prediction_error_prc:.1f}%)",
                            f"Mean IoU: {mean_iou:.2f}"]
        for message in results_messages:
            print('\t\t', message, sep='')
        print("-----------------------------------------------\n")

        self.metrics_dict = {'search_error_2d': mean_search_error_2d, 'prediction_error_2d': mean_prediction_error_2d, 'iou': mean_iou,
                             'search_error_2d_prc': mean_search_error_prc, 'prediction_error_2d_prc': mean_prediction_error_prc}
        for k, v in self.metrics_dict.items():
            self.metrics_dict[k] = float(v)
        return self.metrics_dict

    def save_video(self, video_name=None, matching_time=None):
        log_dir = '/'.join(os.path.abspath(__file__).split('/')[:-2]+['logs'])
        videos_dir = os.path.join(log_dir, 'saved_videos')
        print(f"VIDEOS DIR: {videos_dir}")
        os.makedirs(videos_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        mp4_name = video_name if video_name else time.strftime("%Y-%m-%d_%H-%M-%S")
        mp4_path = os.path.join(videos_dir, f'{mp4_name}.mp4')
        frame_height, frame_width = self.frames[0].shape[:2]
        out = cv2.VideoWriter(mp4_path, fourcc, 7.4, (frame_width, frame_height))
        for frame in self.frames:
            out.write(frame)
        out.release()

        if matching_time is not None:
            self.metrics_dict['matching_time'] = float(matching_time)
            metrics_dir = os.path.join(log_dir, 'saved_metrics')
            os.makedirs(metrics_dir, exist_ok=True)
            with open(os.path.join(metrics_dir, video_name+'.yaml'), "w") as yaml_file:
                yaml.dump(self.metrics_dict, yaml_file)

    def save_config(self, config):
        log_dir = '/'.join(os.path.abspath(__file__).split('/')[:-2]+['logs'])
        configs_dir = os.path.join(log_dir, 'saved_configs')
        os.makedirs(configs_dir, exist_ok=True)

        video_name = config['camera']['video_path'].split('/')[-1]
        with open(os.path.join(configs_dir, video_name+'.yaml'), "w") as yaml_file:
            yaml.dump(config, yaml_file)
