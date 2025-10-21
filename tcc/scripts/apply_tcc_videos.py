import os
import argparse
import pathlib
import sys
import shutil
import yaml
import zarr
import cv2
from tqdm import tqdm
import logging
import torch
script_path = pathlib.Path.absolute(pathlib.Path(__file__))
sys.path.append(str(script_path.parents[1]))
from utils.utils import make_video_data, read_video_chunk
from utils.bboxes import *

def main(args):
    config = yaml.safe_load(open(args.config))
    tracking_window = config['tracking_window']
    sample_overlap = config['sample_overlap']
    video_path = config['video_path']
    num_viz_per_video = config['num_viz_per_video']

    feature_params = config['feature_params']
    lk_params = config['lk_params']
    lk_params['criteria'] = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 
                             config['lk_params']['criteria']['max_iter'], 
                             config['lk_params']['criteria']['epsilon'])
    zarr_path = config['zarr_path']

    if tracking_window is not None:
        assert sample_overlap < tracking_window

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store, overwrite=False)
    
    if not args.save_vid:
        print('Skipping video data conversion to zarr.')
    else:
        make_video_data(video_path, root)


    print('Creating tracking data...')
    tracking_data = root['tracking_data'] if 'tracking_data' in root else root.create_group('tracking_data')
    bbox_data = root['bbox_data'] if 'bbox_data' in root else root.create_group('bbox_data')

    video_files = [filename for filename in sorted(os.listdir(video_path)) if filename.endswith('.mp4')]
    num_tracked_points = {}
    for video_file in tqdm(video_files, leave=False, position=0):
        video_name = video_file.split('.')[0]
        num_tracked_points[video_name] = []
        
        viz_path = os.path.join(config['logs_path'], 'video_viz', f"viz_{video_name}")
        
        # Remove previous viz folder if it exists and then create a new one.
        try:
            shutil.rmtree(viz_path)
        except FileNotFoundError:
            pass
        os.makedirs(viz_path, exist_ok=True)

        # Read video and save frames in grayscale.
        cap = cv2.VideoCapture(os.path.join(video_path, video_file))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        
        split_idxs = []
        curr_start_idx = 0

        if 'video_splits' in config:
            # Split the video into the specified number of samples.
            num_splits = config['video_splits']
            window = frame_count//num_splits
            print(f"Splitting video {video_file} into {num_splits} samples of size {window}.")
            sample_overlap = 0
        else:
            sample_overlap = sample_overlap if tracking_window is not None else 0
            window = tracking_window if tracking_window is not None else frame_count

        while True:
            curr_end_idx = curr_start_idx + window
            if curr_end_idx > frame_count:
                #split_idxs.append((curr_start_idx, frame_count))
                break
            split_idxs.append((curr_start_idx, curr_end_idx))
            curr_start_idx += window-sample_overlap

        # Define when to save a visualization of the tracked points.
        if num_viz_per_video is not None and num_viz_per_video > 0:
            viz_every = len(split_idxs)//num_viz_per_video
            if viz_every == 0:
                logging.info(f"Video {video_file} has less than {num_viz_per_video} samples. Visualizing all samples.")
                viz_every = 1
        else:
            viz_every = None

        tracking_samples = tracking_data.create_group(video_name, overwrite=True)
        bbox_data.create_group(video_name, overwrite=True)
        for sample_idx, (start_idx, end_idx) in enumerate(pbar_1 := tqdm(split_idxs, leave=False, position=1)):
            # all_frames = all_frames_grayscale[start_idx:end_idx]
            cap = cv2.VideoCapture(os.path.join(video_path, video_file))
            frames_gray = read_video_chunk(cap, start_idx, end_idx, rgb=False)
            cap.release()

            try:
                pbar_1.set_description('Forward-backward tracking points')
                consistent_trajectories = get_consistent_point_trajectories(
                    frames_gray, 
                    feature_params, 
                    lk_params, 
                    consistency_threshold=config['consistency_threshold']
                )
                if consistent_trajectories is None:
                    logging.info(f"No consistent trajectories in video {video_file} at sample {sample_idx} with start_idx {start_idx} and end_idx {end_idx}")
                    continue
                num_tracked_points[video_name].append(consistent_trajectories.shape[1])
            except ValueError as e:
                logging.info(f"Error in video {video_file} at sample {sample_idx} with start_idx {start_idx} and end_idx {end_idx}")
                num_tracked_points[video_name].append(0)
                continue
            
            if consistent_trajectories.shape[1] == 0:
                logging.info(f"No points tracked for video {video_file} sample {sample_idx}. Skipping.")
                continue
            tracking_samples.create_dataset(f'sample_{sample_idx}', data=consistent_trajectories, chunks=consistent_trajectories.shape, dtype='f4')
            tracking_samples[f'sample_{sample_idx}'].attrs['start_idx'] = start_idx
            tracking_samples[f'sample_{sample_idx}'].attrs['end_idx'] = end_idx

            # Generate valid bounding boxes from the tracked points.
            pbar_1.set_description('Generating valid bounding boxes')


            tracking_data_sample = root['tracking_data'][video_name][f'sample_{sample_idx}']
            tracking_data_arr = torch.tensor(tracking_data_sample[:])

            # Get initial bounding boxes from the first frame points by checking the H,W distances between them.
            points = tracking_data_arr[0]
            bboxes = points_to_bboxes(points, padding_pixels=0)
            valid_bboxes = get_valid_bboxes(bboxes, video_name, sample_idx, config)
            
            if valid_bboxes is None:
                logging.info(f"No valid bounding boxes found for video {video_file} sample {sample_idx}. Skipping.")
                continue
            
            bbox_corners = box_to_corners(valid_bboxes)  # shape (B, 4, 2)

            tracked_boxes = track_bboxes_optical_flow(
                frames_gray=frames_gray,  # or all_frames_grayscale[start_idx:end_idx]
                bbox_corners=bbox_corners,
                lk_params=lk_params
            )
            print(f" Tracked {len(tracked_boxes)} bounding boxes with 4-point trajectories")

            # output_path = os.path.join(viz_path, f'sample_{sample_idx}_tracked_boxes.mp4')
            # save_tracked_video(tracked_boxes, all_frames_color[start_idx:end_idx], output_path)
            del frames_gray
            torch.cuda.empty_cache()
            cap = cv2.VideoCapture(os.path.join(video_path, video_file))
            frames_rgb = read_video_chunk(cap, start_idx, end_idx)
            cap.release()
            if len(tracked_boxes) == 0:
                logging.info(f"No tracked boxes found for video {video_file} sample {sample_idx}. Skipping.")
                continue

            vis_and_save_bboxes(video_name, sample_idx, config, frames_rgb, tracked_boxes, args.save_vid)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-sv','--save_vid', action='store_true', help='Create also videos')
    parser.add_argument('-c', '--config', type=str, help='config file path', 
                        default=os.path.join(os.path.abspath(str(script_path.parents[1])), 'configs', 'config.yaml'))

    args = parser.parse_args()

    main(args)