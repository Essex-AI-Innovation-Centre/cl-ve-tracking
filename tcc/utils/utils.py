import os
import cv2
from tqdm import tqdm
import numpy as np

def make_video_data(video_path, zarr_root, jpegxl_compression=True):
    print('Creating video data...')
    video_data = zarr_root['video_data'] if 'video_data' in zarr_root else zarr_root.create_group('video_data')
    video_files = [filename for filename in sorted(os.listdir(video_path)) if filename.endswith('.mp4')]
    for video_file in tqdm(video_files, leave=False, position=0):
        video_name = video_file.split('.')[0]
        cap = cv2.VideoCapture(os.path.join(video_path, video_file))
        N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        kwargs = dict(
            shape=(N, H, W, 3),
            chunks=(1, H, W, 3), 
            dtype='u1',
            overwrite=True
            )

        video_data.create_dataset(video_name, **kwargs)
        
        for i in tqdm(range(N), leave=False, position=1):
            ret, frame = cap.read()
            video_data[video_name][i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cap.release()
    print('Done.')

def bbox_overlap_area(bbox1, bbox2):
    """Compute intersection area between two axis-aligned bboxes"""
    x1_min, y1_min = bbox1[0]
    x1_max, y1_max = bbox1[2]
    x2_min, y2_min = bbox2[0]
    x2_max, y2_max = bbox2[2]

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    return inter_area

def compute_iou(b1, b2):
    # b1, b2: (4,) tensors: [x_min, y_min, x_max, y_max]
    xA = max(b1[0], b2[0])
    yA = max(b1[1], b2[1])
    xB = min(b1[2], b2[2])
    yB = min(b1[3], b2[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0 else 0.0

def save_tracked_video(tracked_boxes, frames_color, output_path, fps=5):
    T, H, W, _ = frames_color.shape
    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (W, H)
    )

    for t in range(T):
        frame = frames_color[t].copy()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for quad_track in tracked_boxes:
            quad = quad_track[t].astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame_bgr, [quad], isClosed=True, color=(0, 255, 0), thickness=2)

        out.write(frame_bgr)

    out.release()
    print(f"Saved video to {output_path}")


def read_video_chunk(cap, start_idx, end_idx, rgb=True):
    frames = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
    for idx in range(start_idx, end_idx):
        ret, frame = cap.read()
        if not ret:
            break
        if rgb:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        else:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return np.array(frames)

