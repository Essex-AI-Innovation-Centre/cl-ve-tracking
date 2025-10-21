import os
import pickle
import logging
import torch
import cv2
import numpy as np
import torchvision
from utils.utils import compute_iou

def get_consistent_point_trajectories(video_grayscale, feature_params, lk_params, consistency_threshold=20):
    old_frame = video_grayscale[0]
    start_points = cv2.goodFeaturesToTrack(old_frame, mask=None, **feature_params)
    #print(start_points.squeeze().shape)
    point_ids = np.arange(len(start_points))
    old_points = start_points
    point_trajectories = np.zeros((len(video_grayscale)*2-1, len(start_points), 2))
    point_trajectories[0, :] = start_points.squeeze()
    consistency_flag = np.ones(len(start_points))
    for i, frame in enumerate(np.concatenate([video_grayscale[1:], video_grayscale[::-1][1:]])):
        new_points, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame, old_points.reshape(-1, 1, 2), None, **lk_params)
        if new_points is None:
            logging.debug(f"Lost all points in video. Skipping.")
            return None
        # Select good points
        good_new_points = new_points[st == 1]
        #good_old_points = old_points[st == 1]
        point_ids = point_ids[st.squeeze() == 1].squeeze()
        if len(point_ids.shape) == 0:
            point_ids = np.array([point_ids])
        # get all missing point_ids
        missing_point_ids = np.setdiff1d(np.arange(len(start_points)), point_ids)
        consistency_flag[missing_point_ids] = 0
        point_trajectories[i+1, point_ids] = good_new_points.squeeze()
        old_frame = frame.copy()
        old_points = good_new_points.reshape(-1, 1, 2)
    
    trajectories_fw = point_trajectories[:len(video_grayscale)]
    trajectories_bw = point_trajectories[len(video_grayscale):][::-1]

    for fw_points, bw_points in zip(trajectories_fw[:-1], trajectories_bw):
        for i, (fw_point, bw_point) in enumerate(zip(fw_points, bw_points)):
            dist = np.linalg.norm(fw_point-bw_point)
            if dist > consistency_threshold:
                consistency_flag[i] = 0

    #consistent_trajectories = point_trajectories[:, consistency_flag == 1]

    # get final trajectories by averaging the forward and backward trajectories
    consistent_trajectories_fw = trajectories_fw[:, consistency_flag == 1]    
    consistent_trajectories_bw = trajectories_bw[:, consistency_flag == 1]
    consistent_trajectories_avg = consistent_trajectories_fw
    consistent_trajectories_avg[:-1] = (consistent_trajectories_avg[:-1]+consistent_trajectories_bw)/2
    

    #return consistent_trajectories.astype(np.float32)
    return consistent_trajectories_avg.astype(np.float32)

def get_valid_bboxes(bboxes, video_name, sample_idx, config):

    min_dim = config['bboxes']['min_init_dim'] + config['bboxes']['padding_pixels']*2
    max_dim = config['bboxes']['max_init_dim'] + config['bboxes']['padding_pixels']*2

    widths = bboxes[:, :, 2] - bboxes[:, :, 0]  # x_max - x_min
    heights = bboxes[:, :, 3] - bboxes[:, :, 1]  # y_max - y_min

    valid_size_mask = (widths > min_dim) & (heights > min_dim) & \
                    (widths < max_dim) & (heights < max_dim)

    valid_size_mask = torch.triu(valid_size_mask, diagonal=1)
    valid_indices = torch.nonzero(valid_size_mask, as_tuple=False)  # shape (M, 2)

    selected_pairs = []

    if bboxes.shape[0] == 0 or bboxes.shape[1] == 0:
        logging.info(f"No valid bounding boxes found for video {video_name} sample {sample_idx}. Skipping.")
        return None

    bboxes_flat = bboxes  # shape (N, N, 4)
    candidate_bboxes = [bboxes_flat[i, j] for i, j in valid_indices]
    candidate_indices = [(i.item(), j.item()) for i, j in valid_indices]

    keep_flags = torch.ones(len(candidate_bboxes), dtype=torch.bool)

    for i in range(len(candidate_bboxes)):
        if not keep_flags[i]:
            continue  # Already suppressed

        for j in range(i + 1, len(candidate_bboxes)):
            if not keep_flags[j]:
                continue

            iou = compute_iou(candidate_bboxes[i], candidate_bboxes[j])
            if iou > 0:  # Overlap threshold
                # Suppress the one with smaller area (or just j to keep it simple)
                area_i = (candidate_bboxes[i][2] - candidate_bboxes[i][0]) * (candidate_bboxes[i][3] - candidate_bboxes[i][1])
                area_j = (candidate_bboxes[j][2] - candidate_bboxes[j][0]) * (candidate_bboxes[j][3] - candidate_bboxes[j][1])
                if area_i >= area_j:
                    keep_flags[j] = False
                else:
                    keep_flags[i] = False
                    break  # i is invalid now, stop comparing it



    selected_pairs = [candidate_indices[i] for i in range(len(candidate_indices)) if keep_flags[i]]

    if len(selected_pairs) == 0:
        logging.info(f"No valid bounding boxes found for video {video_name} sample {sample_idx}. Skipping.")
        return None

    final_bboxes = torch.stack([bboxes[i, j] for (i, j) in selected_pairs])

    return final_bboxes

def points_to_bboxes(points, padding_pixels=0):
    # points: (N, 2) tensor
    # Get all pairs of points and create corresponding bounding boxes
    num_points = points.shape[0]
    point_pairs = torch.cat((
        points.unsqueeze(1).expand(-1, num_points, -1), 
        points.unsqueeze(0).expand(num_points, -1, -1)
        ), dim=-1
    )
    point_pairs_x_min = torch.min(point_pairs[:, :, 0], point_pairs[:, :, 2]) + padding_pixels
    point_pairs_y_min = torch.min(point_pairs[:, :, 1], point_pairs[:, :, 3]) + padding_pixels
    point_pairs_x_max = torch.max(point_pairs[:, :, 0], point_pairs[:, :, 2]) - padding_pixels
    point_pairs_y_max = torch.max(point_pairs[:, :, 1], point_pairs[:, :, 3]) - padding_pixels

    point_pairs_bboxes = torch.stack((point_pairs_x_min, point_pairs_y_min, point_pairs_x_max, point_pairs_y_max), dim=-1)
    return point_pairs_bboxes

def box_to_corners(boxes):
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    x_max = boxes[:, 2]
    y_max = boxes[:, 3]

    top_left = torch.stack([x_min, y_min], dim=-1)
    top_right = torch.stack([x_max, y_min], dim=-1)
    bottom_right = torch.stack([x_max, y_max], dim=-1)
    bottom_left = torch.stack([x_min, y_max], dim=-1)

    corners = torch.stack([top_left, top_right, bottom_right, bottom_left], dim=1)  # (B, 4, 2)
    return corners

def track_bboxes_optical_flow(frames_gray, bbox_corners, lk_params):
    T = len(frames_gray)
    B = bbox_corners.shape[0]
    tracks = []

    for b in range(B):
        corners = bbox_corners[b].cpu().numpy().astype(np.float32).reshape(-1, 1, 2)
        track = np.zeros((T, 4, 2), dtype=np.float32)
        track[0] = corners.squeeze()
        prev_frame = frames_gray[0]
        prev_pts = corners

        valid = True
        for t in range(1, T):
            next_frame = frames_gray[t]
            next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_frame, next_frame, prev_pts, None, **lk_params)
            if next_pts is None or np.any(st == 0):
                valid = False
                break
            track[t] = next_pts.squeeze()
            prev_pts = next_pts
            prev_frame = next_frame

        if valid:
            tracks.append(track)

    return tracks  # list of (T, 4, 2)

def get_quad_size(quad):
    w1 = np.linalg.norm(quad[0] - quad[1])
    w2 = np.linalg.norm(quad[2] - quad[3])
    w = int(round((w1 + w2) / 2))

    h1 = np.linalg.norm(quad[0] - quad[3])
    h2 = np.linalg.norm(quad[1] - quad[2])
    h = int(round((h1 + h2) / 2))

    return max(w, 1), max(h, 1)  # avoid zero-size patches

def extract_normalized_patch(frame_rgb, quad_pts, output_size=(32, 32)):
    quad_pts = np.array(quad_pts, dtype=np.float32)
    w, h = output_size

    dst_pts = np.array([
        [0, 0],         # top-left
        [w - 1, 0],     # top-right
        [w - 1, h - 1], # bottom-right
        [0, h - 1]      # bottom-left
    ], dtype=np.float32)

    H = cv2.getPerspectiveTransform(quad_pts, dst_pts)
    patch = cv2.warpPerspective(frame_rgb, H, (w, h), flags=cv2.INTER_LINEAR)

    return patch

def vis_and_save_bboxes(video_name, sample_idx, config, video_frames, tracked_boxes, save_videos=True):

    T, H, W, _ = video_frames.shape
    viz_path = os.path.join(config['logs_path'], 'video_viz', f"viz_{video_name}")
    os.makedirs(viz_path, exist_ok=True)
    
    filename_all = os.path.join(viz_path, f'sample_{sample_idx}_all.mp4')
    filename_grid = os.path.join(viz_path, f'sample_{sample_idx}_bboxes.mp4')
    patch_pkl_data = {}

    

        # --- Create full-frame annotated video ---
    out = cv2.VideoWriter(filename_all, cv2.VideoWriter_fourcc(*'mp4v'), 5, (W, H))
    all_cropped_images = []

    patch_size = config.get('patch_size', 32)  # or whatever you want
    dst_pts = np.array([
        [0, 0],
        [patch_size - 1, 0],
        [patch_size - 1, patch_size - 1],
        [0, patch_size - 1]
    ], dtype=np.float32)

    for t in range(T):
        original_frame = video_frames[t].copy()  # (H, W, 3), RGB
        frame = original_frame.copy()  
        frame_tensor = torch.tensor(frame).permute(2, 0, 1).contiguous()
        annotated = frame_tensor.clone()
        cropped_images = []

        for idx, quad_track in enumerate(tracked_boxes):
            quad = quad_track[t]  # shape (4, 2)

            # Draw polygon on the frame
            quad_int = quad.astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [quad_int], isClosed=True, color=(0, 255, 0), thickness=2)

            # --- NEW: Warp perspective to get aligned patch ---
            src_pts = quad.astype(np.float32)
            width = int(np.linalg.norm(src_pts[0] - src_pts[1]))
            height = int(np.linalg.norm(src_pts[0] - src_pts[3]))

            dst_pts = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)

            H_mat = cv2.getPerspectiveTransform(src_pts, dst_pts)
            patch = cv2.warpPerspective(original_frame, H_mat, (width, height))  # dynamic size

            # Convert to tensor format like before
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).contiguous()
            cropped_images.append(patch_tensor)

            patch_pkl_data.setdefault(idx, []).append(patch_tensor.permute(1, 2, 0).numpy().astype(np.uint8))

        all_cropped_images.append(cropped_images)

        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Optionally save key frames
        if t in [0, T // 2, T - 1]:
            if save_videos:
                cv2.imwrite(os.path.join(viz_path, f'sample_{sample_idx}_frame_{t}.png'), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()

    # --- Grid Video of Cropped Patches ---
    num_boxes = len(all_cropped_images[0])
    max_h = max([img.shape[1] for imgs in all_cropped_images for img in imgs])
    max_w = max([img.shape[2] for imgs in all_cropped_images for img in imgs])
    grid_size = int(np.ceil(np.sqrt(num_boxes)))
    grid_height = grid_size * max_h
    grid_width = grid_size * max_w


    filename_grid = os.path.join(viz_path, f'sample_{sample_idx}_bboxes.mp4')
    out_grid = cv2.VideoWriter(filename_grid, cv2.VideoWriter_fourcc(*'mp4v'), 5, (grid_width, grid_height))

    for t, cropped_images in enumerate(all_cropped_images):
        imgs = torch.zeros((num_boxes, 3, max_h, max_w), dtype=torch.uint8)
        for j, img in enumerate(cropped_images):
            h, w = img.shape[1:]
            imgs[j, :, :h, :w] = img

        grid = torchvision.utils.make_grid(imgs, nrow=grid_size, padding=0)
        write_img = grid.permute(1, 2, 0).numpy().astype(np.uint8)
        out_grid.write(cv2.cvtColor(write_img, cv2.COLOR_RGB2BGR))

        if t % 5 == 0:
            patches_dir = os.path.join(viz_path, f'sample_{sample_idx}_patches')
            concat_dir = os.path.join(viz_path, f'sample_{sample_idx}_concat_patches')
            os.makedirs(patches_dir, exist_ok=True)
            os.makedirs(concat_dir, exist_ok=True)

            for j, img in enumerate(cropped_images):
                img_bgr = cv2.cvtColor(img.permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR)
                if save_videos:
                    cv2.imwrite(os.path.join(patches_dir, f'patch_{j}_frame_{t}.png'), img_bgr)
            if save_videos:
                cv2.imwrite(os.path.join(concat_dir, f'bboxes_frame_{t}.png'), cv2.cvtColor(write_img, cv2.COLOR_RGB2BGR))

    out_grid.release()

    # --- Save .pkl ---
    if config.get('pkl_save_path', None):
        pkl_path = os.path.join(config['pkl_save_path'])
        os.makedirs(pkl_path, exist_ok=True)
        pkl_file = os.path.join(pkl_path, f'{video_name}_window_{sample_idx}_patches.pkl')
        with open(pkl_file, 'wb') as f:
            pickle.dump(patch_pkl_data, f)
    else:
        logging.info('No pkl_save_path specified. Skipping pkl save.')

