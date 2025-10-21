import os
import cv2
import random

def create_and_store_dataset(video_dir, output_dir, N, min_frame_distance, num_bboxes, min_dim, max_dim):
    """
    This function extracts frames from videos, generates patches, and stores them in the specified output directory.

    Args:
        video_dir (str): Path to the directory containing video files.
        output_dir (str): Path to the directory where the dataset will be stored.
        N (int): Number of frames to extract per video.
        min_frame_distance (int): Minimum distance between chosen frames.
        num_bboxes (int): Number of bounding boxes to generate per frame.
        min_dim (tuple): Minimum dimensions for bounding boxes.
        max_dim (tuple): Maximum dimensions for bounding boxes.
        gray_scale (bool): Whether to create grayscale patches.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)

                relative_path = os.path.relpath(root, video_dir)
                video_output_folder = os.path.join(output_dir, relative_path)
                os.makedirs(video_output_folder, exist_ok=True)
                cap = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                print(f"Processing {video_path}, Total frames: {total_frames}")

                max_frame_index = max(0, total_frames - min_frame_distance * (N - 1))
                if max_frame_index < 0:
                    print(f"Skipping {video_path}, not enough frames for sampling.")
                    continue

                frame_indices = sorted(random.sample(range(0, max_frame_index, min_frame_distance), N))

                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        frame_filename = f'frame_{idx}.png'
                        cv2.imwrite(os.path.join(video_output_folder, frame_filename), frame)

                cap.release()

                patch_folder = os.path.join(video_output_folder, 'patches')
                os.makedirs(patch_folder, exist_ok=True)

                for frame_file in os.listdir(video_output_folder):
                    if not frame_file.endswith('.png'):
                        continue
                    frame_path = os.path.join(video_output_folder, frame_file)
                    frame = cv2.imread(frame_path)

                    bboxes = create_random_bboxes(frame, num_bboxes, min_dim, max_dim)

                    for i, bbox in enumerate(bboxes):
                        patch_path = os.path.join(patch_folder, f'{frame_file}_patch_{i}.png')
                        cv2.imwrite(patch_path, bbox)

                print(f"Patches generated for {video_path} in {patch_folder}")
                
def create_random_bboxes(frame, num_bboxes, min_dim, max_dim):
    """
    Creates random bounding boxes within the frame.

    Args:
        frame (np.array): Input image/frame.
        num_bboxes (int): Number of bounding boxes to create.
        min_dim (int): Minimum dimensions for bounding boxes.
        max_dim (int): Maximum dimensions for bounding boxes.

    Returns:
        list: List of bounding boxes (cropped patches).
    """
    img_h, img_w = frame.shape[:2]
    bboxes = []

    for _ in range(num_bboxes):
        w = random.randint(min_dim, min(max_dim, img_w))
        h = random.randint(min_dim, min(max_dim, img_h))

        # Ensure x and y coordinates are valid
        if img_w - w > 0 and img_h - h > 0:
            x = random.randint(0, img_w - w)
            y = random.randint(0, img_h - h)
            bbox = frame[y:y+h, x:x+w]
            bboxes.append(bbox)

    return bboxes
