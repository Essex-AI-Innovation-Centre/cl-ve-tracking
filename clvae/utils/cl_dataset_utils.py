import os
import random
from torch.utils.data import DataLoader
from clvae.modules.cl_dataset import CLDataset
import json

def split_dataset(video_dir, train_split=0.8):
    """
    Splits the dataset into training and validation sets at the video level.
    """
    video_paths = sorted([os.path.join(video_dir, vid) for vid in os.listdir(video_dir) if vid.endswith(('.pkl', '.avi'))])
    random.seed(42)
    random.shuffle(video_paths)
    
    split_idx = int(len(video_paths) * train_split)
    train_videos = video_paths[:split_idx]
    val_videos = video_paths[split_idx:]
    
    return train_videos, val_videos

def create_loaders(video_dir, batch_size=1, pair_step=5, resize=(16,16), visualize=False, save_folder=None, validation_video_dir=None):
    """
    Creates DataLoader objects for training and validation sets.
    """
    if validation_video_dir is None:
        train_videos, val_videos = split_dataset(video_dir, train_split=0.8)
    else:
        train_videos = sorted([os.path.join(video_dir, vid) for vid in os.listdir(video_dir) if vid.endswith(('.pkl', '.avi'))])
        val_videos = sorted([os.path.join(validation_video_dir, vid) for vid in os.listdir(validation_video_dir) if vid.endswith(('.pkl', '.avi'))])
    
    if visualize:
        save_folder = os.path.join(save_folder, "patches")
    train_dataset = CLDataset(train_videos, pair_step=pair_step, resize=resize, visualize=visualize, save_folder=save_folder)
    val_dataset = CLDataset(val_videos, pair_step=pair_step, resize=resize, visualize=visualize, save_folder=save_folder)
    
    print(f"Total number of training samples: {len(train_dataset)}")
    print(f"Total number of validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    dataset_info = {
        "train_videos": train_videos,
        "val_videos": val_videos,
        "num_train_samples": len(train_dataset),
        "num_val_samples": len(val_dataset),
        "num_train_videos": len(train_videos),
        "num_val_videos": len(val_videos)
    }

    if save_folder != None:
        os.makedirs(save_folder, exist_ok=True)
        info_path = os.path.join(save_folder, "dataset_info.json")
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=4)

    return train_loader, val_loader
