import os
import pickle
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from PIL import Image
import math
from clvae.utils.augmentation_utils import largest_rotated_rectangle, rotate_image, crop_around_center


class CLDataset(Dataset):
    def __init__(self, dataset_path, pair_step=5, resize=(16,16), augmentation=True, visualize=False, save_folder=None):
        """
        Arguments:
            - dataset_path: folder containing pickle files with TCC patch tracking
            - pair_step: frame difference of positive pairs
            - resize: model input size
            - augmentation: apply data augmentation
            - visualize: save positive pairs as images
            - save_folder: folder to save positive pair images
        """
        self.dataset_path = dataset_path
        self.pair_step = pair_step
        self.resize = resize
        self.augmentation = augmentation
        self.visualize = visualize
        self.save_folder = save_folder
        self.pairs = self.get_pairs()

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        item = self.pairs[idx] # shape: (2, B, H, W, 3)
        tensor = torch.from_numpy(item.astype(np.float32)).permute(0,1,4,2,3)
        return tensor # shape: (2, B, 3, H, W)

    def get_pairs(self):
        pairs = []
        for file in self.dataset_path:
            if file.endswith('.pkl'):
                with open(file, 'rb') as f:
                    track_data = pickle.load(f)
                pos_pairs = {}
                for patch_idx, patch_traj in track_data.items():
                    pos_pairs[patch_idx] = []
                    for i in range(0, len(patch_traj)-self.pair_step):
                        if 0 not in (patch_traj[i].shape + patch_traj[i+self.pair_step].shape):
                            img1 = patch_traj[i]
                            img2 = patch_traj[i+self.pair_step]
                            if self.augmentation and np.random.rand() >= 0.5:
                                h, w = img2.shape[:2]
                                rectangle_proportion = w / h if w < h else h / w
                                max_angle = 45 * rectangle_proportion
                                angle = np.random.uniform(-max_angle, max_angle)
                                image_rotated = rotate_image(img2, angle)
                                img2 = crop_around_center(image_rotated, 
                                                          *largest_rotated_rectangle(w, h, math.radians(angle)))
                            p1 = cv2.resize(img1, self.resize) / 255.
                            p2 = cv2.resize(img2, self.resize) / 255.
                            pos_pairs[patch_idx].append(np.stack((p1, p2), axis=0))
                # create negative pairs by batching positive pairs
                pair_batches = self.make_pair_batches(list(pos_pairs.values()))
                if self.visualize:
                    self.save_batches(pair_batches, file.split("/")[-1].replace('.pkl', ''))
                pairs.extend(pair_batches)
        return pairs
    
    def make_pair_batches(self, pos_pairs):
        pair_batches = []
        # batch must have at least 2 positive pairs
        while sum(len(lst) > 0 for lst in pos_pairs) > 1:
            current_batch = []
            for l in pos_pairs:
                if l: # from every non-empty patch trajectory
                    rand_idx = random.randint(0, len(l) - 1) # randomly select a positive pair
                    sample = l[rand_idx]
                    current_batch.append(sample)
                    l.pop(rand_idx)
            current_batch = np.stack(current_batch, axis=1)
            pair_batches.append(current_batch)
        return pair_batches

    def save_batches(self, pair_batches, name):
        os.makedirs(self.save_folder, exist_ok=True)
        for b_idx, batch in enumerate(pair_batches):
            num_rows = 2
            num_cols = batch.shape[1]
            combined_image = Image.new('RGB', (num_cols * self.resize[0], num_rows * self.resize[1]))
            for i in range(num_rows):
                for j in range(num_cols):
                    image = batch[i, j]
                    pil_image = Image.fromarray((image * 255).astype(np.uint8))
                    combined_image.paste(pil_image, (j * self.resize[0], i * self.resize[1]))
            combined_image.save(os.path.join(self.save_folder, f"{name}_{b_idx}.png"))
