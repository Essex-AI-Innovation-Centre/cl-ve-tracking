from scipy.spatial.distance import cdist
import cv2
import numpy as np
import torch

def extract_features(img, model):
    img = cv2.resize(img, (64, 64))  
    img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0 
    with torch.no_grad():
        features = model.encoder(img_tensor)
    return features.squeeze().cpu().numpy()

def match_features(img, model, roi_descriptors, threshold=0.5):
    img_features = extract_features(img, model)
    distances = cdist(roi_descriptors.reshape(1, -1), img_features.reshape(1, -1), metric='euclidean')
    matches = np.where(distances < threshold)  
    return matches
