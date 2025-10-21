import cv2
import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from online.utils.visualization import BoundingBoxSelector

class TrackingArea:
    '''
    For establishing probe detection area and handle independent area tracking 
    '''
    def __init__(self, frame, idx, probe_box, xyz, z_kf, measurement=random.random()):
        self.idx = idx
        self.x, self.y, self.w, self.h = probe_box
        self.measurement = measurement
        self.xyz = xyz
        self.z_kf = z_kf
        self.w_kf, self.h_kf = self.w, self.h
        self.use_flow = True
        self.lm_steps = 0
        self.search = None
        self.vis = True
        self.kps = None
        self.matching_candidates = []
        self.heatmap_data = None
        roi_image = frame.astype(np.uint8)[int(self.y):int(self.y+self.h), int(self.x):int(self.x+self.w), :]
        self.gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_RGB2GRAY)
        self.last_detected_area = roi_image.astype(np.float32)

    def extract_features(self, detector):
        _, self.roi_descriptors = detector.detectAndCompute(self.gray_roi, None)
        self.n_desc = self.roi_descriptors.shape[0] if self.roi_descriptors is not None else 0

    def extract_embedding(self, detector, resize):
        try:
            initial_patch = cv2.cvtColor(self.last_detected_area, cv2.COLOR_BGR2RGB)
            initial_patch = cv2.resize(initial_patch, resize)/255.0
            initial_patch = torch.tensor(initial_patch, dtype=torch.float32).to("cuda")
            self.template_embedding = detector.get_embedding(initial_patch)
        except:
            pass

    def track(self, T):
        T_inv = np.linalg.inv(T)
        R = T_inv[:3, :3]
        t = T_inv[:3, 3]
        xyz_i = R @ self.xyz + t
        self.transformed_xyz = xyz_i

    def adjust_size(self):
    # Adjust bounding box size according to depth estimation,
    # the larger the depth, the smaller the bounding box
        z_curr = self.transformed_xyz[2]
        gain = self.z_kf / z_curr
        if gain > 0:
            centroid_x = self.x + self.w/2
            centroid_y = self.y + self.h/2
            self.w = self.w_kf * np.sqrt(gain) # area multiplied by bbox_gain
            self.h = self.h_kf * np.sqrt(gain) # so each side multiplied by sqrt
            self.x = centroid_x - self.w/2
            self.y = centroid_y - self.h/2
            (self.x, self.y, self.w, self.h) = tuple(map(round, (self.x, self.y, self.w, self.h)))


def compare_embeddings(target_embeddings, initial_embed):
    target_embeddings_normalized = F.normalize(target_embeddings, p=2, dim=1)
    initial_embed_normalized = F.normalize(initial_embed, p=2, dim=1)
    similarities = F.cosine_similarity(target_embeddings_normalized, initial_embed_normalized, dim=1)
    return similarities

def normalize_similarities(similarities):
    min_sim = torch.min(similarities)
    max_sim = torch.max(similarities)
    normalized_similarities = (similarities - min_sim) / (max_sim - min_sim)
    return normalized_similarities

def match_features(search_area, detector, roi_descriptors, norm, outlier_percentage=0.5):
    gray_frame = cv2.cvtColor(search_area, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = detector.detectAndCompute(gray_frame, None)
    if descriptors is None:
        return [], []
    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(roi_descriptors, descriptors)
    mean_distance = np.mean([match.distance for match in matches])
    max_distance = mean_distance * (1 + outlier_percentage)
    matches = [match for match in matches if match.distance <= max_distance]
    return matches, keypoints

def bbox_size_adjust(bbox, z_init, z_new):
    x, y, w, h = bbox
    bbox_gain = z_init / z_new
    centroid_x = x + w/2
    centroid_y = y + h/2
    new_w = w * np.sqrt(bbox_gain) # area multiplied by bbox_gain
    new_h = h * np.sqrt(bbox_gain) # so each side multiplied by sqrt
    new_x = centroid_x - new_w/2
    new_y = centroid_y - new_h/2
    adjusted_bbox = (new_x, new_y, new_w, new_h) 
    return adjusted_bbox

def get_bbox_centroid(bbox):
    x, y, w, h = bbox
    bbox_cent = np.array([int(x + w/2), int(y + h/2)])
    return bbox_cent

def get_search_box(localization_area, frame_height, frame_width, search_expand_factor):
    x, y, w, h = localization_area
    error = (w + h) * search_expand_factor
    search_min_y = round(max(0, y-error))
    search_max_y = round(min(y + h + error, frame_height))
    search_min_x = round(max(0, x-error))
    search_max_x = round(min(x + w + error, frame_width))
    return search_min_x, search_min_y, search_max_x, search_max_y

def generate_bboxes_in_search_area(area, frame_height, frame_width, search_expand_factor, stride=10):
    _, _, w, h = area
    search_min_x, search_min_y, search_max_x, search_max_y = get_search_box(area, frame_height, frame_width, search_expand_factor)
    search_max_x = search_max_x - w  
    search_max_y = search_max_y - h  
    bboxes = []
    for y in range(search_min_y, search_max_y + 1, stride):  
        for x in range(search_min_x, search_max_x + 1, stride):  
            bboxes.append({
                "x": x,
                "y": y,
                "w": w,
                "h": h
            })
    return bboxes

def select_roi(frame, bgr=False):
    use_pyqt = False
    try:
        from PyQt5.QtCore import QLibraryInfo
        cv2_qt_path = os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QLibraryInfo.location(QLibraryInfo.PluginsPath)
        use_pyqt = True
    except:
        print("PyQt5 is not Required")
    if bgr:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    bbox_selector = BoundingBoxSelector(frame/255.)
    x1, y1, x2, y2 = bbox_selector.select_bbox()
    w, h = x2-x1, y2-y1
    x, y = x1, y1
    if use_pyqt:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = cv2_qt_path
    return x, y, w, h

def get_iou(bbox_gt, bbox_p):
    gt_left, gt_top, gt_right, gt_bot = [bbox_gt[0], bbox_gt[1], bbox_gt[0]+bbox_gt[2], bbox_gt[1]+bbox_gt[3]]
    gt_width, gt_height = bbox_gt[2], bbox_gt[3]
    p_left, p_top, p_right, p_bot = [bbox_p[0], bbox_p[1], bbox_p[0]+bbox_p[2], bbox_p[1]+bbox_p[3]]
    p_width, p_height = bbox_p[2], bbox_p[3]
    inter_left = max(gt_left, p_left)
    inter_top = max(gt_top, p_top)
    inter_right = min(gt_right, p_right)
    inter_bot = min(gt_bot, p_bot)
    inter_width = np.maximum(0,inter_right - inter_left)
    inter_height = np.maximum(0,inter_bot - inter_top)
    inter_area = inter_width * inter_height
    gt_area = gt_width * gt_height
    p_area = p_width * p_height
    union_area = gt_area + p_area - inter_area
    iou = inter_area / float(union_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def get_l2_norm(centr_gt, centr_p):
    return np.linalg.norm(centr_gt - centr_p)

def back_project(pixel_coords, depth, K_inv):
    '''Convert pixel coordinates to 3D coordinates'''
    u, v = pixel_coords
    uv1 = np.array([u, v, 1.0])
    xyz = depth * K_inv @ uv1
    return xyz

def transform_point(xyz_0, T):
    '''Transform 3D coordinates from one frame to another'''
    T_inv = np.linalg.inv(T)
    R = T_inv[:3, :3]
    t = T_inv[:3, 3]
    xyz_i = R @ xyz_0 + t
    return xyz_i

def project_point(xyz, K):
    '''Project 3D coordinates to pixel coordinates'''
    x, y, z = xyz
    uvw = K @ np.array([x, y, z])
    u, v = uvw[:2] / uvw[2]
    return np.array([u, v])
