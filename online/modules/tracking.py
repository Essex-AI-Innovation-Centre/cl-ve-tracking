import cv2
import os
import numpy as np
import random
import json
import torch
import time
import matplotlib.pyplot as plt
from neuflow.utils.flow import flow_to_bbox
from online.utils.tracking import *
from online.utils.visualization import overlay_heatmap
from clvae.modules.vae import VAE, CLVAE

class TrackingHandler:
    def __init__(self, cfg, camera, rescale_factor=None):
        self.tracked_areas = {}
        self.camera = camera
        self.rescale = rescale_factor if rescale_factor is not None else 1.0
        for key, value in cfg.items():
            setattr(self, key, cfg[key])
        for key, value in cfg[self.feature_extractor.replace('cl','')].items():
            setattr(self, key, value)
        self.features_detected = False

        self.matching_times = []
        with open(f"{self.checkpoint}/training_params.json", 'r') as f:
            model_config = json.load(f)

        if self.feature_extractor == 'vae' and self.checkpoint != None: 
            img_channels = 1 if model_config["gray_scale"] else 3
            self.detector = VAE(img_channels=img_channels, latent_dim=model_config["latent_dim"])
            model_path = os.path.join(self.checkpoint, "model.pth")
            self.detector.load_model(model_path)
        elif self.feature_extractor == 'clvae' and self.checkpoint != None:
            clvae_params = {param: model_config[param]
                            for param in ["img_channels", "latent_dim","projection_dim"]}
            clvae_params["img_height"], clvae_params["img_width"] = model_config['resize']
            self.resize = (clvae_params["img_width"], clvae_params["img_height"])
            self.detector = CLVAE(**clvae_params)
            model_path = os.path.join(self.checkpoint, "vae_best.pth")
            model_state_dict = torch.load(model_path, weights_only=True)["model_state_dict"]
            model_state_dict.pop("temperature")
            self.detector.load_state_dict(model_state_dict)
        self.detector.to("cuda")

    def register_tracking_area(self, frame, idx, flows, depth_map, T_kf):
        # Show selection window
        bbox = self.select_roi(frame, idx)

        # Move centroid backwards with flows to get centroid of selected ROI on last keyframe
        x, y, w, h = bbox
        backward_flows = [-1.0 * flow for flow in flows[::-1]]
        for i in range(len(backward_flows)):
            x, y = flow_to_bbox(backward_flows[i], (x, y, w, h), precise=True)
        kf_x_pix = x + w/2
        kf_y_pix = y + h/2

        # Get corresponding depth on last keyframe
        depth_coords = (round(kf_x_pix*self.rescale), round(kf_y_pix*self.rescale))
        depth_map_height, depth_map_width = depth_map.shape
        depth_map_y = max(0, min(depth_map_height-1, depth_coords[1]))
        depth_map_x = max(0, min(depth_map_width-1, depth_coords[0]))
        y_bounds = (max(0,depth_map_y-h//2), min(depth_map_height-1,depth_map_y+h//2))
        x_bounds = (max(0,depth_map_x-w//2), min(depth_map_width-1,depth_map_x+w//2))
        depth = np.median(depth_map[y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]])

        # Back project to get last keyframe 3D coordinates
        kf_xyz = back_project((kf_x_pix, kf_y_pix), depth, self.camera.K_inv)
        # Get initial 3D coordinates and track this point instead
        xyz = transform_point(kf_xyz, T_kf)

        new_area = TrackingArea(frame, idx, bbox, xyz, kf_xyz[2])
        new_area.extract_embedding(self.detector, self.resize)
        print(f'Area {idx} registered!')
        self.tracked_areas[idx] = new_area

    def select_roi(self, frame, idx):
        if self.manual_selection:
            cv2.destroyAllWindows()
            display_frame = frame.copy()
            cv2.namedWindow(f"Select area No. {idx}", cv2.WINDOW_NORMAL)
            cv2.resizeWindow(f"Select area No. {idx}", 800, 600)
            probe_box = cv2.selectROI(f"Select area No. {idx}", display_frame, showCrosshair=True, fromCenter=False)
            cv2.destroyAllWindows()
        else:
            ranges = [(200, 1000), (200, 500), (50, 150), (30, 100)]
            probe_box = tuple([random.randint(a, b) for a, b in ranges])
        return probe_box

    def register_ground_truth_area(self, gt_bbox, frame, idx, flows, depth_map, T_kf, search_area=None):
        x, y, w, h = gt_bbox
        backward_flows = [-1.0 * flow for flow in flows[::-1]]
        for i in range(len(backward_flows)):
            x, y = flow_to_bbox(backward_flows[i], (x, y, w, h), precise=True)
        kf_x_pix = x + w/2
        kf_y_pix = y + h/2

        # Get corresponding depth on last keyframe
        depth_coords = (round(kf_x_pix*self.rescale), round(kf_y_pix*self.rescale))
        depth_map_height, depth_map_width = depth_map.shape
        depth_map_y = max(0, min(depth_map_height-1, depth_coords[1]))
        depth_map_x = max(0, min(depth_map_width-1, depth_coords[0]))
        y_bounds = (max(0,depth_map_y-h//2), min(depth_map_height-1,depth_map_y+h//2))
        x_bounds = (max(0,depth_map_x-w//2), min(depth_map_width-1,depth_map_x+w//2))
        depth = np.median(depth_map[y_bounds[0]:y_bounds[1], x_bounds[0]:x_bounds[1]])

        # Back project to get last keyframe 3D coordinates
        kf_xyz = back_project((kf_x_pix, kf_y_pix), depth, self.camera.K_inv)
        # Get initial 3D coordinates and track this point instead
        xyz = transform_point(kf_xyz, T_kf)

        new_area = TrackingArea(frame, idx, gt_bbox, xyz, kf_xyz[2])
        if search_area:
            new_area.search = search_area        
        self.tracked_areas[idx] = new_area
        new_area.extract_embedding(self.detector, self.resize)

    def visualize_tracking_areas(self, frame, voldor_test_mode=False):
        for area in self.tracked_areas.values():
            if area.vis:
                if (not area.use_flow) or voldor_test_mode:
                    s_x, s_y, s_w, s_h = area.search

                    if self.heatmap and area.heatmap_data is not None:
                        heatmap_frame = overlay_heatmap(frame, area.heatmap_data, alpha=self.alpha)
                        frame[s_y:s_y+s_h, s_x:s_x+s_w] = heatmap_frame[s_y:s_y+s_h, s_x:s_x+s_w]

                    cv2.rectangle(frame, (s_x, s_y), (s_x+s_w, s_y+s_h), (255, 0, 0), 2)

                    if area.kps is not None:
                        for kp in area.kps:
                            cv2.circle(frame, kp, 2, (255, 0, 0), 2)
    
                cv2.rectangle(frame, (area.x, area.y), (area.x + round(area.w), area.y + round(area.h)), (0, 255, 0), 2)

        return frame

    def visualize_ground_truth(self, display_frame, gt_bbox):
        for i in range(len(self.tracked_areas)):
            x, y, w, h = gt_bbox#[i] # TODO: for multiple areas
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (33, 222, 255), 2)
        return display_frame

    def OF_bbox_step(self, frame, area, flows):
        ''' In bounds case, move using OF (self.x_pix flow) '''
        # Apply sequential optical flow transformations from previous to current keyframe
        for i in range(len(flows)):
            area.x, area.y = flow_to_bbox(flows[i], (area.x, area.y, area.w, area.h), precise=True, occlusion=False)

        # Adjust bbox size according to the estimated depth
        area.adjust_size()

        frame_height, frame_width = frame.shape[:2]

        # Incorporate template matching during in-frame tracking
        if 0 < area.x < frame_width and 0 < area.y < frame_height:
            current_patch = frame[area.y:area.y+area.h, area.x:area.x+area.w]
            current_patch = cv2.cvtColor(current_patch, cv2.COLOR_BGR2RGB)
            current_patch = cv2.resize(current_patch, self.resize)/255.0
            current_patch = torch.tensor(current_patch, dtype=torch.float32).to("cuda")
            current_embedding = self.detector.get_embedding(current_patch)
            similarity = compare_embeddings(area.template_embedding, current_embedding)
            if similarity < self.similarity_threshold:
                area.use_flow = False
                area.vis = False
                area.extract_embedding(self.detector, self.resize)
                return
            else:
                area.vis = True
        
        # Move ROI
        if 0 < area.x < frame_width and 0 < area.y < frame_height:
            area.use_flow = True
            area.last_detected_area = frame[area.y:area.y+area.h, area.x:area.x+area.w, :]
        else:
            area.vis = False
            area.use_flow = False
        
        self.features_detected = True

    def LM_bbox_step(self, frame, area, flows, depth_map, T_kf, voldor_test_mode=False):
        ''' Out of bound case, track using LM (Localization + Feature Matching) '''
        # Transform ROI centroid to the current frame
        x_pix, y_pix = project_point(area.transformed_xyz, self.camera.K)

        # Estimated ROI bbox
        area.x = x_pix - area.w/2
        area.y = y_pix - area.h/2

        # Adjust bbox size according to the estimated depth
        area.adjust_size()

        # Search for features only if within frame
        frame_height, frame_width = frame.shape[:2]
        if 0 < x_pix < frame_width and 0 < y_pix < frame_height:
            area.vis = True
            search_min_x, search_min_y, search_max_x, search_max_y = get_search_box((area.x, area.y, area.w, area.h), frame_height, frame_width, self.search_expand_factor)
            search_area = frame[search_min_y:search_max_y, search_min_x:search_max_x, :]
            area.search = (search_min_x, search_min_y, search_max_x-search_min_x, search_max_y-search_min_y) 
            area.lm_steps += 1

            t1 = time.time()

            # Prepare candidate image matches
            candidate_images = []
            candidate_bboxes = generate_bboxes_in_search_area((area.x,area.y,area.w,area.h), frame.shape[0], frame.shape[1],
                                                                self.search_expand_factor, self.stride)
            if len(candidate_bboxes) > 0:
                for candidate_bbox in candidate_bboxes:
                    c_x, c_y, c_w, c_h = [candidate_bbox[key] for key in ["x", "y", "w", "h"]]
                    candidate_patch = frame[c_y:c_y+c_h, c_x:c_x+c_w]
                    candidate_patch = cv2.cvtColor(candidate_patch, cv2.COLOR_BGR2RGB)
                    candidate_patch = cv2.resize(candidate_patch, self.resize)/255.0
                    candidate_images.append(candidate_patch)
                candidate_patches = torch.tensor(np.array(candidate_images), dtype=torch.float32).to("cuda")
                candidate_embeddings = self.detector.get_embedding(candidate_patches)
                similarities = compare_embeddings(candidate_embeddings, area.template_embedding)

                # Keep best match for this frame
                closest_idx = torch.argmax(similarities).item()
                area.x, area.y, area.w, area.h = [candidate_bboxes[closest_idx][key] for key in ["x", "y", "w", "h"]]

                t2 = time.time()
                self.matching_times.append(t2 - t1)

                # Explainability visualization
                if self.heatmap:
                    normalized_similarities = normalize_similarities(similarities)
                    area.heatmap_data = np.zeros_like(frame, dtype=np.float32)
                    sorted_indices = np.argsort(normalized_similarities)
                    sorted_similarities = normalized_similarities[sorted_indices]
                    sorted_candidate_bboxes = [candidate_bboxes[i] for i in sorted_indices]
                    for i, candidate_bbox in enumerate(sorted_candidate_bboxes):
                        c_x, c_y, c_w, c_h = [candidate_bbox[key] for key in ["x", "y", "w", "h"]]
                        area.heatmap_data[c_y:c_y+c_h, c_x:c_x+c_w] = sorted_similarities[i].item()

                if not voldor_test_mode:
                    bbox_tuple = tuple([candidate_bboxes[closest_idx][key] for key in ["x", "y", "w", "h"]])
                    max_similarity = similarities[closest_idx].item()
                    area.matching_candidates.append({
                        'bbox': bbox_tuple,
                        'similarity': max_similarity,
                        'flow': [np.zeros_like(frame)]*len(flows) if area.lm_steps == 1 else flows,
                        'bbox_curr_frame': None,
                        'bbox_centroid': None,
                        'matching_distance': None})

            # Matching decision
            if area.lm_steps == self.decision_steps and not voldor_test_mode:
                area.lm_steps = 0

                if area.matching_candidates == []:
                    area.use_flow = False
                    return
                
                area.use_flow = True
                cand_frame = frame.copy()

                # Move matching bboxes on the current frame using flows stored
                matching_flows = [match['flow'] for match in area.matching_candidates]
                for step, match in enumerate(area.matching_candidates):
                    x_b, y_b, w_b, h_b = match['bbox']
                    flows_to_curr = [flow for match in matching_flows[step+1:] for flow in match]
                    for flow in flows_to_curr:
                        x_b, y_b = flow_to_bbox(flow, (x_b, y_b, w_b, h_b), precise=True)

                    new_x = min(max(0, x_b), frame.shape[1])
                    new_y = min(max(0, y_b), frame.shape[0])

                    new_w = w_b - abs(new_x - x_b)
                    new_h = h_b - abs(new_y - y_b)

                    area.matching_candidates[step]['bbox_curr_frame'] = tuple(map(round, (new_x, new_y, new_w, new_h)))

                    if self.visualize_decision:
                        x_b, y_b, w_b, h_b = area.matching_candidates[step]['bbox_curr_frame']
                        cv2.rectangle(cand_frame, (x_b, y_b), (x_b + w_b, y_b + h_b), (0, 100, 100), 2)
                        cv2.putText(cand_frame, str(step+1), (x_b, y_b-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 100), 2)                            
                
                if self.decision_consensus:
                    # Reach consensus based on the median of the matching bboxes
                    for match in area.matching_candidates:
                        match['bbox_centroid'] = get_bbox_centroid(match['bbox_curr_frame'])

                    # Compute median of the centroids
                    matching_bbox_centroids = [match['bbox_centroid'] for match in area.matching_candidates]
                    matching_bbox_centroids = np.stack(matching_bbox_centroids, axis=0)
                    centroid_median_y = round(np.median(matching_bbox_centroids[:,1]))
                    centroid_median_x = round(np.median(matching_bbox_centroids[:,0]))

                    # Compute distances to the median
                    for match in area.matching_candidates:
                        match['matching_distance'] = np.linalg.norm(np.array([centroid_median_x, centroid_median_y]) - np.array(match['bbox_centroid']))

                    # Reject outlier bounding box matches
                    centroid_matching_distances = [match['matching_distance'] for match in area.matching_candidates]
                    mean_centroid_distance = np.mean([dist for dist in centroid_matching_distances])
                    centroid_distance_threshold = mean_centroid_distance * (1 + self.outlier_percentage)

                    if self.centroid_consensus:
                        filtered_matching_centroids = [match['bbox_centroid'] for match in area.matching_candidates if match['matching_distance'] <= centroid_distance_threshold]
                        filtered_matching_centroids = np.stack(filtered_matching_centroids, axis=0)
                        centroid_median_y = round(np.median(filtered_matching_centroids[:,1]))
                        centroid_median_x = round(np.median(filtered_matching_centroids[:,0]))
                        area.x = centroid_median_x - w_b//2
                        area.y = centroid_median_y - h_b//2
                        area.w, area.h = w_b, h_b

                        if self.visualize_decision:
                            x_c, y_c, w_c, h_c = tuple(map(round, (area.x, area.y, area.w, area.h)))
                            cv2.rectangle(cand_frame, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 255, 0), 2)
                            cv2.putText(cand_frame, f"Centroid", (x_c, y_c-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        filtered_matching_candidates = [match for match in area.matching_candidates if match['matching_distance'] <= centroid_distance_threshold]
                        best_match = max(filtered_matching_candidates, key=lambda x: x['similarity'])
                        best_match_x, best_match_y, _, _ = best_match['bbox_curr_frame']
                        area.x, area.y, area.w, area.h = best_match_x, best_match_y, w_b, h_b

                        if self.visualize_decision:
                            x_c, y_c, w_c, h_c = tuple(map(round, (area.x, area.y, area.w, area.h)))
                            cv2.rectangle(cand_frame, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 255, 0), 2)
                            cv2.putText(cand_frame, f"Max", (x_c, y_c-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    # Choose based on the single best similarity match
                    best_match = max(area.matching_candidates, key=lambda x: x['similarity'])
                    area.x, area.y, area.w, area.h = best_match['bbox_curr_frame']

                    if self.visualize_decision:
                        x_c, y_c, w_c, h_c = tuple(map(round, (area.x, area.y, area.w, area.h)))
                        cv2.rectangle(cand_frame, (x_c, y_c), (x_c + w_c, y_c + h_c), (0, 255, 0), 2)
                        cv2.putText(cand_frame, f"Max", (x_c, y_c-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                if self.visualize_decision:
                    plt.imshow(cv2.cvtColor(cand_frame, cv2.COLOR_BGR2RGB))
                    plt.axis('off')
                    plt.show()

                area.last_detected_area = frame[area.y:area.y+area.h, area.x:area.x+area.w, :]
                area.extract_embedding(self.detector, self.resize)
                area.matching_candidates = []

                if self.localization_correction:
                    area.x, area.y, area.w, area.h = list(map(lambda x : max(0, x), (area.x, area.y, area.w, area.h)))
                    self.register_ground_truth_area((area.x, area.y, area.w, area.h), frame, area.idx, flows, depth_map, T_kf, area.search)
        else:
            area.vis = False
            area.lm_steps = 0
            area.matching_candidates = []
