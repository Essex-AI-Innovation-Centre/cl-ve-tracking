import numpy as np
import torch

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

def flow_to_bbox(flow, bbox, precise=False, occlusion=False, roi_expand=1.0):
    '''
    Convert optical flow to new bounding box.
        Args: flow (numpy.ndarray): Optical flow of shape (H, W, 2)
              bbox (tuple): Bounding box in the format (x, y, width, height)

        Returns: tuple: New x, y coordinates for the bounding box.
    '''

    # calculate flow in a larger area around the ROI to account for object motion
    if occlusion:
        adjusted_bbox = bbox_size_adjust(bbox, 1, 1+roi_expand)
        flow_x, flow_y, flow_w, flow_h = tuple(map(round, adjusted_bbox))
        roi_flow = flow[flow_y:flow_y+flow_h, flow_x:flow_x+flow_w]
    else:
        x, y, w, h = tuple(map(round, bbox))
        roi_flow = flow[y:y+h, x:x+w]

    median_flow = np.median(roi_flow, axis=(0, 1))
    if np.isnan(median_flow[0]):
        median_flow[0] = 0
    if np.isnan(median_flow[1]):
        median_flow[1] = 0
    if precise:
        precise_x, precise_y = bbox[:2]
        precise_x += median_flow[0]
        precise_y += median_flow[1]
        return (precise_x, precise_y)
    else:
        new_x = round(x + median_flow[0])
        new_y = round(y + median_flow[1])
        return (new_x, new_y)
    
def resize_flow(flow, new_shape):
    '''
    Resize flow to new shape.
        Args: flow (torch.Tensor): Optical flow tensor of shape (1, 2, H, W)
              new_shape (tuple): New shape (H, W)
              
        Returns: torch.Tensor: Resized optical flow tensor of shape (1, 2, new_H, new_W)
    '''
    _, _, h, w = flow.shape
    new_h, new_w = new_shape
    flow = torch.nn.functional.interpolate(flow, (new_h, new_w),
                                           mode='bilinear', align_corners=True)
    scale_h, scale_w = h / float(new_h), w / float(new_w)
    flow[:, 0] /= scale_w
    flow[:, 1] /= scale_h
    return flow

def save_flow(path, flow):
    magic = np.array([202021.25], np.float32)
    h, w = flow.shape[:2]
    h, w = np.array([h], np.int32), np.array([w], np.int32)

    with open(path, 'wb') as f:
        magic.tofile(f); w.tofile(f); h.tofile(f); flow.tofile(f)
