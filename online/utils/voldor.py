import cv2

def rescale_flow(np_flow, rescale_factor):
    # Needed to downscale flow for VOLDOR
    new_size = (int(np_flow.shape[1]*rescale_factor), int(np_flow.shape[0]*rescale_factor))
    flow = cv2.resize(np_flow, new_size)
    flow *= rescale_factor
    return flow


class Frame:
    '''
    Needed to store camera pose, depth, and depth confidence for VOLDOR
    '''
    def __init__(self, Tcw, depth=None, depth_conf=None, scale=1.0, is_keyframe=False):
        self.Tcw = Tcw.copy()
        self.depth = depth
        self.depth_conf = depth_conf
        self.scale = scale
        self.is_keyframe = is_keyframe

    def get_scaled_depth(self):
        return self.depth * self.scale
