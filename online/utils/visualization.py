import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

def overlay_heatmap(image, heatmap_data, alpha=1.0):
    heatmap_data = (heatmap_data*255).astype(np.uint8)
    heatmap_normalized = cv2.normalize(heatmap_data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    heatmap_smoothed = cv2.GaussianBlur(heatmap_colored, (0, 0), 5)
    blended_image = cv2.addWeighted(image, 1 - alpha, heatmap_smoothed, alpha, 0)
    blended_image = cv2.cvtColor(blended_image, cv2.COLOR_RGB2BGR)
    return blended_image


class BoundingBoxSelector:
    """
    Auxiliary visualization class to select a bounding box from a plotted image.
    """
    def __init__(self, image):
        self.ax = plt.subplots()[1]
        self.image = image
        self.bbox = None
        self.rectangle_selector = RectangleSelector(
            self.ax, self.onselect,
            useblit=True, button=[1], minspanx=5, minspany=5,
            spancoords='pixels', interactive=True
        )
        
    def onselect(self, eclick, erelease):
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        self.bbox = (x1, y1, x2, y2)

    def select_bbox(self):
        self.ax.imshow(self.image)
        plt.show()
        return self.bbox
