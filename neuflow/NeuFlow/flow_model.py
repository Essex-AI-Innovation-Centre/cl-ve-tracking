import os
import cv2
import torch
import torch.nn.functional as F
from neuflow.NeuFlow.neuflow import NeuFlow
from neuflow.NeuFlow.backbone_v7 import ConvBlock
from neuflow.NeuFlow.utils import fuse_conv_and_bn


class NeuFlowModel:
    def __init__(self, cfg):
        """
        Args:
            cfg (dict): Configuration dictionary containing model settings.
                pretrained_model: str (pre-trained checkpoint path)
                input_shape: [int, int] (image input shape for flow computation)
                rescale: float (resize factor)
        """
        self.cfg = cfg
        self._initialized = False
        self.init_neuflow()

    def init_neuflow(self):
        self.neuflow = NeuFlow().to("cuda:0").eval()

        # load checkpoint
        current_folder = os.path.dirname(os.path.abspath(__file__))
        parent_folder = os.path.dirname(current_folder)
        ckpts_folder = os.path.join(parent_folder, "checkpoints")
        print(self.cfg)
        ckpt = torch.load(
            os.path.join(ckpts_folder, self.cfg["pretrained_model"]),
            map_location="cuda",
            weights_only=True,
        )
        self.neuflow.load_state_dict(ckpt["model"], strict=True)

        # fuse conv+bn
        for m in self.neuflow.modules():
            if isinstance(m, ConvBlock):
                m.conv1 = fuse_conv_and_bn(m.conv1, m.norm1)
                m.conv2 = fuse_conv_and_bn(m.conv2, m.norm2)
                delattr(m, "norm1")
                delattr(m, "norm2")
                m.forward = m.forward_fuse

        self.neuflow.eval()
        self.neuflow.half()

        # set input resolution
        self.input_height, self.input_width = [
            int(dim * self.cfg["rescale"]) for dim in self.cfg["input_shape"]
        ]

    def _ensure_bhwd(self, batch_size):
        """Initialize bhwd only once or if batch size changes."""
        if not self._initialized or batch_size != self._last_batch_size:
            self.neuflow.init_bhwd(batch_size, self.input_height, self.input_width, "cuda")
            self._initialized = True
            self._last_batch_size = batch_size

    def prepare_input(self, image):
        image = cv2.resize(image, (self.input_width, self.input_height))
        image = torch.from_numpy(image).permute(2, 0, 1).half()
        return image.cuda().unsqueeze(0)  # (1,3,H,W)

    def run(self, imgs):
        """Run optical flow on a single pair (img0, img1)."""
        img_0, img_1 = [self.prepare_input(img) for img in imgs]
        self._ensure_bhwd(batch_size=1)
        flow = self.neuflow(img_0, img_1)[-1]  # (1,2,H,W)
        return flow.float()

    def prepare_batch(self, images, tensor=False):
        if not tensor:
            tensors = []
            for img in images:
                img = cv2.resize(img, (self.input_width, self.input_height))
                img = torch.from_numpy(img).permute(2, 0, 1).half()
                tensors.append(img)
            return torch.stack(tensors, dim=0).cuda() # (B,3,H,W)
        else:
            batch = torch.stack(images, dim=0).cuda().half()
            batch_resized = F.interpolate(batch, size=(self.input_height, self.input_width), mode='bilinear', align_corners=False)
            return batch_resized # (B,3,H,W)

    def run_batch(self, batch_imgs, tensor=False):
        """Run optical flow on a batch of pairs [(img0,img1),...]."""
        imgs0 = [pair[0] for pair in batch_imgs]
        imgs1 = [pair[1] for pair in batch_imgs]

        img_0 = self.prepare_batch(imgs0, tensor=tensor)  # (B,3,H,W)
        img_1 = self.prepare_batch(imgs1, tensor=tensor)  # (B,3,H,W)

        self._ensure_bhwd(batch_size=img_0.shape[0])

        flow = self.neuflow(img_0, img_1)[-1]  # (B,2,H,W)
        return flow.float()
