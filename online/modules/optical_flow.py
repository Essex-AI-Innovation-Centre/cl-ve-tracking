import numpy as np
from neuflow.NeuFlow.flow_model import NeuFlowModel
from neuflow.utils.flow import resize_flow


class FlowModelSetup:
    def __init__(self, config):
        self.flow_model = NeuFlowModel(config)

    def flow_step(self, imgs):
        """
        Args:
            imgs: either
                - tuple/list [img0, img1] for a single pair
                - list of tuples [(img0, img1), ...] for a batch
        Returns:
            np.ndarray or list of np.ndarray (HxWx2 optical flow)
        """
        # single pair
        if isinstance(imgs[0], (np.ndarray,)):
            flow = self.flow_model.run(imgs)  # (1,2,H,W)
            flow = resize_flow(flow, imgs[0].shape[:2])
            np_flow = flow.squeeze(0).detach().cpu().numpy().transpose([1, 2, 0])
            return np_flow

        # batch of pairs
        elif isinstance(imgs[0], (tuple, list)):
            batch_flows = self.flow_model.run_batch(imgs)  # (B,2,H,W)
            shape = imgs[0][0].shape[:2]
            np_flows = []
            for i in range(batch_flows.shape[0]):
                flow = resize_flow(batch_flows[i].unsqueeze(0), shape)
                np_flow = flow.squeeze(0).detach().cpu().numpy().transpose([1, 2, 0])
                np_flows.append(np_flow)
            return np_flows

        else:
            raise ValueError("Invalid input format for flow(): expected [img0,img1] or list of pairs")
