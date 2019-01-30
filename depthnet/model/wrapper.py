import torch
import torch.nn as nn
import numpy as np

from .utils import ModelWrapper

class DepthNetWrapper(ModelWrapper):
    """Wrapper specific for depth estimation networks.
    """
    def __init__(self, network, pre_active, post_active, 
                 rgb_key, rgb_mean, rgb_var, min_depth, max_depth, device):
        super(DepthNetWrapper, self).__init__(network, pre_active, post_active)
        """
        rgb_mean - iterable - length should be the same as nchannels of rgb
        rgb_var - iterable - length should be the same as nchannels of rgb
        min_depth - minimum depth to clip to
        max_depth - maximum depth to clip to
        """
        # Preprocessing params
        self.rgb_key = rgb_key
        self.rgb_mean = rgb_mean
        self.rgb_var = rgb_var
        self.device = device


        # Postprocessing params
        self.min_depth = min_depth
        self.max_depth = max_depth

    def pre(self, input_):
        """Depth preprocessing
        - Make RGB zero mean unit variance.
        """
        rgb = input_[self.rgb_key]
        mean_tensor = torch.tensor(self.rgb_mean, dtype=rgb.dtype).unsqueeze(-1).unsqueeze(-1).to(self.device)
        var_tensor = torch.tensor(self.rgb_var, dtype=rgb.dtype).unsqueeze(-1).unsqueeze(-1).to(self.device)
        rgb = rgb.to(self.device)
        rgb = rgb - mean_tensor
        rgb = rgb / var_tensor
        input_[self.rgb_key] = rgb
        return input_

    def post(self, output):
        """Depth postprocessing
        - Clip depth to min and max values
        """
        depth = output
        if self.min_depth is not None:
            depth[depth < self.min_depth] = self.min_depth
        if self.max_depth is not None:
            depth[depth > self.max_depth] = self.max_depth
        return depth

class DORNWrapper(DepthNetWrapper):
    """Wrapper for depth networks using the Ordinal Regression Loss as in
    H. Fu et al., “Deep Ordinal Regression Network for Monocular Depth Estimation.”

    Wrapped model should output a per-pixel list of probabilities P_0,...,P_k-1
    where
    P_i = P(L > i)
    where L is the bin index corresponding to the estimated depth of this pixel.
    """

    def __init__(self, network, pre_active, post_active, 
                 rgb_key, rgb_mean, rgb_var, min_depth, max_depth, 
                 sid_bins, device):
        """
        :param depth_bins - array of same length as the number of output probs of the network.
                            Maps the bin number to the real-world depth value.
        """
        super(DORNWrapper, self).__init__(network, pre_active, post_active, rgb_key, rgb_mean,
                                          rgb_var, min_depth, max_depth, device)
        self.sid_offset = 1.0 - min_depth
        self.sid_bins = sid_bins
        # Compute sid_depths
        # If bin i is from t_k to t_k+1, then the depth associated with that bin is
        # d = (t_k + t_k+1)/2 - offset
        start = 1.0
        end = max_depth + self.sid_offset
        self.sid_bin_edges = np.array([np.power(end, i/sid_bins) for i in range(sid_bins+1)])
        self.sid_depths = (self.sid_bin_edges[:-1] + self.sid_bin_edges[1:])/2 - self.sid_offset
        self.sid_depths = torch.from_numpy(self.sid_depths).float().to(device)

    def post(self, output_probs):
        depth_index = torch.sum((output_probs >= 0.5), dim=1).long().unsqueeze(1)
        depth_vals = torch.take(self.sid_depths, depth_index)
        return depth_vals



if __name__ == '__main__':
    a = {"rgb": torch.tensor([[4, 6, 8], [8, 6, 4], [4, 6, 8]], dtype=torch.float32)}
    mean = torch.tensor([6, 5, 4], dtype=torch.float32)
    var = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    model = lambda d: torch.sum(d["rgb"], dim=0)
    wrapper = DepthNetWrapper(model, True, True, "rgb", mean, var, 0., 8, "cpu")

    out = wrapper(a)
    print(out)
    # print(wrapper.preprocessed)
    # print(wrapper.model_output)
    # print(wrapper.postprocessed)
