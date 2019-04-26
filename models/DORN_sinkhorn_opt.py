import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
from models.data.utils.sid_utils import SIDTorch
from models.sinkhorn_dist import optimize_depth_map
from torch.optim import SGD
import matplotlib
# matplotlib.use("TKAgg")
# import matplotlib.pyplot as plt
import os

# from models.core.model_core import Model

from models.DORN_nohints import DORN_nyu_nohints

class DORN_sinkhorn_opt:
    """
    Performs SGD to optimize the depth map further after being given an initial depth map
    estimate from DORN.
    """
    def __init__(self, sgd_iters=200, sinkhorn_iters=20, sigma=3., lam=1e0,
                 use_albedo=True, use_squared_falloff=True,
                 lr=1e4, hints_len=68, spad_weight=1.,
                 in_channels=3, in_height=257, in_width=353,
                 sid_bins=68, offset=0.,
                 min_depth=0., max_depth=10.,
                 alpha=0.6569154266167957, beta=9.972175646365525,
                 frozen=True, pretrained=True,
                 state_dict_file=os.path.join("models", "torch_params_nyuv2_BGR.pth.tar")):
        """
        :param hints_len: Uniformly spaced noisy depth hints (i.e. raw SPAD data)
        :param num_hints_layers: The number of layers for performing upsampling
        """
        self.sgd_iters = sgd_iters
        self.sinkhorn_iters = sinkhorn_iters
        self.sigma = sigma
        self.lam = lam

        # Define cost matrix for optimal transport problem
        C = np.array([[(i - j)**2 for j in range(sid_bins)] for i in range(sid_bins)])
        self.cost_mat = torch.from_numpy(C).float()

        self.use_albedo = use_albedo
        self.use_squared_falloff = use_squared_falloff
        self.lr = lr
        # self.loss.to(device)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.spad_weight = spad_weight
        self.hints_len = hints_len
        self.sid_bins = sid_bins
        self.feature_extractor = \
            DORN_nyu_nohints(in_channels, in_height, in_width,
                             sid_bins, offset,
                             min_depth, max_depth,
                             alpha, beta,
                             frozen, pretrained,
                             state_dict_file)
        self.feature_extractor.eval()    # Only use DORN in eval mode.

        self.sid_obj = SIDTorch(sid_bins, alpha, beta, offset)
        self.one_over_depth_squared = 1./(self.sid_obj.sid_bin_values[:-2] ** 2)
        self.one_over_depth_squared.requires_grad = False

    def to(self, device):
        self.feature_extractor.to(device)
        self.cost_mat = self.cost_mat.to(device)
        self.sid_obj.to(device)

    def initialize(self, input_, device):
        """Feed rgb through DORN
        :return per-pixel one-hot indicating the depth bin for that pixel.
        """
        rgb = input_["rgb"].to(device)
        with torch.no_grad():
            x = self.feature_extractor(rgb)
            log_probs, _ = self.to_logprobs(x)
            depth_index = torch.sum((log_probs >= np.log(0.5)), dim=1, keepdim=True).long()
        print(depth_index[:,:,100:105,100:105])
        return depth_index

    # def optimize_depth_map(self, depth_hist, input_, device, resize_output=False):
    def optimize_depth_map(self, depth_index_init, input_, device, resize_output=False):
        spad_hist = input_["spad"].to(device)
        # print(spad_hist)
        depth_index_final, depth_img_final, depth_hist_final = \
            optimize_depth_map(depth_index_init, self.sigma, self.sid_bins,
                               self.cost_mat, self.lam, spad_hist,
                               self.lr, self.sgd_iters, self.sinkhorn_iters)
        depth_index_final = depth_index_final.detach().long()

        # Get depth maps and compute metrics
        depth_pred = self.sid_obj.get_value_from_sid_index(depth_index_final)
        if resize_output:
            original_size = input_["rgb_orig"].size()[-2:]
            # Note: align_corners=False gives same behavior as cv2.resize
            depth_pred = F.interpolate(depth_pred, size=original_size,
                                       mode="bilinear", align_corners=False)
            print("resized")
        return depth_pred

    def spad_forward(self, depth_hist, albedo):
        """Perform the forward simulation of a model"""
        N, C, W, H = depth_hist.size()
        # probs = torch.exp(log_probs)
        # Get histogram from 1-cdf
        # probs = probs[:, :-1, :, :] - probs[:, 1:, :, :]
        # print(torch.sum(probs[0,:,0,0]))
        # probs = torch.cat([self.zero_pad, probs], dim=1)
        # Use probs to get the weighted sum of albedo/depth^2
        one_over_depth_squared_unsqueezed = self.one_over_depth_squared.unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        # print(probs.shape)
        # print(self.one_over_depth_squared.expand(N, C, W, H).shape)
        # print(albedo_expanded.shape)

        weights = depth_hist
        if self.use_albedo:
            weights *= albedo
        if self.use_squared_falloff:
            weights *= one_over_depth_squared_unsqueezed
        simulated_spad = torch.sum(weights, dim=(2, 3), keepdim=True)
        print("simulated scale factor: {}".format(W*H/torch.sum(simulated_spad)))
        simulated_spad *= W*H/torch.sum(simulated_spad)
        return simulated_spad

    def evaluate(self, data, device):
        depth_init = self.initialize(data, device)
        pred = self.optimize_depth_map(depth_init, data, device, resize_output=False)
        # compute metrics
        gt = data["rawdepth_orig"].to(device)
        mask = data["mask_orig"].to(device)

        # gt = data["rawdepth"].to(device)
        # mask = data["mask"].to(device)

        # Also compute initial metrics:
        depth_init_map = self.sid_obj.get_value_from_sid_index(depth_init)
        torch.set_printoptions(profile="full")
        metrics = self.get_metrics(pred, gt, mask)
        # before_metrics = self.get_metrics(depth_init_map, gt, mask)
        return pred, metrics

DORN_sinkhorn_opt.to_logprobs = staticmethod(DORN_nyu_nohints.to_logprobs)
# DORN_sinkhorn_opt.ord_decode = staticmethod(DORN_nyu_nohints.ord_decode)
DORN_sinkhorn_opt.get_metrics = staticmethod(DORN_nyu_nohints.get_metrics)

if __name__ == "__main__":
    import os
    from time import perf_counter
    import numpy as np
    from torch.utils.data import DataLoader
    from utils.train_utils import init_randomness
    from models.data.nyuv2_official_hints_sid_dataset import load_data, cfg
    from models.data.utils.spad_utils import cfg as spad_cfg
    data_config = cfg()
    spad_config = spad_cfg()
    spad_config["dc_count"] = 0.
    spad_config["use_albedo"] = False
    spad_config["use_squared_falloff"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(config)
    # print(spad_config)
    del data_config["data_name"]
    model = DORN_sinkhorn_opt(sgd_iters=1,
                              sinkhorn_iters=1,
                              use_albedo=spad_config["use_albedo"],
                              use_squared_falloff=spad_config["use_squared_falloff"])
    model.to(device)
    _, _, test = load_data(**data_config, spad_config=spad_config)

    dataloader = DataLoader(test)
    start = perf_counter()
    init_randomness(0)
    input_ = test.get_item_by_id("living_room_0059/1591")
    for key in ["rgb", "albedo", "rawdepth", "spad", "mask", "rawdepth_orig", "mask_orig"]:
        input_[key] = input_[key].unsqueeze(0)
    data_load_time = perf_counter() - start
    print("dataloader: {}".format(data_load_time))
    # print(input_["entry"])
    # print(model.hints_extractor[0].weight)
    pred, metrics = model.evaluate(input_, "cuda")
    # print(before_metrics)
    print(metrics)
    # print(model.sid_obj)
