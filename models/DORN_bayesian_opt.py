import torch
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
from models.data.utils.sid_utils import SIDTorch
from torch.optim import SGD

import os

# from models.core.model_core import Model

from models.DORN_nohints import DORN_nyu_nohints

class DORN_bayesian_opt:
    """
    Performs SGD to optimize the depth map further after being given an initial depth map
    estimate from DORN.
    """
    def __init__(self, sgd_iters=1, use_albedo=True, use_squared_falloff=True, lr=1e-3, hints_len=68, spad_weight=1.,
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
        self.use_albedo = use_albedo
        self.use_squared_falloff = use_squared_falloff
        self.lr = lr
        self.loss = MSELoss()
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

    def initialize(self, input_, device):
        """Feed rgb through DORN
        :return per-pixel one-hot indicating the depth bin for that pixel.
        """
        self.feature_extractor.to(device)
        rgb = input_["rgb"].to(device)
        with torch.no_grad():
            x = self.feature_extractor(rgb)
            log_probs, _ = self.to_logprobs(x)
            depth_index = torch.sum((log_probs >= np.log(0.5)), dim=1, keepdim=True).long()
            # print(depth_index[:,:,3,3])
            depth_hist = torch.zeros_like(log_probs, device=device)
            depth_hist.scatter_(1, depth_index, 1)
            # print(depth_hist[:,:,3,3])
        return depth_hist

    def optimize_depth_map(self, depth_hist, input_, device):
        """Perform SGD on the initial input to get the refined input"""
        gt_hist = input_["spad"].to(device)
        albedo = input_["albedo"].to(device)
        N, C, W, H = depth_hist.size()

        albedo = albedo[:,1,:,:].requires_grad_(False)
        output = depth_hist.clone().detach().requires_grad_(True)
        optimizer = SGD([output], lr=self.lr)
        # self.zero_pad = torch.zeros((N, 1, W, H), requires_grad=False).to(device)
        self.one_over_depth_squared = self.one_over_depth_squared.to(device)
        # self.zero_pad.requires_grad = False
        for it in range(self.sgd_iters):
            output_hist = self.spad_forward(output, albedo)
            loss_val = self.loss(gt_hist, output_hist)
            optimizer.zero_grad()
            loss_val.backward()
            print(torch.norm(output.grad))
            optimizer.step()
        return output

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
        simulated_spad /= torch.sum(simulated_spad)
        return simulated_spad

    def evaluate(self, data, device):
        depth_hist = model.initialize(data, device)
        output_depth_probs = model.optimize_depth_map(depth_hist, data, device)
        output_dir = "."
        outfile = os.path.join(output_dir, "{}_pred_{}.png".format(input_["entry"].replace("/", "_"), sgd_iters))
        output_depth = model.ord_decode((output_depth_probs, None), model.sid_obj)

DORN_bayesian_opt.to_logprobs = staticmethod(DORN_nyu_nohints.to_logprobs)
DORN_bayesian_opt.ord_decode = staticmethod(DORN_nyu_nohints.ord_decode)


if __name__ == "__main__":
    import os
    import numpy as np
    from torch.utils.data import DataLoader
    from models.data.nyuv2_official_hints_sid_dataset import load_data, cfg
    from models.data.utils.spad_utils import cfg as spad_cfg
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt
    data_config = cfg()
    # print(data_config)
    # data_config["normalization"] = "dorn"
    spad_config = spad_cfg()
    spad_config["dc_count"] = 0.
    spad_config["use_albedo"] = False
    spad_config["use_squared_falloff"] = False
    # print(config)
    # print(spad_config)
    del data_config["data_name"]
    device = torch.device("cuda")
    _, val, _ = load_data(**data_config, spad_config=spad_config)
    input_ = val[1]
    for key in ["rgb", "albedo", "spad", "mask"]:
        input_[key] = input_[key].unsqueeze(0)
    print(input_["entry"])

    for sgd_iters in range(5):
        print("Running with {} iters of SGD...".format(sgd_iters))
        model = DORN_bayesian_opt(sgd_iters=sgd_iters, use_albedo=spad_config["use_albedo"],
                                  use_squared_falloff=spad_config["use_squared_falloff"], lr=1.)
        model.feature_extractor.to(device)
        depth_hist = model.initialize(input_, device)
        if sgd_iters == 0:
            print(depth_hist[:,:,30,30])
        output = model.optimize_depth_map(depth_hist, input_, device)
        print(output[:,:,30,30])
        output_dir = "."
        outfile = os.path.join(output_dir, "{}_pred_{}.png".format(input_["entry"].replace("/", "_"), sgd_iters))
        # output_depth = model.ord_decode((output_depth_probs, None), model.sid_obj)
        # # print(output_depth)
        #
        # vutils.save_image(output_depth, outfile, nrow=1, normalize=True, range=(model.min_depth, model.max_depth))

