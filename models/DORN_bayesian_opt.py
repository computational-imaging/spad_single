import torch
import torch.nn as nn
from torch.nn import MSELoss
import numpy as np
from models.data.utils.sid_utils import SIDTorch
from torch.optim import SGD

import os

# from models.core.model_core import Model

from .DORN_nohints import DORN_nyu_nohints

class DORN_bayesian_opt:
    """
    Performs SGD to optimize the depth map further after being given an initial depth map
    estimate from DORN.
    """
    def __init__(self, sgd_iters=20, lr=1e-3, hints_len=68, spad_weight=1.,
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
        self.feature_extractor.eval()

        self.sid_obj = SIDTorch(sid_bins, alpha, beta, offset)
        self.one_over_depth_squared = 1./(SIDTorch.sid_bin_values ** 2)
        self.one_over_depth_squared.requires_grad = False


    def initialize(self, input_, device):
        """Feed rgb through DORN
        :return depth_init: Per-pixel vector of log-probabilities.
        """
        rgb = input_["rgb"].to(device)
        with torch.no_grad():
            x = self.feature_extractor(rgb)
            x, _ = self.to_logprobs(x)
        return x

    def optimize_depth_map(self, log_depth_probs_init, input_, device):
        """Perform SGD on the initial input to get the refined input"""
        gt_hist = input_["spad"].to(device)
        albedo = input_["albedo"].to(device)
        N, C, W, H = log_depth_probs_init.size()
        albedo_expanded = albedo.expand(N, C, W, H)
        albedo_expanded.requires_grad = False
        log_probs = torch.tensor(log_depth_probs_init, requires_grad=True)
        optimizer = SGD([log_probs], lr=self.lr)
        for it in range(self.sgd_iters):
            output_hist = self.spad_forward(log_probs, albedo_expanded)
            loss_val = self.loss(gt_hist, output_hist)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
        return log_probs

    def spad_forward(self, log_probs, albedo_expanded):
        """Perform the forward simulation of a model"""
        N, _, W, H = log_probs.size()
        probs = torch.exp(log_probs)
        probs = probs[:, :-1, :, :] - probs[:, 1:, :, :]
        probs = torch.cat([torch.zeros((N, 1, W, H)), probs], dim=1)
        # Use probs to get the weighted sum of albedo/depth^2
        weights = probs * self.one_over_depth_squared * albedo_expanded
        simulated_spad = torch.sum(weights, dim=(2, 3), keepdim=True)
        return simulated_spad



DORN_bayesian_opt.to_logprobs = staticmethod(DORN_nyu_nohints.to_logprobs)
