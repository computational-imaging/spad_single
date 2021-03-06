import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import skimage

import numpy as np
from time import perf_counter

import matplotlib.pyplot as plt
plt.switch_backend('agg')
from mpl_toolkits.mplot3d import Axes3D

from models.core.checkpoint import safe_makedir
from models.core.model_core import Model
from models.DORN_nohints import DORN_nyu_nohints
from models.data.data_utils.sid_utils import SIDTorch
from models.pytorch_prototyping.pytorch_prototyping import Unet

class DORN_median_matching(DORN_nyu_nohints):
    def __init__(self, in_channels=3, in_height=257, in_width=353,
                 sid_bins=68, offset=0.,
                 min_depth=0., max_depth=10.,
                 alpha=0.6569154266167957, beta=9.972175646365525,
                 frozen=True, pretrained=True,
                 state_dict_file=os.path.join("models", "torch_params_nyuv2_BGR.pth.tar")):
        """
        :param hints_len: Uniformly spaced noisy depth hints (i.e. raw SPAD data)
        :param num_hints_layers: The number of layers for performing upsampling
        """
        super(DORN_median_matching, self).__init__(in_channels, in_height, in_width,
                                                   sid_bins, offset,
                                                   min_depth, max_depth,
                                                   alpha, beta,
                                                   frozen, pretrained,
                                                   state_dict_file)

    def predict(self, bgr, bgr_orig, gt_orig, device, resize_output=True):
        depth_pred = self.forward(bgr)
        logprobs = self.to_logprobs(depth_pred)
        if resize_output:
            original_size = bgr_orig.size()[-2:]
            # Note: align_corners=False gives same behavior as cv2.resize
            depth_pred_full = F.interpolate(depth_pred, size=original_size,
                                            mode="bilinear", align_corners=False)
            logprobs_full = self.to_logprobs(depth_pred_full)
            pred_init = self.ord_decode(logprobs_full, self.sid_obj)
        else:
            pred_init = self.ord_decode(logprobs, self.sid_obj)
        pred = pred_init * (torch.median(gt_orig)/torch.median(pred_init))
        return pred

    def evaluate(self, bgr, bgr_orig, crop, gt, gt_orig, mask, device):
        pred = self.predict(bgr, bgr_orig, gt_orig, device, resize_output=True)
        pred = pred[..., crop[0]:crop[1], crop[2]:crop[3]]
        pred = pred.cpu()
        gt = gt.cpu()
        mask = mask.cpu()
        metrics = self.get_metrics(pred, gt, mask)
        return pred, metrics, torch.sum(mask).item()
