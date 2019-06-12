import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils as vutils
import numpy as np

from time import perf_counter
import os
from collections import defaultdict
from time import perf_counter

from models.core.checkpoint import safe_makedir
from models.core.model_core import Model
from models.data.utils.sid_utils import SIDTorch
from models.data.utils.spad_utils import remove_dc_from_spad, bgr2gray
from models.loss import get_depth_metrics
from utils.inspect_results import add_hist_plot, log_single_gray_img

from models.DORN_nohints import DORN_nyu_nohints
from models.sinkhorn_dist import optimize_depth_map_masked

class SinkhornOpt:
    """
    Takes as input an initial depth image, an intensity image, and returns the rescaled depth.
    """
    def __init__(self, sgd_iters=250, sinkhorn_iters=40, sigma=2., lam=1e-2, kde_eps=1e-5,
                 sinkhorn_eps=1e-2, dc_eps=1e-5,
                 remove_dc=True, use_intensity=True, use_squared_falloff=True,
                 lr=1e3, min_depth=0., max_depth=10., sid_bins=68,
                 alpha=0.6569154266167957, beta=9.972175646365525, offset=0):
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.sid_bins = sid_bins
        self.sgd_iters = sgd_iters
        self.sinkhorn_iters = sinkhorn_iters
        self.sigma = sigma
        self.lam = lam
        self.kde_eps = kde_eps
        self.sinkhorn_eps = sinkhorn_eps
        self.dc_eps = dc_eps
        self.remove_dc = remove_dc
        self.use_intensity = use_intensity
        self.use_squared_falloff = use_squared_falloff
        self.lr = lr
        self.sid_obj = SIDTorch(sid_bins, alpha, beta, offset)
        self.one_over_depth_squared = 1./(self.sid_obj.sid_bin_values[:-2] ** 2)
        self.one_over_depth_squared.requires_grad = False
        C = np.array([[(self.sid_obj.sid_bin_values[i] - self.sid_obj.sid_bin_values[j]).item()**2 for i in range(sid_bins+1)]
                                                                                                   for j in range(sid_bins+1)])
        self.cost_mat = torch.from_numpy(C).float()
        self.writer = None

    def to(self, device):
        self.sid_obj.to(device)

    def evaluate(self, depth_init, bgr, spad, mask, gt, device):
        """
        :param depth_init: Full size depth map to initialize the model.
        :param bgr: Color image, given in BGR channel order.
        :param spad: Input SPAD data
        :param mask: Masks off areas with invalid gt depth
        :param gt: Ground truth depth map
        :param device: Device to run the torch stuff on.
        :return: pred, metrics - the refined depth map and the metrics calculated for it.
        """
        # DC Check
        if self.writer is not None:
            add_hist_plot(self.writer, "hist/raw_spad", spad)

        denoised_spad = spad
        if self.remove_dc:
            denoised_spad = self.remove_dc_from_spad_torch(denoised_spad)
        denoised_spad = denoised_spad / torch.sum(denoised_spad, dim=1, keepdim=True)

        # Scaling check
        scaling = None
        if self.use_intensity:
            scaling = bgr2gray(bgr)/255.
        # Squared depth check
        inv_squared_depths = None
        if self.use_squared_falloff:
            inv_squared_depths = (self.sid_obj.sid_bin_values[:self.sid_bins + 1] ** (-2)).unsqueeze(0).unsqueeze(
                -1).unsqueeze(-1)

        if self.writer is not None:
            import torchvision.utils as vutils
            log_single_gray_img(self.writer, "depth/pred_init", depth_init, self.min_depth, self.max_depth)
            log_single_gray_img(self.writer, "depth/gt", gt, self.min_depth, self.max_depth)
            log_single_gray_img(self.writer, "img/mask", mask, 0., 1.)
            if scaling is not None:
                print("min scaling", torch.min(scaling))
                print("max scaling", torch.max(scaling))
                log_single_gray_img(self.writer, "img/intensity", scaling, 0., 1.)
            rgb_img = vutils.make_grid(torch.flip(bgr, dims=(1,)) / 255, nrow=1)
            self.writer.add_image("img/rgb", rgb_img, 0)
            add_hist_plot(self.writer, "hist/spad_no_noise", denoised_spad)

        with torch.enable_grad():
            depth_index_init = self.sid_obj.get_sid_index_from_value(depth_init)
            print("max depth index:", torch.max(depth_index_init))
            print("min depth index:", torch.min(depth_index_init))
            depth_index_final, depth_hist_final = \
                optimize_depth_map_masked(depth_index_init, mask, sigma=self.sigma, n_bins=self.sid_bins,
                                          cost_mat=self.cost_mat, lam=self.lam, gt_hist=denoised_spad,
                                          lr=self.lr, num_sgd_iters=self.sgd_iters,
                                          num_sinkhorn_iters=self.sinkhorn_iters,
                                          kde_eps=self.kde_eps,
                                          sinkhorn_eps=self.sinkhorn_eps,
                                          inv_squared_depths=inv_squared_depths,
                                          scaling=scaling, writer=self.writer, gt=gt,
                                          model=self)
            depth_index_final = torch.floor(depth_index_final).detach().long()
            # depth_index_final = torch.round(depth_index_final).detach().long()

            # Get depth maps and compute metrics
            pred = self.sid_obj.get_value_from_sid_index(depth_index_final)
            # Note: align_corners=False gives same behavior as cv2.resize

        # Compute metrics
        pred = pred.cpu()
        gt = gt.cpu()
        mask = mask.cpu()
        metrics = self.get_metrics(pred, gt, mask)

        # Also compute initial metrics:
        before_metrics = self.get_metrics(depth_init.float().cpu(), gt, mask)
        print("before", before_metrics)
        return pred, metrics

    def remove_dc_from_spad_torch(self, noisy_spad_tensor, lam=1e-2, eps=1e-5):
        """
        Wrapper function for remove_dc_from_spad.
        Takes and returns a torch tensor.
        """
        bin_edges = self.sid_obj.sid_bin_edges.cpu().numpy().squeeze()
        denoised_spad = torch.from_numpy(remove_dc_from_spad(noisy_spad_tensor.squeeze(-1).squeeze(-1).cpu().numpy(),
                                                             bin_edges,
                                                             self.max_depth,
                                                             lam=lam,
                                                             eps=eps)).unsqueeze(-1).unsqueeze(-1)
        denoised_spad[denoised_spad < self.dc_eps] = self.dc_eps
        return denoised_spad

    @staticmethod
    def get_metrics(pred, truth, gt):
        return get_depth_metrics(pred, truth, gt)
