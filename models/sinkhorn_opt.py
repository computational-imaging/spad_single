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
from models.data.data_utils.sid_utils import SIDTorch
from models.data.data_utils.spad_utils import remove_dc_from_spad, bgr2gray
from models.loss import get_depth_metrics
from utils.inspect_results import add_hist_plot, log_single_gray_img

from models.DORN_nohints import DORN_nyu_nohints
from models.sinkhorn_dist import optimize_depth_map_masked

class SinkhornOpt:
    """
    Takes as input an initial depth image, an intensity image, and returns the rescaled depth.
    Operates in pytorch.
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
        self.inv_squared_depths = None
        if self.use_squared_falloff:
            self.inv_squared_depths = (self.sid_obj.sid_bin_values[:self.sid_bins + 1] ** (-2)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        C = np.array([[(self.sid_obj.sid_bin_values[i] - self.sid_obj.sid_bin_values[j]).item()**2 for i in range(sid_bins+1)]
                                                                                                   for j in range(sid_bins+1)])
        self.cost_mat = torch.from_numpy(C).float()

        self.optimize_depth_map = lambda depth_index_init, mask, spad, scaling, gt: \
            optimize_depth_map_masked(depth_index_init, mask, sigma=self.sigma, n_bins=self.sid_bins,
                                      cost_mat=self.cost_mat, lam=self.lam, gt_hist=spad,
                                      lr=self.lr, num_sgd_iters=self.sgd_iters,
                                      num_sinkhorn_iters=self.sinkhorn_iters,
                                      kde_eps=self.kde_eps,
                                      sinkhorn_eps=self.sinkhorn_eps,
                                      inv_squared_depths=self.inv_squared_depths,
                                      scaling=scaling, writer=self.writer, gt=gt,
                                      model=self)

        self.writer = None

    def to(self, device):
        self.sid_obj.to(device)
        self.cost_mat = self.cost_mat.to(device)
        if self.inv_squared_depths is not None:
            self.inv_squared_depths = self.inv_squared_depths.to(device)

    def optimize(self, depth_init, bgr, spad, mask, gt=None):
        """
        Works in pytorch.
        :param depth_init:
        :param bgr:
        :param spad:
        :param mask:
        :param gt:
        :return:
        """
        scaling = None
        if self.use_intensity:
            scaling = bgr2gray(bgr)/255.
            if self.writer is not None:
                log_single_gray_img(self.writer, "img/intensity", scaling, 0., 1.)

        depth_index_init = self.sid_obj.get_sid_index_from_value(depth_init)
        with torch.enable_grad():
            depth_index_final, _ = self.optimize_depth_map(depth_index_init, mask, spad, scaling, gt)
        depth_index_final = torch.floor(depth_index_final).detach().long()
        pred = self.sid_obj.get_value_from_sid_index(depth_index_final)
        return pred

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

        if self.remove_dc:
            spad = self.remove_dc_from_spad_torch(spad)
            if self.writer is not None:
                add_hist_plot(self.writer, "hist/spad_no_noise", spad)

        spad = spad / torch.sum(spad, dim=1, keepdim=True)

        if self.writer is not None:
            import torchvision.utils as vutils
            log_single_gray_img(self.writer, "depth/pred_init", depth_init, self.min_depth, self.max_depth)
            log_single_gray_img(self.writer, "depth/gt", gt, self.min_depth, self.max_depth)
            log_single_gray_img(self.writer, "img/mask", mask, 0., 1.)
            rgb_img = vutils.make_grid(torch.flip(bgr, dims=(1,)) / 255, nrow=1)
            self.writer.add_image("img/rgb", rgb_img, 0)

        # Do the optimization
        pred = self.optimize(depth_init, bgr, spad, mask, device, gt=gt)

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


class SinkhornOptFull:
    """
    Just the Sinkhorn Opt class with a built-in initializer.
    """
    def __init__(self, initializer, sinkhorn_optimizer):
        """

        :param initializer: CNN with a predict(rgb, device) method that can be used to get a prediction.
        :param sinkhorn_optimizer:
        """
        self.initializer = initializer
        self.sinkhorn_opt = sinkhorn_optimizer

    def evaluate(self, rgb, depth, spad, mask, device):
        # Run RGB through cnn to get depth_init
        depth_init = self.initializer.predict(rgb, device)
        # Run sinkhorn optimizer
        pred = self.sinkhorn_opt.optimize(depth_init, spad, rgb, mask, device)

        # Report metrics
        metrics = self.get_metrics(pred, depth, mask)

        # Report initial metrics
        before_metrics = self.get_metrics(depth_init, depth, mask)
        print("before", before_metrics)
        return pred, metrics

    @staticmethod
    def get_metrics(pred, gt, mask):
        return get_depth_metrics(pred, gt, mask)

