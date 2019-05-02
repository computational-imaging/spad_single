import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MSELoss
import numpy as np
from models.data.utils.sid_utils import SIDTorch
from models.sinkhorn_dist import optimize_depth_map
from models.data.utils.spad_utils import remove_dc_from_spad
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
    def __init__(self, sgd_iters=250, sinkhorn_iters=40, sigma=2., lam=1e-2, kde_eps=1e-5,
                 sinkhorn_eps = 1e-2,
                 remove_dc=True, use_albedo=True, use_squared_falloff=True,
                 lr=1e3, hints_len=68,
                 in_channels=3, in_height=257, in_width=353,
                 sid_bins=68, offset=0.,
                 min_depth=0., max_depth=10.,
                 alpha=0.6569154266167957, beta=9.972175646365525,
                 frozen=True, pretrained=True,
                 state_dict_file=os.path.join("models", "torch_params_nyuv2_BGR.pth.tar")):
        """

        :param sgd_iters: Number of iters to optimize depth map for.
        :param sinkhorn_iters: Number of iterations to run sinkhorn per sgd iteration
        :param sigma: Controls width of gaussian for kernel density estimation
        :param lam: sinkhorn iteration parameter
        :param kde_eps: Epsilon for kernel density estimation, involved in setting floor for the kernel.
        :param sinkhorn_eps: Epsilon for sinkhorn iterations, controls convergence.
        :param remove_dc: Whether or not to remove any dc component in the spad histogram before denoising.
        :param use_albedo: Whether or not to use albedo in kernel density estimation
        :param use_squared_falloff: Whether or not to use squared falloff in kernel density estimation
        :param lr: Learning rate of sgd for optimizing the depth map
        :param hints_len: Length of input hints histogram
        :param in_channels: Number of input channels (input is NxCxHxW)
        :param in_height: Input height dimension
        :param in_width: Input width dimension
        :param sid_bins: Number of sid bins to output
        :param offset: Parameter for SID object
        :param min_depth: Minimum depth of a pixel to output (for purpose of displaying images)
        :param max_depth: Maximum depth of a pixel to output (for purpose of displaying images)
        :param alpha: Parameter for SID object
        :param beta: Parameter for SID object
        :param frozen: Whether or not to freeze the DORN feature extractor
        :param pretrained: Whether or not to use the pretrained DORN model
        :param state_dict_file: The file to load the pretrained DORN model from
        """
        self.sgd_iters = sgd_iters
        self.sinkhorn_iters = sinkhorn_iters
        self.sigma = sigma
        self.lam = lam
        self.kde_eps = kde_eps
        self.sinkhorn_eps = sinkhorn_eps

        # Define cost matrix for optimal transport problem
        C = np.array([[(i - j)**2 for j in range(sid_bins)] for i in range(sid_bins)])/1000.
        self.cost_mat = torch.from_numpy(C).float()

        self.remove_dc = remove_dc
        self.use_albedo = use_albedo
        self.use_squared_falloff = use_squared_falloff
        self.lr = lr
        # self.loss.to(device)
        self.min_depth = min_depth
        self.max_depth = max_depth
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
        # print(depth_index[:,:,100:105,100:105])
        return depth_index


    # def optimize_depth_map(self, depth_hist, input_, device, resize_output=False):
    # def optimize_depth_map(self, depth_index_init, input_, device, resize_output=False):
    #     spad_hist = input_["spad"].to(device)
    #     # print(spad_hist)
    #     depth_index_final, depth_img_final, depth_hist_final = \
    #         optimize_depth_map(depth_index_init, self.sigma, self.sid_bins,
    #                            self.cost_mat, self.lam, spad_hist,
    #                            self.lr, self.sgd_iters, self.sinkhorn_iters)
    #     depth_index_final = depth_index_final.detach().long()
    #
    #     # Get depth maps and compute metrics
    #     depth_pred = self.sid_obj.get_value_from_sid_index(depth_index_final)
    #     if resize_output:
    #         original_size = input_["rgb_orig"].size()[-2:]
    #         # Note: align_corners=False gives same behavior as cv2.resize
    #         depth_pred = F.interpolate(depth_pred, size=original_size,
    #                                    mode="bilinear", align_corners=False)
    #         # print("resized")
    #     return depth_pred


    def evaluate(self, input_, device):
        # Run RGB through DORN
        depth_init = self.initialize(input_, device)
        # DC Check
        denoised_spad = input_["spad"].to(device)
        if self.remove_dc:
            bin_widths = (self.sid_obj.sid_bin_edges[1:] - self.sid_obj.sid_bin_edges[:-1]).cpu().numpy()
            denoised_spad = torch.from_numpy(remove_dc_from_spad(denoised_spad.squeeze(-1).squeeze(-1).cpu().numpy(),
                                                                 bin_widths)).unsqueeze(-1).unsqueeze(-1).to(device)
        # Albedo check
        albedo = None
        if self.use_albedo:
            albedo = input_["albedo"][:, 1:2, ...].to(device) / 255.

        # Squared depth check
        inv_squared_depths = None
        if self.use_squared_falloff:
            inv_squared_depths = (self.sid_obj.sid_bin_values[:68]**(-2)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)

        # Actually run the optimization
        with torch.enable_grad():
            depth_index_final, depth_img_final, depth_hist_final = \
                optimize_depth_map(depth_init, self.sigma, self.sid_bins,
                                   self.cost_mat, self.lam, denoised_spad,
                                   self.lr, self.sgd_iters, self.sinkhorn_iters,
                                   kde_eps=self.kde_eps,
                                   sinkhorn_eps=self.sinkhorn_eps,
                                   inv_squared_depths=inv_squared_depths,
                                   albedo=albedo)
            depth_index_final = depth_index_final.detach().long()

            # Get depth maps and compute metrics
            depth_pred = self.sid_obj.get_value_from_sid_index(depth_index_final)
            original_size = input_["rgb_orig"].size()[-2:]
            # Note: align_corners=False gives same behavior as cv2.resize
            pred = F.interpolate(depth_pred, size=original_size,
                                 mode="bilinear", align_corners=False)

        # compute metrics
        gt = input_["rawdepth_orig"].to(device)
        mask = input_["mask_orig"].to(device)
        metrics = self.get_metrics(pred, gt, mask)
        # print("after", metrics)

        # Also compute initial metrics:
        # gt_dorn = input_["rawdepth"].to(device)
        # mask_dorn = input_["mask"].to(device)
        # depth_init_map = self.sid_obj.get_value_from_sid_index(depth_init)
        # before_metrics = self.get_metrics(depth_init_map, gt_dorn, mask_dorn)
        # print("before", before_metrics)

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
