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
from models.loss import delta, mse, rmse, rel_abs_diff, rel_sqr_diff, log10

from models.DenseDepth.utils import evaluate, predict, scale_up
from models.DenseDepth.model import create_model
from models.sinkhorn_dist import optimize_depth_map_masked

class DenseDepth(Model):
    """
    DenseDepth Network

    https://github.com/ialhashim/DenseDepth

    Meant to be run as a part of a larger network.

    Only works in eval mode.

    Thin wrapper around the Keras implementation.
    """
    def __init__(self, min_depth=0., max_depth=10., existing=os.path.join("models", "nyu.h5")):
        super(Model, self).__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.model = create_model(existing)

    def predict(self, input_):
        """
        """
        rgb = input_["rgb"]
        crop = input_["crop"]
        pred = scale_up(2, predict(self.model, rgb/255,
                                   minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_flip = scale_up(2, predict(self.model, rgb[...,::-1,:]/255,
                                        minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0

        pred = pred[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_flip = pred_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]

        pred_final = 0.5*pred + 0.5*pred_flip[:,:,::-1]
        return pred_final

    def evaluate(self, input_):
        # Output full-size depth map, so set resize_output=True
        pred = self.predict(input_)
        gt = input_["depth_cropped"]
        metrics = {}
        metrics["delta1"], metrics["delta2"], metrics["delta3"], \
        metrics["abs_rel_diff"], metrics["mse"], metrics["rmse"], metrics["log10"] = self.get_metrics(gt, pred)
        return pred, metrics

    # Error computaiton based on https://github.com/tinghuiz/SfMLearner
    @staticmethod
    def get_metrics(gt, pred):
        # print(gt.shape)
        # print(pred.shape)
        thresh = np.maximum((gt / pred), (pred / gt))

        a1 = (thresh < 1.25).mean()
        a2 = (thresh < 1.25 ** 2).mean()
        a3 = (thresh < 1.25 ** 3).mean()

        abs_rel = np.mean(np.abs(gt - pred) / gt)

        mse = ((gt - pred) ** 2).mean()
        rmse = np.sqrt(mse)

        log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
        return a1, a2, a3, abs_rel, mse, rmse, log_10


class DenseDepthMedianRescaling(DenseDepth):
    def predict(self, input_):

        rgb = input_["rgb"]
        crop = input_["crop"]
        pred = scale_up(2, predict(self.model, rgb/255,
                                   minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_flip = scale_up(2, predict(self.model, rgb[...,::-1,:]/255,
                                        minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0

        pred = pred[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_flip = pred_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]

        pred_combined = 0.5*pred + 0.5*pred_flip[:,:,::-1]

        # Do median rescaling
        gt_median = np.median(input_["depth_cropped"])
        pred_median = np.median(pred_combined)
        pred_rescaled = np.clip(pred_combined * (gt_median/pred_median), a_min=self.min_depth, a_max=self.max_depth)

        return pred_rescaled


class DenseDepthSinkhornOpt(DenseDepth):
    def __init__(self, sgd_iters=250, sinkhorn_iters=40, sigma=2., lam=1e-2, kde_eps=1e-5,
                 sinkhorn_eps=1e-2, dc_eps=1e-5,
                 remove_dc=True, use_intensity=True, use_squared_falloff=True,
                 lr=1e3, min_depth=0., max_depth=10., sid_bins=68,
                 alpha=0.6569154266167957, beta=9.972175646365525, offset=0.,
                 existing=os.path.join("models", "nyu.h5")):
        super(DenseDepthSinkhornOpt, self).__init__(min_depth, max_depth, existing)
        self.sid_bins = sid_bins
        self.sid_obj = SIDTorch(sid_bins, alpha, beta, offset)

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


    def initialize(self, input_):
        """

        :param input_: Dict of numpy arrays with key "rgb" for the rgb input and "crop" for the pixel
        locations to cropy the output image at.
        :return: Depth map of per-pixel depth indices.
        """
        rgb = input_["rgb"].numpy() # Use uncropped version as input to the network.
        crop = input_["crop"][0,:].numpy()
        print(crop.shape)
        pred = scale_up(2, predict(self.model, rgb/255,
                                   minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_flip = scale_up(2, predict(self.model, rgb[...,::-1,:]/255,
                                        minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0

        pred = pred[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_flip = pred_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_combined = 0.5*pred + 0.5*pred_flip[:,:,::-1]

        # Convert to pytorch
        pred_torch = torch.from_numpy(pred_combined)
        print(pred_torch.shape)

        # Apply discretization
        pred_torch_index = self.sid_obj.get_sid_index_from_value(pred_torch)
        print(torch.min(pred_torch_index))
        print(torch.max(pred_torch_index))
        # Convert back to numpy

        return pred_torch_index

    def evaluate(self, input_, device):
        """

        :param input_: Inputs (in numpy) to the network.
        :param device: Device to run the torch optimization on.
        :return: pred (numpy array), metrics (dict of metric-value pairs)
        """
        # Run RGB through DORN
        depth_init = self.initialize(input_) # Already resized properly. Array of depth indices.
        # DC Check
        denoised_spad = input_["spad"].to(device)
        if self.remove_dc:
            # bin_widths = (self.sid_obj.sid_bin_edges[1:] - self.sid_obj.sid_bin_edges[:-1]).cpu().numpy()
            bin_edges = self.sid_obj.sid_bin_edges.cpu().numpy().squeeze()
            denoised_spad = torch.from_numpy(remove_dc_from_spad(denoised_spad.squeeze(-1).squeeze(-1).cpu().numpy(),
                                                                 bin_edges)).unsqueeze(-1).unsqueeze(-1).to(device)
            denoised_spad[denoised_spad < self.dc_eps] = self.dc_eps
        # Normalize to 1
        denoised_spad = denoised_spad / torch.sum(denoised_spad, dim=1, keepdim=True)
        # Scaling check
        scaling = None
        if self.use_intensity:
            # intensity = input_["albedo_orig"][:, 1:2, ...].to(device) / 255.
            scaling = bgr2gray(input_["rgb_cropped_orig"])
        # Squared depth check
        inv_squared_depths = None
        if self.use_squared_falloff:
            inv_squared_depths = (self.sid_obj.sid_bin_values[:68]**(-2)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(device)

        with torch.enable_grad():
            depth_index_final, depth_hist_final = \
                optimize_depth_map_masked(depth_init, input_["mask_orig"], sigma=self.sigma, n_bins=self.sid_bins,
                                          cost_mat=self.cost_mat, lam=self.lam, gt_hist=denoised_spad,
                                          lr=self.lr, num_sgd_iters=self.sgd_iters, num_sinkhorn_iters=self.sinkhorn_iters,
                                          kde_eps=self.kde_eps,
                                          sinkhorn_eps=self.sinkhorn_eps,
                                          inv_squared_depths=inv_squared_depths,
                                          scaling=scaling)
            depth_index_final = torch.round(depth_index_final).detach().long().cpu()

            # Get depth maps and compute metrics
            pred = self.sid_obj.get_value_from_sid_index(depth_index_final)
            # Note: align_corners=False gives same behavior as cv2.resize

        # compute metrics
        gt = input_["depth_cropped_orig"].cpu().numpy()
        mask = input_["mask_orig"].cpu().numpy()
        metrics = self.get_metrics(pred, gt, mask)

        # Also compute initial metrics:
        before_metrics = self.get_metrics(depth_init.cpu().numpy(), gt, mask)
        print("before", before_metrics)
        return pred.cpu().numpy(), metrics


if __name__ == "__main__":
    from models.data.nyuv2_test_split_dataset_hints_sid import cfg, load_data
    from models.data.utils.spad_utils import cfg as get_spad_config
    from torch.utils.data._utils.collate import default_collate
    data_config = cfg()
    spad_config = get_spad_config()
    if "data_name" in data_config:
        del data_config["data_name"]
    print(data_config)
    test = load_data(**data_config, spad_config=spad_config)
    # print(test[0])
    # print(test[0]["rgb"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Try it out
    # densedepth = DenseDepthMedianRescaling()
    densedepth = DenseDepthSinkhornOpt()
    pred, metrics = densedepth.evaluate(default_collate([test[0]]), device)
    print(np.max(pred))
    print(np.min(pred))
    print(metrics)
