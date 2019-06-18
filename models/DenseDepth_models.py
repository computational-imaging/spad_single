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
from utils.inspect_results import add_hist_plot, log_single_gray_img

from models.DORN_nohints import DORN_nyu_nohints
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

    def predict(self, rgb, crop):
        """
        Works in numpy.
        """
        pred = scale_up(2, predict(self.model, rgb/255,
                                   minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_flip = scale_up(2, predict(self.model, rgb[...,::-1,:]/255,
                                        minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0

        pred = pred[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_flip = pred_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]

        pred_final = 0.5*pred + 0.5*pred_flip[:,:,::-1]
        return pred_final

    def evaluate(self, rgb, crop, gt):
        """
        Works in numpy, but returns a torch tensor prediction.
        :param rgb: N x H x W x C in RGB order (not BGR)
        :param crop: length-4 array with crop pixel coordinates
        :param gt: N x H x W x C
        :return: torch tensor prediction and metrics dict
        """
        # Output full-size depth map, so set resize_output=True
        pred = self.predict(rgb, crop)
        metrics = {}
        metrics["delta1"], metrics["delta2"], metrics["delta3"], \
        metrics["abs_rel_diff"], metrics["mse"], metrics["rmse"], metrics["log10"] = self.get_metrics(gt, pred)
        return torch.from_numpy(pred).float(), metrics

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

    def initialize(self, rgb, crop):
        """

        :param input_: Dict of numpy arrays with key "rgb" for the rgb input and "crop" for the pixel
        locations to cropy the output image at.
        :return: Depth map of per-pixel depth indices.
        """
        rgb = rgb.numpy() # Use uncropped version as input to the network.
        crop = crop[0,:].numpy()
        pred = scale_up(2, predict(self.model, rgb/255,
                                   minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_flip = scale_up(2, predict(self.model, rgb[...,::-1,:]/255,
                                        minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0

        # Test-time augmentation
        pred = pred[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_flip = pred_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_combined = 0.5*pred + 0.5*pred_flip[:,:,::-1]

        # Convert to pytorch
        pred_torch = torch.from_numpy(pred_combined).unsqueeze(0)
        print("pred_torch", pred_torch.shape)
        return pred_torch

    def evaluate(self, rgb, rgb_cropped, crop, spad, mask_cropped, gt, device):
        # Run RGB through DORN
        depth_init = self.initialize(rgb, crop) # rgb is uncropped
        # Move to another GPU for pytorch part...
        # os.environ["CUDA_VISIBLE_DEVICES"] = torch_cuda_device
        # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
        # rgb = rgb.to(device)
        depth_init = depth_init.to(device)
        rgb_cropped = rgb_cropped.to(device)
        spad = spad.to(device)
        mask_cropped = mask_cropped.to(device)
        self.sid_obj.to(device)
        self.cost_mat = self.cost_mat.to(device)


        # DC Check
        if self.writer is not None:
            add_hist_plot(self.writer, "hist/raw_spad", spad)
        denoised_spad = spad
        if self.remove_dc:
            # bin_widths = (self.sid_obj.sid_bin_edges[1:] - self.sid_obj.sid_bin_edges[:-1]).cpu().numpy()
            bin_edges = self.sid_obj.sid_bin_edges.cpu().numpy().squeeze()
            denoised_spad = torch.from_numpy(remove_dc_from_spad(denoised_spad.squeeze(-1).squeeze(-1).cpu().numpy(),
                                                                 bin_edges,
                                                                 self.max_depth)).unsqueeze(-1).unsqueeze(-1)
            denoised_spad[denoised_spad < self.dc_eps] = self.dc_eps
        # Normalize to 1
        denoised_spad = denoised_spad / torch.sum(denoised_spad, dim=1, keepdim=True)
        denoised_spad = denoised_spad.to(device)
        if self.writer is not None:
            add_hist_plot(self.writer, "hist/spad_no_noise", denoised_spad)
        # Scaling check
        scaling = None
        if self.use_intensity:
            # intensity = input_["albedo_orig"][:, 1:2, ...].to(device) / 255.
            scaling = bgr2gray(rgb_cropped)/255.
        # Squared depth check
        inv_squared_depths = None
        if self.use_squared_falloff:
            inv_squared_depths = (self.sid_obj.sid_bin_values[:self.sid_bins+1]**(-2)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        with torch.enable_grad():
            depth_index_init = self.sid_obj.get_sid_index_from_value(depth_init)
            print("max depth index:", torch.max(depth_index_init))
            print("min depth index:", torch.min(depth_index_init))
            depth_index_final, depth_hist_final = \
                optimize_depth_map_masked(depth_index_init, mask_cropped, sigma=self.sigma, n_bins=self.sid_bins,
                                          cost_mat=self.cost_mat, lam=self.lam, gt_hist=denoised_spad,
                                          lr=self.lr, num_sgd_iters=self.sgd_iters, num_sinkhorn_iters=self.sinkhorn_iters,
                                          kde_eps=self.kde_eps,
                                          sinkhorn_eps=self.sinkhorn_eps,
                                          min_sgd_iters=100,
                                          inv_squared_depths=inv_squared_depths,
                                          scaling=scaling, writer=self.writer, gt=gt,
                                          model=self)
            depth_index_final = torch.floor(depth_index_final).detach().long()
            # depth_index_final = torch.round(depth_index_final).detach().long()


            # Get depth maps and compute metrics
            pred = self.sid_obj.get_value_from_sid_index(depth_index_final)
            # Note: align_corners=False gives same behavior as cv2.resize

        # compute metrics
        pred = pred.cpu()
        gt = gt.cpu()
        mask = mask_cropped.cpu()
        metrics = self.get_metrics(pred, gt, mask)


        if self.writer is not None:
            import torchvision.utils as vutils
            log_single_gray_img(self.writer, "depth/pred_init", depth_init, self.min_depth, self.max_depth)
            log_single_gray_img(self.writer, "depth/gt", gt, self.min_depth, self.max_depth)
            log_single_gray_img(self.writer, "depth/pred", pred, self.min_depth, self.max_depth)
            log_single_gray_img(self.writer, "img/mask", mask, 0., 1.)
            if scaling is not None:
                # print("min scaling", torch.min(scaling))
                # print("max scaling", torch.max(scaling))
                log_single_gray_img(self.writer, "img/intensity", scaling, 0., 1.)
            rgb_img = vutils.make_grid(rgb_cropped/ 255, nrow=1)
            self.writer.add_image("img/rgb", rgb_img, 0)

        # Also compute initial metrics:
        before_metrics = self.get_metrics(depth_init.float().cpu(), gt, mask)
        print("before", before_metrics)
        return pred, metrics

DenseDepthSinkhornOpt.get_metrics = staticmethod(DORN_nyu_nohints.get_metrics)


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
