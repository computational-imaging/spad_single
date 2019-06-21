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
from models.sinkhorn_opt import SinkhornOptFull, SinkhornOpt

class DenseDepth(Model):
    """
    DenseDepth Network

    https://github.com/ialhashim/DenseDepth

    Meant to be run as a part of a larger network.

    Only works in eval mode.

    Thin wrapper around the Keras implementation.
    """
    def __init__(self, existing=os.path.join("models", "nyu.h5"), crop=[ 20, 460,  24, 616]):
        """

        :param existing:
        :param crop: default: Wonka crop
        """
        super(Model, self).__init__()
        self.model = create_model(existing)
        self.crop = np.array(crop)

    def predict(self, rgb, device, crop=None):
        """
        Works in numpy.
        """
        if crop is None:
            crop = self.crop
        pred = scale_up(2, predict(self.model, rgb/255,
                                   minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_flip = scale_up(2, predict(self.model, rgb[...,::-1,:]/255,
                                        minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0

        pred = pred[:,crop[0]:crop[1], crop[2]:crop[3]]
        pred_flip = pred_flip[:,crop[0]:crop[1], crop[2]:crop[3]]

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
        pred = self.predict(rgb, device=None)
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
    def __init__(self, min_depth=0., max_depth=10.,
                 existing=os.path.join("models", "nyu.h5"), crop=[ 20, 460,  24, 616]):
        super(DenseDepthMedianRescaling, self).__init__(existing, crop) # Initializes model as well
        self.min_depth = min_depth
        self.max_depth = max_depth


    def predict(self, rgb, gt, device, crop=None):
        if crop is None:
            crop = self.crop
        pred = scale_up(2, predict(self.model, rgb/255,
                                   minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_flip = scale_up(2, predict(self.model, rgb[...,::-1,:]/255,
                                        minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0

        pred = pred[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]
        pred_flip = pred_flip[:,crop[0]:crop[1]+1, crop[2]:crop[3]+1]

        pred_combined = 0.5*pred + 0.5*pred_flip[:,:,::-1]

        # Do median rescaling
        gt_median = np.median(gt)
        pred_median = np.median(pred_combined)
        pred_rescaled = np.clip(pred_combined * (gt_median/pred_median), a_min=self.min_depth, a_max=self.max_depth)

        return pred_rescaled


class DenseDepthSinkhornOpt(SinkhornOptFull):
    def __init__(self, sgd_iters=100, sinkhorn_iters=40, sigma=0.5, lam=1e1, kde_eps=1e-4,
                 sinkhorn_eps=1e-7, dc_eps=1e-5,
                 remove_dc=True, use_intensity=True, use_squared_falloff=True,
                 lr=1e5, min_depth=0., max_depth=10.,
                 sid_bins=68, offset=0.,
                 alpha=0.6569154266167957, beta=9.972175646365525,
                 existing=os.path.join("models", "nyu.h5")):
        self.initializer = DenseDepth(existing)
        self.sinkhorn_opt = SinkhornOpt(sgd_iters=sgd_iters, sinkhorn_iters=sinkhorn_iters, sigma=sigma, lam=lam, kde_eps=kde_eps,
                                        sinkhorn_eps=sinkhorn_eps, dc_eps=dc_eps,
                                        remove_dc=remove_dc, use_intensity=use_intensity, use_squared_falloff=use_squared_falloff,
                                        lr=lr, min_depth=min_depth, max_depth=max_depth,
                                        sid_bins=sid_bins,
                                        alpha=alpha, beta=beta, offset=offset)





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
