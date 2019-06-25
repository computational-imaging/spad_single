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
        :param crop: default crop. Only used if no crop is provided to the predict function.
        """
        super(Model, self).__init__()
        self.model = create_model(existing)
        # from keras.models import load_model
        # from models.DenseDepth.layers import BilinearUpSampling2D
        # # Custom object needed for inference and training
        # custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': depth_loss_function}
        #
        # # Load model into GPU / CPU
        # print('Loading model...')
        # model = load_model(args.model, custom_objects=custom_objects, compile=False)
        self.default_crop = np.array(crop)

    def forward(self, rgb, device, crop=None):
        """
        Works in numpy.
        """
        if crop is None:
            crop = self.default_crop
        pred = scale_up(2, predict(self.model, rgb/255,
                                   minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_flip = scale_up(2, predict(self.model, rgb[...,::-1,:]/255,
                                        minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0

        pred = pred[:,crop[0]:crop[1], crop[2]:crop[3]]
        pred_flip = pred_flip[:,crop[0]:crop[1], crop[2]:crop[3]]

        pred_final = 0.5*pred + 0.5*pred_flip[:,:,::-1]
        return pred_final

    def predict(self, rgb, device, crop=None):
        return self.forward(rgb, device, crop)

    def evaluate(self, rgb, crop, gt, mask):
        """
        Works in numpy, but returns a torch tensor prediction.
        :param rgb: N x H x W x C in RGB order (not BGR)
        :param crop: length-4 array with crop pixel coordinates
        :param gt: N x H x W x C
        :return: torch tensor prediction, metrics dict, and number of valid pixels
        """
        pred = self.predict(rgb, device=None, crop=crop)
        pred = torch.from_numpy(pred).cpu().unsqueeze(0).float()
        metrics = self.get_metrics(pred, gt, mask)
        return pred, metrics, torch.sum(mask).item()

    @staticmethod
    def get_metrics(pred, gt, mask):
        return get_depth_metrics(pred, gt, mask)


class DenseDepthMedianRescaling(DenseDepth):
    def __init__(self, min_depth=0., max_depth=10.,
                 existing=os.path.join("models", "nyu.h5"), crop=[ 20, 460,  24, 616]):
        super(DenseDepthMedianRescaling).__init__(existing, crop) # Initializes model as well
        self.min_depth = min_depth
        self.max_depth = max_depth

    def predict(self, rgb, gt, device, crop=None):
        pred = self.forward(rgb, device, crop)
        # Do median rescaling
        gt_median = np.median(gt)
        pred_median = np.median(pred)
        pred_rescaled = np.clip(pred * (gt_median/pred_median), a_min=self.min_depth, a_max=self.max_depth)
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
    from models.data.data_utils.spad_utils import cfg as get_spad_config
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
