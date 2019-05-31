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
from models.loss import delta, mse, rmse, rel_abs_diff, rel_sqr_diff, log10

from models.DenseDepth.utils import evaluate, predict, scale_up
from models.DenseDepth.model import create_model

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



if __name__ == "__main__":
    from models.data.nyuv2_test_split_dataset import cfg, load_data
    data_config = cfg()
    if "data_name" in data_config:
        del data_config["data_name"]
    test = load_data(**data_config)
    # print(test[0])

    # Try it out
    densedepth = DenseDepthMedianRescaling()
    pred, metrics = densedepth.evaluate(test[0])
    print(np.max(pred))
    print(np.min(pred))
    print(metrics)
