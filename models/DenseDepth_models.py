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
from models.data.data_utils.sid_utils import SID
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
    def __init__(self, existing=os.path.join("models", "nyu.h5"), crop=(20, 460,  24, 616)):
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

    def forward(self, rgb):
        """
        Works in numpy.
        """
        pred = scale_up(2, predict(self.model, rgb/255,
                                   minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_flip = scale_up(2, predict(self.model, rgb[...,::-1,:]/255,
                                        minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
        pred_final = 0.5*pred + 0.5*pred_flip[:,:,::-1]
        return pred_final

    def predict(self, rgb):
        return self.forward(rgb)

    def evaluate(self, rgb, crop, gt, mask):
        """
        Works in numpy, but returns a torch tensor prediction.
        :param rgb: N x H x W x C in RGB order (not BGR)
        :param crop: length-4 array with crop pixel coordinates
        :param gt: N x H x W x C
        :return: torch tensor prediction, metrics dict, and number of valid pixels
        """
        pred = self.predict(rgb.numpy())
        pred = pred[:, crop[0]:crop[1], crop[2]:crop[3]]
        pred = torch.from_numpy(pred).cpu().unsqueeze(0).float()
        metrics = self.get_metrics(pred, gt, mask)
        return pred, metrics, torch.sum(mask).item()

    @staticmethod
    def get_metrics(pred, gt, mask):
        return get_depth_metrics(pred, gt, mask)


class DenseDepthMedianRescaling(DenseDepth):
    def __init__(self, min_depth=0., max_depth=10.,
                 existing=os.path.join("models", "nyu.h5"), crop=(20, 460,  24, 616)):
        super(DenseDepthMedianRescaling, self).__init__(existing, crop) # Initializes model as well
        self.min_depth = min_depth
        self.max_depth = max_depth

    # def forward(self, rgb, gt):
    #     """
    #     Works in numpy.
    #     """
    #     pred = scale_up(2, predict(self.model, rgb/255,
    #                                minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
    #     pred = pred*(np.median(gt)/np.median(pred))
    #     pred_flip = scale_up(2, predict(self.model, rgb[...,::-1,:]/255,
    #                                     minDepth=10, maxDepth=1000, batch_size=1)[:,:,:,0]) * 10.0
    #     pred_flip = pred_flip*(np.median(gt)/np.median(pred_flip))
    #     pred_final = 0.5*pred + 0.5*pred_flip[:,:,::-1]
    #     return pred_final

    def evaluate(self, rgb, crop, gt, rawdepth, rawdepth_mask, mask):
        """
        https://github.com/tinghuiz/SfMLearner/blob/master/kitti_eval/eval_depth.py
        -> Crop first, then scale using median
        -> Clip afterwords.
        :param rgb:
        :param crop:
        :param gt:
        :param gt_full:
        :param mask:
        :return:
        """
        pred = self.forward(rgb.numpy())
        pred = pred[...,crop[0]:crop[1], crop[2]:crop[3]]
        rawdepth = rawdepth.numpy()
        rawdepth_mask = rawdepth_mask.numpy()
        pred = np.clip(pred*(np.median(rawdepth[rawdepth_mask > 0])/np.median(pred[rawdepth_mask > 0])),
                       a_min=self.min_depth, a_max=self.max_depth)
        pred = torch.from_numpy(pred).cpu().unsqueeze(0).float()
        metrics = self.get_metrics(pred, gt, mask)
        return pred, metrics, torch.sum(mask).item()


class DenseDepthHistogramMatching(DenseDepth):
    def __init__(self, min_depth=0, max_depth=10.,
                 existing=os.path.join("models","nyu.h5"), crop=(20, 460,  24, 616)):
                 # sid_bins=68, offset=0.,
                 # alpha=0.6569154266167957, beta=9.972175646365525):
        super(DenseDepthHistogramMatching, self).__init__(existing, crop)
        self.min_depth = min_depth
        self.max_depth = max_depth
        # self.sid_obj = SID(sid_bins, alpha, beta, offset)

    def evaluate(self, rgb, crop, gt, gt_orig, mask):
        pred_init = self.forward(rgb.numpy())
        # pred_sid_index = self.sid.get_sid_index_from_value(pred_init)
        # gt_hist, _ = np.histogram(gt.cpu().numpy(), bins=self.sid.sid_bin_edges)
        pred = self.hist_match(pred_init, gt_orig)
        pred = torch.from_numpy(pred).cpu().unsqueeze(0).float()
        pred = pred[...,crop[0]:crop[1], crop[2]:crop[3]]
        metrics = self.get_metrics(pred, gt, mask)
        return pred, metrics, torch.sum(mask).item()

    @staticmethod
    def hist_match(source, template):
        """
        From https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

        Adjust the pixel values of a grayscale image such that its histogram
        matches that of a target image

        Arguments:
        -----------
            source: np.ndarray
                Image to transform; the histogram is computed over the flattened
                array
            template: np.ndarray
                Template image; can have different dimensions to source
        Returns:
        -----------
            matched: np.ndarray
                The transformed output image
        """

        oldshape = source.shape
        source = source.ravel()
        template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        t_values, t_counts = np.unique(template, return_counts=True)

        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float64)
        s_quantiles /= s_quantiles[-1]
        t_quantiles = np.cumsum(t_counts).astype(np.float64)
        t_quantiles /= t_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

        return interp_t_values[bin_idx].reshape(oldshape)


class DenseDepthHistogramMatchingWasserstein(DenseDepth):
    def __init__(self, sgd_iters=100, sinkhorn_iters=40, sigma=0.5, lam=1e1, kde_eps=1e-4,
                 sinkhorn_eps=1e-7, dc_eps=1e-5,
                 lr=1e5, min_depth=0., max_depth=10.,
                 sid_bins=68, offset=0.,
                 alpha=0.6569154266167957, beta=9.972175646365525,
                 existing=os.path.join("models", "nyu.h5"),
                 crop=(20, 460,  24, 616)):
        super(DenseDepthHistogramMatchingWasserstein, self).__init__(existing, crop)
        self.sinkhorn_opt = SinkhornOpt(sgd_iters=sgd_iters, sinkhorn_iters=sinkhorn_iters, sigma=sigma,
                                        lam=lam, kde_eps=kde_eps,
                                        sinkhorn_eps=sinkhorn_eps, dc_eps=dc_eps,
                                        remove_dc=False, use_intensity=False, use_squared_falloff=False,
                                        lr=lr, min_depth=min_depth, max_depth=max_depth,
                                        sid_bins=sid_bins,
                                        alpha=alpha, beta=beta, offset=offset)

    def evaluate(self, rgb, crop, gt, mask, device):
        """

        :param rgb: N x H' x W' x C numpy array
        :param gt: N x 1 x H x W torch array
        :param mask: N x 1 x H x W torch array
        :param device:
        :return:
        """
        # Run RGB through cnn to get depth_init
        pred_init = self.predict(rgb.numpy())
        pred_init = pred_init[:, crop[0]:crop[1], crop[2]:crop[3]]

        pred_init = torch.from_numpy(pred_init).unsqueeze(0).float().to(device)
        extended_bin_edges = self.sinkhorn_opt.sid_obj.sid_bin_edges.cpu().numpy()
        extended_bin_edges = np.append(extended_bin_edges, float('inf'))
        gt_hist, _ = np.histogram(gt.cpu().numpy(), bins=extended_bin_edges)
        gt_hist = torch.from_numpy(gt_hist).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float().to(device)
        rgb = rgb.permute(0, 3, 1, 2).to(device)

        mask = mask.to(device)
        pred = self.sinkhorn_opt.optimize(pred_init, torch.flip(rgb, dims=(1,)), gt_hist, mask, device)
        pred = pred.cpu()
        gt = gt.cpu()
        mask = mask.cpu()
        metrics = self.get_metrics(pred, gt, mask)

        pred_init = pred_init.cpu()
        before_metrics = self.get_metrics(pred_init, gt, mask)
        print("before", before_metrics)
        return pred, metrics, torch.sum(mask).item()



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
