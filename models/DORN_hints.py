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
from models.data.utils.sid_utils import SIDTorch
from models.pytorch_prototyping.pytorch_prototyping import Unet

class BayesianHints(nn.Module):
    def __init__(self, hints_len, sid_bins):
        super(BayesianHints, self).__init__()
        self.net = nn.Conv2d(hints_len, 2 * sid_bins, kernel_size=1, bias=False)

        # Initialize hints_extractor do do the bayesian thing.
        # Weight is of shape [out_channels, in_channels, kernel_size, kernel_size]
        # weight_shape = self.hints_extractor[0].weight.size()
        # A: P(l > k)
        # B: P(l <= k)
        with torch.no_grad():
            A = torch.zeros(sid_bins, hints_len, 1, 1, requires_grad=False)
            for i in range(hints_len):
                A[i, i:, 0, 0] = torch.ones(hints_len - i, requires_grad=False)
            B = 1. - A
            self.net.weight[1::2, ...] = A
            self.net.weight[::2, ...] = B
            # self.net.bias.zero_()

    def forward(self, x):
        x = self.net(x)
        return x


class BayesianHintsResNet(nn.Module):
    def __init__(self, hints_len, sid_bins):
        super(BayesianHintsResNet, self).__init__()
        self.skip = BayesianHints(hints_len, sid_bins)
        self.net = nn.Conv2d(hints_len, sid_bins, kernel_size=1)

    def forward(self, x):
        x1 = self.net(x)
        x2 = self.skip(x)
        return x1 + x2


class BayesianHintsUnet(nn.Module):
    def __init__(self, hints_len, sid_bins, in_height, in_width):
        super(BayesianHintsUnet, self).__init__()
        self.hints_len = hints_len
        self.sid_bins = sid_bins
        self.in_height = in_height
        self.in_width = in_width

        self.unet = Unet(in_channels=hints_len+3,
                         out_channels=2*sid_bins,
                         nf0 = 64,
                         num_down=4,
                         max_channels=512,
                         use_dropout=True,
                         upsampling_mode="nearest",
                         dropout_prob=0.1,
                         outermost_linear=False
                         )

    def forward(self, rgb, spad):
        in_height_orig, in_width_orig = rgb.size()[-2:]
        rgb = F.interpolate(rgb, (self.in_height, self.in_width), mode="bilinear", align_corners=True)
        x = torch.cat([spad.expand(-1, -1, rgb.size()[2], rgb.size()[3]), rgb], 1)
        x = self.unet(x)
        # print("unet output", (x < 0).any())
        x = F.interpolate(x, (in_height_orig, in_width_orig), mode="bilinear", align_corners=True)
        return x


class DORN_nyu_hints(Model):
    def __init__(self, hints_len=68, spad_weight=1.,
                 in_channels=3, in_height=257, in_width=353,
                 sid_bins=68, offset=0.,
                 min_depth=0., max_depth=10.,
                 alpha=0.6569154266167957, beta=9.972175646365525,
                 frozen=True, pretrained=True,
                 state_dict_file=os.path.join("models", "torch_params_nyuv2_BGR.pth.tar")):
        """
        :param hints_len: Uniformly spaced noisy depth hints (i.e. raw SPAD data)
        :param num_hints_layers: The number of layers for performing upsampling
        """
        super(DORN_nyu_hints, self).__init__()
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.spad_weight = spad_weight
        self.hints_len = hints_len
        self.sid_bins = sid_bins
        self.feature_extractor = \
            DORN_nyu_nohints(in_channels, in_height, in_width,
                             sid_bins, offset,
                             min_depth, max_depth,
                             alpha, beta,
                             frozen, pretrained,
                             state_dict_file)
        self.sid_obj = SIDTorch(sid_bins, alpha, beta, offset)
        self.hints_extractor = BayesianHints(hints_len, sid_bins)
        # self.hints_extractor = BayesianHintsUnet(hints_len, sid_bins, 256, 352)

    # Don't call .train() or eval on the feature extractor, since it should
    # always remain in eval mode.
    def train(self):
        self.hints_extractor.train()

    def eval(self):
        self.hints_extractor.eval()

    def forward(self, rgb, spad):
        img_features = self.feature_extractor(rgb)
        spad_features = self.hints_extractor(spad)
        spad_features = spad_features.expand(-1, -1, img_features.size(2), img_features.size(3))
        img_features.add_(self.spad_weight * torch.log_(spad_features + 1e-5))
        return img_features

    def get_loss(self, input_, device, resize_output=False):
        one = perf_counter()
        rgb = input_["rgb"].to(device)
        spad = input_["spad"].to(device)
        target = input_["rawdepth_sid"].to(device)
        mask = input_["mask"].to(device)
        two = perf_counter()
        # print("Move data to device: {}".format(two - one))
        depth_pred = self.forward(rgb, spad)
        # torch.cuda.synchronize()
        three = perf_counter()
        # print("Forward pass: {}".format(three - two))
        logprobs = self.to_logprobs(depth_pred)
        four = perf_counter()
        # print("To logprobs: {}".format(four - three))
        if resize_output:
            original_size = input_["rgb_orig"].size()[-2:]
            depth_pred_full = F.interpolate(depth_pred, size=original_size,
                                            mode="bilinear", align_corners=False)
            logprobs_full = self.to_logprobs(depth_pred_full)
            return self.ord_reg_loss(logprobs, target, mask), logprobs_full
        return self.ord_reg_loss(logprobs, target, mask), logprobs

    def write_updates(self, writer, input_, output, loss, it, tag):
        # Metrics
        # Ordinal regression loss
        writer.add_scalar(tag + "/ord_reg_loss", loss.item(), it)

        # Get predicted depth image
        depth_pred = self.ord_decode(output, self.sid_obj).cpu()
        depth_truth = input_["rawdepth"].cpu()
        mask = input_["mask"].cpu()
        metrics = self.get_metrics(depth_pred, depth_truth, mask)

        # write metrics
        for metric_name in metrics:
            writer.add_scalar(tag + "/{}".format(metric_name), metrics[metric_name], it)

        # min/max
        writer.add_scalar(tag + "/depth_min", torch.min(depth_pred).item(), it)
        writer.add_scalar(tag + "/depth_max", torch.max(depth_pred).item(), it)

        # CDF-ness of the stuff
        # X, Y = np.meshgrid(range(self.sid_bins), range(self.hints_len))
        # Z = self.hints_extractor.net.weight.cpu().numpy()[1::2, :, 0, 0]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X, Y, Z, cmap='Blues')
        # ax.view_init(40, 80)
        # writer.add_figure("model/hints_extractor_weights", fig, it)
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(self.hints_extractor.net.bias.cpu().numpy())
        # writer.add_figure("model/hints_extractor_bias", fig, it)

        # Images
        # RGB
        rgb_orig = vutils.make_grid(input_["rgb_orig"] / 255, nrow=4)
        writer.add_image(tag + "/rgb_orig", rgb_orig, it)
        # SID ground truth
        depth_sid_truth = self.sid_obj.get_value_from_sid_index(input_["rawdepth_sid_index"].cpu().long())
        depth_sid_truth = vutils.make_grid(depth_sid_truth, nrow=4,
                                               normalize=True, range=(self.min_depth, self.max_depth))
        writer.add_image(tag + "/depth_sid_truth", depth_sid_truth, it)
        # Original raw depth image
        depth_truth = vutils.make_grid(input_["rawdepth"], nrow=4,
                                       normalize=True, range=(self.min_depth, self.max_depth))
        writer.add_image(tag + "/depth_truth", depth_truth, it)
        # Output depth image
        depth_output = vutils.make_grid(depth_pred, nrow=4,
                                        normalize=True, range=(self.min_depth, self.max_depth))
        writer.add_image(tag + "/depth_output", depth_output, it)
        # Depth mask
        depth_mask = vutils.make_grid(input_["mask"], nrow=4, normalize=False)
        writer.add_image(tag + "/depth_mask", depth_mask, it)

# Steal methods from nohints
DORN_nyu_hints.get_metrics = staticmethod(DORN_nyu_nohints.get_metrics)
DORN_nyu_hints.ord_decode = staticmethod(DORN_nyu_nohints.ord_decode)
DORN_nyu_hints.ord_reg_loss = staticmethod(DORN_nyu_nohints.ord_reg_loss)
DORN_nyu_hints.to_logprobs = staticmethod(DORN_nyu_nohints.to_logprobs)
DORN_nyu_hints.evaluate = DORN_nyu_nohints.evaluate


class DORN_nyu_hints_Unet(DORN_nyu_hints):
    def __init__(self, hints_len=68, spad_weight=1.,
                 in_channels=3, in_height=257, in_width=353,
                 sid_bins=68, offset=0.,
                 min_depth=0., max_depth=10.,
                 alpha=0.6569154266167957, beta=9.972175646365525,
                 frozen=True, pretrained=True,
                 state_dict_file=os.path.join("models", "torch_params_nyuv2_BGR.pth.tar")):
        super(DORN_nyu_hints_Unet, self).__init__(hints_len=hints_len, spad_weight=spad_weight,
                 in_channels=in_channels, in_height=in_height, in_width=in_width,
                 sid_bins=sid_bins, offset=offset,
                 min_depth=min_depth, max_depth=max_depth,
                 alpha=alpha, beta=beta,
                 frozen=frozen, pretrained=pretrained,
                 state_dict_file=state_dict_file)
        self.hints_extractor = BayesianHintsUnet(hints_len, sid_bins, 256, 352)

    def forward(self, rgb, spad):
        img_features = self.feature_extractor(rgb)
        spad_features = self.hints_extractor(rgb, spad)
        spad_features = spad_features.expand(-1, -1, img_features.size(2), img_features.size(3))
        img_features.add_(self.spad_weight * torch.log_(spad_features + 1e-5))
        return img_features


class DORN_nyu_histogram_matching(DORN_nyu_nohints):
    def __init__(self, hints_len=68,
                 in_channels=3, in_height=257, in_width=353,
                 sid_bins=68, offset=0.,
                 min_depth=0., max_depth=10.,
                 alpha=0.6569154266167957, beta=9.972175646365525,
                 frozen=True, pretrained=True,
                 state_dict_file=os.path.join("models", "torch_params_nyuv2_BGR.pth.tar")):
        """
        :param hints_len: Uniformly spaced noisy depth hints (i.e. raw SPAD data)
        :param num_hints_layers: The number of layers for performing upsampling
        """
        super(DORN_nyu_histogram_matching, self).__init__(in_channels, in_height, in_width,
                                                          sid_bins, offset,
                                                          min_depth, max_depth,
                                                          alpha, beta,
                                                          frozen, pretrained,
                                                          state_dict_file)
        self.hints_len = hints_len

    def evaluate(self, data, device):
        # one = perf_counter()
        _, logprobs = self.get_loss(data, device, resize_output=True)

        # Ordinal decoding to index
        log_probs, _ = logprobs
        depth_index = torch.sum(log_probs >= np.log(0.5) ,dim=1, keepdim=True).long().cpu()

        # Histogram matching
        spad = data["spad"].numpy().squeeze()
        # Histogram denoising
        spad = self.denoise_hist(spad)

        depth_rescaled_index = torch.from_numpy(self.hist_match(depth_index.numpy(), spad)).long()
        # print(torch.min(depth_rescaled_index))
        # print(torch.max(depth_rescaled_index))

        # Finish ordinal decoding
        pred = self.sid_obj.get_value_from_sid_index(depth_rescaled_index)
        # depth_map_orig = self.sid_obj.get_value_from_sid_index(depth_index)

        gt = data["rawdepth_orig"].cpu()
        # rgb = data["rgb_orig"].cpu()
        # albedo = data["albedo_orig"].cpu()
        mask = data["mask_orig"].cpu()
        # spad = data["spad"].cpu()
        # out = {"depth_map": depth_map_rescaled,
        #        "depth_map_no_rescale": depth_map_orig,
        #        "gt": gt,
        #        "mask": mask,
        #        "rgb": rgb,
        #        "albedo": albedo,
        #        "spad": spad,
        #        "entry": data["entry"][0]
        #       }
        #        # "logprobs": logprobs}
        metrics = self.get_metrics(pred,
                                   gt,
                                   mask)
        return pred, metrics

    def denoise_hist(self, sid_hist, snr_threshold=0.2):
        """
        :param sid_hist: Numpy array
        :return:
        """
        # Normalize by bin width
        bin_widths = (self.sid_obj.sid_bin_edges[1:] - self.sid_obj.sid_bin_edges[:-1]).numpy()
        sid_hist = sid_hist / bin_widths
        # print(np.min(sid_hist))
        # print(np.max(sid_hist))

        # Threshold out all values less than threshold of total count
        # total_count = np.sum(sid_hist)
        # sid_hist[sid_hist < snr_threshold * total_count/len(sid_hist)] = 0.

        # Keep snr_threshold fraction of bins with the largest counts
        # i.e. zero out bottom (1 - snr_threshold) of bins
        # kth = int(np.ceil((1 - snr_threshold)*len(sid_hist)))
        # smallest_k = np.argpartition(sid_hist, kth)[:kth]
        # sid_hist[smallest_k] = 0.

        # Adaptively find snr_threshold by looking for largest decrease in the 1-cdf of the histogram
        # of the counts histogram (!)
        hist_of_hist, bin_edges = np.histogram(sid_hist, bins = 100)
        hist_of_hist = hist_of_hist / np.sum(hist_of_hist)
        one_m_cdf = 1. - np.cumsum(hist_of_hist)
        d_one_m_cdf = one_m_cdf[1:] - one_m_cdf[:-1]
        min_i = np.argmin(d_one_m_cdf)
        threshold = bin_edges[min_i]

        sid_hist[sid_hist < threshold] = 0.

        # Remultiply by bin width
        sid_hist = sid_hist * bin_widths
        return sid_hist

    @staticmethod
    def hist_match(source_depth_sid_index, histogram_target):
        """
        https://stackoverflow.com/questions/32655686/histogram-matching-of-two-images-in-python-2-x

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

        oldshape = source_depth_sid_index.shape
        source = source_depth_sid_index.ravel()
        # template = template.ravel()

        # get the set of unique pixel values and their corresponding indices and
        # counts
        s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                                return_counts=True)
        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        s_quantiles = np.cumsum(s_counts).astype(np.float32)
        s_quantiles /= s_quantiles[-1]

        # print(histogram_target)
        t_quantiles = np.cumsum(histogram_target).astype(np.float32)
        t_quantiles /= t_quantiles[-1]
        # print(np.min(t_quantiles))
        # print(np.max(t_quantiles))

        t_values = np.array(range(len(t_quantiles))).astype(np.float32)
        # print(len(t_quantiles))
        # print(len(t_values))
        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.round(np.interp(s_quantiles, t_quantiles, t_values))

        return interp_t_values[bin_idx].reshape(oldshape)



if __name__ == "__main__":
    import os
    import numpy as np
    from torch.utils.data import DataLoader
    from models.data.nyuv2_official_hints_sid_dataset import load_data, cfg
    from models.data.utils.spad_utils import cfg as spad_cfg
    data_config = cfg()
    spad_config = spad_cfg()
    # print(config)
    # print(spad_config)
    del data_config["data_name"]
    model = DORN_nyu_hints(
            in_channels=3,
            in_height=257,
            in_width=353,
            sid_bins=68,
            offset=data_config["offset"],
            min_depth=data_config["min_depth"],
            max_depth=data_config["max_depth"],
            alpha=data_config["alpha"],
            beta=data_config["beta"],
            frozen=True,
            pretrained=True,
            state_dict_file=os.path.join("models", "torch_params_nyuv2_BGR.pth.tar"),
            hints_len=68,
            spad_weight=1.
    )
    train, _, _ = load_data(**data_config, spad_config=spad_config)

    dataloader = DataLoader(train)
    start = perf_counter()
    input_ = next(iter(dataloader))
    data_load_time = perf_counter() - start
    print("dataloader: {}".format(data_load_time))
    # print(input_["entry"])
    # print(model.hints_extractor[0].weight)
    loss, output = model.get_loss(input_, "cpu")
    print(loss)