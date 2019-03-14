import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.core.model_core import Model
from models.DORN_nohints import DORN_nyu_nohints
from models.data.utils.sid_utils import SIDTorch

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
        self.feature_extractor = \
            DORN_nyu_nohints(in_channels, in_height, in_width,
                             sid_bins, offset,
                             min_depth, max_depth,
                             alpha, beta,
                             frozen, pretrained,
                             state_dict_file)
        self.sid_obj = SIDTorch(sid_bins, alpha, beta, offset)
        self.hints_extractor = nn.Sequential(
            nn.Conv2d(hints_len, 512, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(512, 512, kernel_size=1),
            nn.ReLU(True),
            nn.Conv2d(512, 2*sid_bins, kernel_size=1)
            # nn.ReLU(True)
        )

    def forward(self, rgb, spad):
        img_features = self.feature_extractor(rgb)
        spad_features = self.hints_extractor(spad)

        spad_features = spad_features.expand(-1, -1, img_features.size(2), img_features.size(3))
        # cdf complement: 1 - P(l <= k) = P(l > k)
        # spad_features_comp = 1. - spad_features
        # img_features[:, 1::2, :, :].add_(self.alpha * spad_features_comp)
        img_features.add_(self.spad_weight * spad_features)
        return img_features

    def get_loss(self, input_, device, resize_output=False):
        rgb = input_["rgb"].to(device)
        spad = input_["spad"].to(device)
        target = input_["rawdepth_sid"].to(device)
        mask = input_["mask"].to(device)
        depth_pred = self.forward(rgb, spad)
        logprobs = self.to_logprobs(depth_pred)
        if resize_output:
            original_size = input_["rgb_orig"].size()[-2:]
            depth_pred_full = F.interpolate(depth_pred, size=original_size,
                                            mode="bilinear", align_corners=False)
            logprobs_full = self.to_logprobs(depth_pred_full)
            return self.ord_reg_loss(logprobs, target, mask), logprobs_full
        return self.ord_reg_loss(logprobs, target, mask), logprobs

# Steal methods from nohints
DORN_nyu_hints.get_metrics = staticmethod(DORN_nyu_nohints.get_metrics)
DORN_nyu_hints.ord_decode = staticmethod(DORN_nyu_nohints.ord_decode)
DORN_nyu_hints.ord_reg_loss = staticmethod(DORN_nyu_nohints.ord_reg_loss)
DORN_nyu_hints.to_logprobs = staticmethod(DORN_nyu_nohints.to_logprobs)
DORN_nyu_hints.write_eval = DORN_nyu_nohints.write_eval
DORN_nyu_hints.evaluate_dir = DORN_nyu_nohints.evaluate_dir
DORN_nyu_hints.evaluate_file = DORN_nyu_nohints.evaluate_file
DORN_nyu_hints.write_updates = DORN_nyu_nohints.write_updates


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
    train, _, _ = load_data(**data_config, spad_config=spad_config)

    dataloader = DataLoader(train)
    input_ = next(iter(dataloader))
    print(input_["entry"])

    model = DORN_nyu_hints(68, 136, 1.)

    loss, output = model.get_loss(input_, "cpu")
    print(loss)