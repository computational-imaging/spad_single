"""depthnet.model"""
import torch.nn as nn
from .depthmodel import DepthNet, DepthNetWithHints
from .unet_model import UNet, UNetWithHints, UNetMultiScaleHints, UNetDORN, UNetDORNWithHints
from .loss import (berhu, delta, rmse, rel_abs_diff,
                   rel_sqr_diff)
from .utils import make_network, split_params_weight_bias, initialize
