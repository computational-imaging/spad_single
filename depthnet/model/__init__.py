"""depthnet.model"""
import torch.nn as nn
from .depthmodel import DepthNet, DepthNetWithHints
from .unet_model import UNet, UNetWithHints, UNetMultiScaleHints
from .loss import (get_loss, berhu, delta, rmse, rel_abs_diff,
                   rel_sqr_diff)
from .utils import make_model, split_params_weight_bias, initialize
