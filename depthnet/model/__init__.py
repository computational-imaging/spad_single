"""depthnet.model"""
import torch.nn as nn
from .depthmodel import DepthNet, DepthNetWithHints
from .unet_model import UNet, UNetWithHints, UNetMultiScaleHints
from .loss import (get_loss, berhu, delta, rmse, rel_abs_diff,
                   rel_sqr_diff)

def make_model(model_name, model_params, model_state_dict_fn):
    # model
    model_class = globals()[model_name]
    model = model_class(**model_params)
    if model_state_dict_fn is not None:
        model.load_state_dict(model_state_dict_fn())
    else: # New model - apply initialization
        # m.initialize(model)
        pass # Use default pytorch initialization (He initialization)
    return model


def split_params_weight_bias(model):
    """Split parameters into weight and bias terms,
    in order to apply different regularization."""
    split_params = [{"params": []}, {"params": [], "weight_decay": 0.0}]
    for name, param in model.named_parameters():
        # print(name)
        # print(param)
        if "weight" in name:
            split_params[0]["params"].append(param)
        elif "bias" in name:
            split_params[1]["params"].append(param)
        else:
            raise ValueError("Unknown param type: {}".format(name))
    return split_params


def initialize(model):
    """Initialize a model.
    Conv weights: Xavier initialization
    Batchnorm weights: Constant 1
    All biases: 0
    """
    for name, param in model.named_parameters():
        #            print(param.shape)
        #            print(len(param.shape))
        #            print(name)
        if "conv" in name and "weight" in name and len(param.shape) == 1:
            nn.init.normal_(param) # 1x1 conv
        elif "conv" in name and "weight" in name:
            #             print(name)
            nn.init.xavier_normal_(param)
            #nn.init.constant_(param, 1)
        elif "norm" in name and "weight" in name:
            #             print(name)
            nn.init.constant_(param, 1)
        elif "bias" in name:
            nn.init.constant_(param, 0)
