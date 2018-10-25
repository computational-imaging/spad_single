# depthnet.model init file
import torch.nn as nn
from .depthmodel import DepthNet, DepthNetWithHints
from .unet_model import UNet, UNetWithHints
from .loss import berhu, get_loss

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
