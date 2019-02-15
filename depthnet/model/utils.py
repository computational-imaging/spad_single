import abc
import torch.nn as nn
from copy import deepcopy

from .unet_model import UNet, UNetWithHints, UNetDORN, UNetDORNWithHints

class ModelWrapper(abc.ABC):
    def __init__(self, network, pre_active=True, post_active=False):
        """
        model - nn.Module - the neural net we're wrapping
        pre_active - bool - whether or not to perform the preprocessing in pre()
            - Usually active for both test and train time.
        post_active - bool - whether or not to perform the postprocessing in post()
            - Usually only active at test time, not at train time.
        """
        self.network = network
        self.pre_active = pre_active
        self.post_active = post_active

    @abc.abstractmethod
    def pre(self, input_):
        """Data preprocessing"""
        return NotImplemented

    @abc.abstractmethod
    def post(self, output):
        """Data postprocessing"""
        return NotImplemented

    def __call__(self, input_):
        if self.pre_active:
            input_ = self.pre(input_)
        output = self.network(input_)
        if self.post_active:
            output = self.post(output)
        return output


def make_network(network_name, network_params, network_state_dict_fn):
    """
    Make a network from the name, params, and function for getting the state dict (if not None).
    :param network_name: The name of the network's class in the global namespace.
    :param network_params: A dictionary for initializing the network with the appropriate parameters
    :param network_state_dict_fn: If not None, a function that, when called, returns a
                                  state dict for initializing the network.
    :return: An initialized network.
    """
    # network
    network_class = globals()[network_name]
    network = network_class(**network_params)
    if network_state_dict_fn is not None:
        network.load_state_dict(network_state_dict_fn())
    else: # New network - apply initialization
        # m.initialize(network)
        pass # Use default pytorch initialization (He initialization)
    return network

def split_params_weight_bias(network):
    """Split parameters into weight and bias terms,
    in order to apply different regularization."""
    split_params = [{"params": []}, {"params": [], "weight_decay": 0.0}]
    for name, param in network.named_parameters():
        # print(name)
        # print(param)
        if "weight" in name:
            split_params[0]["params"].append(param)
        elif "bias" in name:
            split_params[1]["params"].append(param)
        else:
            raise ValueError("Unknown param type: {}".format(name))
    return split_params


def initialize(network):
    """Initialize a network.
    Conv weights: Xavier initialization
    Batchnorm weights: Constant 1
    All biases: 0
    """
    for name, param in network.named_parameters():
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
