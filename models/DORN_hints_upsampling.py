import torch
import torch.nn as nn

from .DORN_nohints import DORN_nyu_nohints

class DORN_nyu_hints(nn.Module):
    def __init__(self, hints_len, num_hints_layers):
        """

        :param hints_len: Uniformly spaced noisy depth hints (i.e. raw SPAD data)
        :param num_hints_layers: The number of layers for performing upsampling
        """
        self.feature_extractor = DORN_nyu_nohints()
        self.hints_net = nn.Sequential([

        ]
        )

