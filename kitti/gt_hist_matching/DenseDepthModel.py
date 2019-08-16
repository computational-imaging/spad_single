import torch
import numpy as np

import os

from models.loss import get_depth_metrics

from DenseDepth.model import create_model
from DenseDepth.utils import scale_up, predict


class DenseDepth:
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
        self.model = create_model(existing)
        self.crop = np.array(crop)

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

    def evaluate(self, rgb, gt, mask):
        """
        Takes torch tensors, but returns a torch tensor prediction.
        :param rgb: N x H x W x C in RGB order (not BGR)
        :param crop: length-4 array with crop pixel coordinates
        :param gt: N x H x W x C
        :return: torch tensor prediction, metrics dict, and number of valid pixels
        """
        pred = self.predict(rgb.cpu().numpy())
        pred = pred[:, self.crop[0]:self.crop[1], self.crop[2]:self.crop[3]]
        pred = torch.from_numpy(pred).cpu().unsqueeze(0).float()
        metrics = self.get_metrics(pred, gt.cpu(), mask.cpu())
        return pred, metrics, torch.sum(mask).item()

    @staticmethod
    def get_metrics(pred, gt, mask):
        return get_depth_metrics(pred, gt, mask)
