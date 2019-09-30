import torch
import numpy as np
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.utils as utils


def get_midas(model_path, device="cpu"):
    model = MonoDepthNet(model_path)
    model.to(device)
    model.eval()
    return model


def midas_predict(model, img, depth_range, device):
    """
    Predict depth from RGB
    :param model: a midas model
    :param img: RGB image in 0-1
    :param depth_range: Range of output depths (e.g. (0., 10) for NYUv2)
    :param device: torch.device object to run on
    :return: Inverse Depth image, same size as RGB image, scaled to be in the output range.
    """
    assert depth_range[0] < depth_range[1]
    assert depth_range[0] >= 0.
    img_input = utils.resize_image(img)
    img_input = img_input.to(device)

    # compute
    with torch.no_grad():
        idepth = model.forward(img_input)

    idepth = utils.resize_depth(idepth, img.shape[1], img.shape[0])

    idepth_min = idepth.min()
    idepth_max = idepth.max()

    # Arbitrarily cap at 1./depth_range[1] + 10 for "infinite depth"
    idepth_range = (1. / depth_range[1],
                    1. / depth_range[0] if depth_range[0] > 0 else 1. / depth_range[1] + 10.)
    idepth_scaled = (idepth_range[1] - idepth_range[0]) * \
                    (idepth - idepth_min) / (idepth_max - idepth_min) + \
                    idepth_range[0]
    depth_scaled = 1. / idepth_scaled
    return depth_scaled