import torch
import numpy as np
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.utils as utils


def get_midas(model_path, device="cpu"):
    model = MonoDepthNet(model_path)
    model.to(device)
    model.eval()
    return model


def midas_idepth_predict(model, img, device):
    """
    Predict depth from RGB
    :param model: a midas model
    :param img: RGB image in 0-1 (numpy)
    :param device: torch.device object to run on
    :return: Inverse Depth image, same size as RGB image, scaled to be in the output range.
    """

    img_input = utils.resize_image(img)
    img_input = img_input.to(device)

    # compute
    with torch.no_grad():
        idepth = model.forward(img_input)

    idepth = utils.resize_depth(idepth, img.shape[1], img.shape[0])
    return idepth


def midas_predict(model, img, depth_range, device):
    assert depth_range[0] < depth_range[1]
    assert depth_range[0] >= 0.
    idepth = midas_idepth_predict(model, img, depth_range, device)

    idepth_min = idepth.min()
    idepth_max = idepth.max()

    # Arbitrarily cap at 1./depth_range[1] + 10 for "infinite depth"
    idepth_range = (1. / depth_range[1],
                    1. / depth_range[0] if depth_range[0] > 0 else 1. / depth_range[1] + 10.)
    idepth_scaled = (idepth_range[1] - idepth_range[0]) * \
                    (idepth - idepth_min) / (idepth_max - idepth_min) + \
                    idepth_range[0]
    return idepth_scaled


def midas_gt_predict(model, img, gt, crop, device):
    """
    Original MiDaS method for scaling the raw prediction using least squares.
    :param model:
    :param img:
    :param gt:
    :param crop:
    :param device:
    :return:
    """
    assert (gt > 0.).all()
    idepth = midas_idepth_predict(model, img, device)
    # print(idepth.shape)
    idepth = idepth[crop[0]:crop[1], crop[2]:crop[3]]
    # Use Least Squares to align the prediction to ground truth

    igt = 1./gt

    idepth_vec = np.hstack((idepth.reshape(-1, 1), np.ones((np.size(idepth), 1))))
    igt_vec = igt.reshape(-1, 1)
    scaleshift, _, _, _ = np.linalg.lstsq(idepth_vec.T.dot(idepth_vec), idepth_vec.T.dot(igt_vec), rcond=None)
    idepth_opt = scaleshift[0]*idepth + scaleshift[1]

    depth_opt = np.clip(1./idepth_opt, a_min=np.min(gt), a_max=10.)

    return depth_opt


def midas_gt_predict_masked(model, img, gt, mask, crop, device):
    """
    Original MiDaS method for scaling the raw prediction using least squares.
    :param model:
    :param img:
    :param gt:
    :param mask:
    :param crop:
    :param device:
    :return:
    """
    idepth = midas_idepth_predict(model, img, device)
    # print(idepth.shape)
    idepth = idepth[crop[0]:crop[1], crop[2]:crop[3]]
    # Use Least Squares to align the prediction to ground truth

    idepth_vec = np.hstack((idepth[mask > 0].reshape(-1, 1), np.ones((np.size(idepth[mask > 0]), 1))))
    igt_vec = (1./gt[mask > 0]).reshape(-1, 1)
    scaleshift, _, _, _ = np.linalg.lstsq(idepth_vec.T.dot(idepth_vec), idepth_vec.T.dot(igt_vec), rcond=None)
    idepth_opt = scaleshift[0]*idepth + scaleshift[1]

    depth_opt = np.clip(1./idepth_opt, a_min=np.min(gt), a_max=10.)

    return depth_opt

