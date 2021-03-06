import torch
import numpy as np
from MiDaS.monodepth_net import MonoDepthNet
import MiDaS.utils as utils

from midas_loss import compute_scale_and_shift


def get_midas(model_path, device="cpu"):
    model = MonoDepthNet(model_path)
    model.to(device)
    model.eval()
    return model


def midas_idepth_predict(model, img, device):
    """
    Predict inverse depth from RGB
    :param model: a midas model
    :param img: RGB image in 0-1 (numpy)
    :param device: torch.device object to run on
    :return: Inverse Depth image, same size as RGB image, scaled to be in the output range. (numpy)
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
    :param model: a MiDaS model
    :param img: numpy RGB image in [0, 1]
    :param gt: Ground truth depth map
    :param mask: {0, 1} mask for valid depth pixels.
    :param crop: Center crop to evaluate on
    :param device: torch.device
    :return: depth estimate, scaled and shifted
    """
    idepth = midas_idepth_predict(model, img, device)
    # print(idepth.shape)
    idepth = idepth[crop[0]:crop[1], crop[2]:crop[3]]
    # Use Least Squares to align the prediction to ground truth in inverse depth space
    # Only use the masked pixels to do the least squares fit.
    igt = 1. / (gt + 1e-4)
    # scaleshift = lsq_scale_shift(idepth, igt, mask)
    # idepth_opt = scaleshift[0]*idepth + scaleshift[1]

    scale, shift = compute_scale_and_shift(torch.from_numpy(idepth).unsqueeze(0).float(),
                                           torch.from_numpy(igt).unsqueeze(0).float(),
                                           torch.from_numpy(mask).unsqueeze(0).float())
    # print(scale.item())
    # print(shift.item())
    idepth_opt = scale.item() * idepth + shift.item()

    depth_opt = np.clip(1./idepth_opt, a_min=np.min(gt[mask > 0]), a_max=10.)

    return depth_opt

def lsq_scale_shift(pred, target, mask):
    pred_vec= np.hstack((pred[mask > 0].reshape(-1, 1), np.ones((np.size(pred[mask > 0]), 1))))
    target_vec = target[mask > 0].reshape(-1, 1)
    scaleshift, _, _, _ = np.linalg.lstsq(pred_vec.T.dot(pred_vec), pred_vec.T.dot(target_vec), rcond=None)
    return scaleshift

