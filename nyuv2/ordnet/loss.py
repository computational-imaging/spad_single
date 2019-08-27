import torch

def scale_invariant_error(pred, gt):
    """
    Taken from Eigen et. al. Multi Scale Network paper
    :param pred: Predicted Depth, N x C x H x W
    :param gt: Ground Truth Depth, N x C x H x W
    :return: The scale invariant error
    """
