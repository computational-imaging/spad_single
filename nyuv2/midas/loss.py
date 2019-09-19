import numpy as np

def get_depth_metrics(depth_pred, depth_truth, mask):
    """
    Takes torch tensors.
    :param depth_pred: Depth prediction
    :param depth_truth: Ground truth
    :param mask: Masks off invalid pixels
    :return: Dictionary of metrics
    """
    metrics = dict()
    # deltas
    metrics["delta1"] = delta(depth_pred, depth_truth, mask, 1.25)
    metrics["delta2"] = delta(depth_pred, depth_truth, mask, 1.25 ** 2)
    metrics["delta3"] = delta(depth_pred, depth_truth, mask, 1.25 ** 3)
    # rel_abs_diff
    metrics["rel_abs_diff"] = rel_abs_diff(depth_pred, depth_truth, mask)
    # rel_sqr_diff
    metrics["rel_sqr_diff"] = rel_sqr_diff(depth_pred, depth_truth, mask)
    # log10
    metrics["log10"] = log10(depth_pred, depth_truth, mask)
    # mse
    metrics["mse"] = mse(depth_pred, depth_truth, mask)
    # rmse
    metrics["rmse"] = rmse(depth_pred, depth_truth, mask)
    # rmse(log)
    metrics["log_rmse"] = rmse(np.log(depth_pred),
                               np.log(depth_truth),
                               mask)
    # print(metrics)
    return metrics

def delta(prediction, target, mask, threshold):
    """
    Given prediction and target, compute the fraction of indices i
    such that
    max(prediction[i]/target[i], target[i]/prediction[i]) < threshold
    """
    # print(prediction.dtype)
    # print(mask.dtype)
    # print(target.dtype)
    c = np.maximum(prediction[mask > 0]/target[mask > 0], target[mask > 0]/prediction[mask > 0])
    return np.sum((c < threshold))/(np.sum(mask))

def mse(prediction, target, mask):
    """
    Return the MSE of prediction and target
    """
    return np.mean((prediction[mask > 0] - target[mask > 0])**2)


def rmse(prediction, target, mask):
    """
    Return the RMSE of prediction and target
    """
    return np.sqrt(mse(prediction, target, mask))


def log10(prediction, target, mask):
    """
    Return the log10 loss metric:
    1/N * sum(| log_10(prediction) - log_10(target) |)
    :param prediction:
    :param target:
    :param mask:
    :return:
    """
    return np.mean(np.abs(np.log10(prediction[mask > 0]) - np.log10(target[mask > 0])))


def rel_abs_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative absolute difference:

    1/N*sum(|prediction - target|/target)
    """
    return np.mean(np.abs(prediction[mask > 0] - target[mask > 0])/(target[mask > 0] + eps))


def rel_sqr_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative squared difference:

    1/N*sum(||prediction - target||**2/target)
    """
    return np.mean((prediction[mask > 0] - target[mask > 0])**2/(target[mask > 0] + eps))
