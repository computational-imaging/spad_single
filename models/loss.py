import torch
from torch.nn import MSELoss, L1Loss


##################
# Loss functions #
##################
# Run on import

def ord_reg_loss(prediction, target, mask, size_average=True, eps=1e-6):
    """Calculates the Ordinal Regression loss
    :param prediction: a tuple (log_ord_c0, log_ord_c1).
        log_ord_c0 is is an N x K x H x W tensor
        where each pixel location is a length K vector containing log-probabilities log P(l > 0),..., log P(l > K-1).

        The log_ord_c1 is the same, but contains the log-probabilities log (1 - P(l > 0)),..., log (1 - P(l > K-1))
        instead.
    :param target - per-pixel vector of 0's and 1's such that if the true depth
    bin is k then the vector contains 1's up to entry k-1 and 0's for the remaining entries.
    e.g. if k = 3 and the total number of bins is 7 then

    target[:, i, j] = [1, 1, 1, 0, 0, 0, 0]

    :param mask - same size as prediction and target, 1.0 if that position is
    to be used in the loss calculation, 0 otherwise.
    :param size_average - whether or not to take the average over all the mask pixels.
    """
    log_ord_c0, log_ord_c1 = prediction
    nbins = log_ord_c0.size(1)
    mask_L = ((target > 0) & (mask > 0))
    mask_U = (((1. - target) > 0) & (mask > 0))

    out = -(torch.sum(log_ord_c0[mask_L]) + torch.sum(log_ord_c1[mask_U]))
    if size_average:
        total = torch.sum(mask).item()
        if total > 0:
            return (1./torch.sum(mask))*out
        else:
            return torch.zeros(1)
    return out

def berhu(prediction, target, mask, size_average=True):
    """Function for calculating the reverse Huber Loss.
    Does not backpropagate through the threshold calculation.

    Returns a single tensor

    mask - should be same size as prediction and target, and have a 1 if that position
    is to be used in the loss calculation, 0 otherwise.

    """
    # print("prediction nans: {}".format(torch.isnan(prediction).any()))
    # print("target nans: {}".format(torch.isnan(target).any()))
    diff = torch.abs(prediction[mask > 0] - target[mask > 0])
    threshold = 0.2*torch.max(diff)
    c = threshold.detach()
    l2_part = (diff**2 + c**2)/(2*c)
    l1_part = diff
    out = torch.sum(l1_part[diff <= c])+torch.sum(l2_part[diff > c])
    if size_average:
        total = torch.sum(mask).item()
        if total > 0:
            return (1./torch.sum(mask))*out
        else:
            return torch.zeros(1)
    return out

#################
# Other Metrics #
#################
def delta(prediction, target, mask, threshold):
    """
    Given prediction and target, compute the fraction of indices i
    such that
    max(prediction[i]/target[i], target[i]/prediction[i]) < threshold
    """
    # print(prediction.dtype)
    # print(mask.dtype)
    # print(target.dtype)
    c = torch.max(prediction[mask > 0]/target[mask > 0], target[mask > 0]/prediction[mask > 0])
    return torch.sum((c < threshold).float())/(torch.sum(mask))

def mse(prediction, target, mask):
    """
    Return the RMSE of prediction and target
    """
    diff = prediction - target
    squares = (diff[mask > 0]).pow(2)
    out = torch.sum(squares)
    total = torch.sum(mask).item()
    if total > 0:
        return (1. / torch.sum(mask)) * out
    else:
        return torch.zeros(1)

def rmse(prediction, target, mask):
    """
    Return the RMSE of prediction and target
    """
    return torch.sqrt(mse(prediction, target, mask))


def log10(prediction, target, mask, size_average=True):
    """
    Return the log10 loss metric:
    1/N * sum(| log_10(prediction) - log_10(target) |)
    :param prediction:
    :param target:
    :param mask:
    :return:
    """
    out = torch.sum(torch.abs(torch.log10(prediction[mask > 0]) - torch.log10(target[mask > 0])))
    if size_average:
        total = torch.sum(mask).item()
        if total > 0:
            return (1./torch.sum(mask))*out
        else:
            return torch.zeros(1)
    return out


def test_rmse():
    prediction = 2*torch.ones(3, 3, 3)
    target = torch.zeros(3, 3, 3)
    err = rmse(prediction, target, torch.ones(3, 3, 3))
    return err


def rel_abs_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative absolute difference:

    1/N*sum(|prediction - target|/target)
    """
    diff = prediction - target
    out = torch.sum(torch.abs(diff[mask > 0])/(target[mask > 0] + eps))
    total = torch.sum(mask).item()
    if total > 0:
        return (1. / torch.sum(mask)) * out
    else:
        return torch.zeros(1)


def rel_sqr_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative squared difference:

    1/N*sum(||prediction - target||**2/target)
    """
    diff = prediction - target
    out = torch.sum((diff[mask > 0]).pow(2)/(target[mask > 0] + eps))
    total = torch.sum(mask).item()
    if total > 0:
        return (1. / torch.sum(mask)) * out
    else:
        return torch.zeros(1)


if __name__ == '__main__':
    print(test_rmse())
