import torch
import torch.cuda
from torch.nn import MSELoss, L1Loss

##################
# Loss functions #
##################

def berhu(prediction, target, size_average=True):
    """Function for calculating the reverse Huber Loss.
    Does not backpropagate through the threshold calculation.

    Returns a single tensor

    """
    diff = torch.abs(prediction - target)
    threshold = 0.2*torch.max(diff)
    c = threshold.detach()
    l2_part = (diff**2 + c**2)/(2*c)
    l1_part = diff
    out = torch.sum(l1_part[diff <= c])+torch.sum(l2_part[diff > c])
    if size_average:
        return (1./diff.numel())*out
    return out

def get_loss(loss_fn):
    """Get a function for evaluating the loss from a string."""
    loss = None
    if loss_fn == "berhu":
        loss = berhu
    elif loss_fn == "l2":
        loss = MSELoss()
        if torch.cuda.is_available():
            loss.cuda()
    elif loss_fn == "l1":
        loss = L1Loss()
        if torch.cuda.is_available():
            loss.cuda()
    return loss

#################
# Other Metrics #
#################
def delta(prediction, target, threshold):
    """
    Given prediction and target, compute the fraction of indices i
    such that
    max(prediction[i]/target[i], target[i]/prediction[i]) < threshold
    """
    c = torch.max(prediction/target, target/prediction)
    return torch.sum((c > threshold).float())/c.numel()

def rmse(prediction, target):
    """
    Return the RMSE of prediction and target
    """
    sum_squares = torch.sum((prediction - target).pow(2))
    return torch.sqrt((1./sum_squares.numel())*sum_squares)

def rel_abs_diff(prediction, target):
    """
    The average relative absolute difference:

    1/N*sum(|prediction - target|/target)
    """
    sum_abs_rel = torch.sum(torch.abs(prediction - target)/target)
    return (1./sum_abs_rel.numel())*sum_abs_rel

def rel_sqr_diff(prediction, target):
    """
    The average relative squared difference:

    1/N*sum(||prediction - target||**2/target)
    """
    sum_sqr_rel = torch.sum((prediction - target).pow(2)/target)
    return (1./sum_sqr_rel.numel())*sum_sqr_rel
