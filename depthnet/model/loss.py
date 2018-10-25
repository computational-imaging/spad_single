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
