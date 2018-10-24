import torch
import torch.cuda
from torch.nn import MSELoss, L1Loss

##################
# Loss functions #
##################

def berhu(prediction, target):
    """Function for calculating the reverse Huber Loss.
    Does not backpropagate through the threshold calculation"""
    diff = prediction - target
    threshold = 0.2*torch.max(torch.abs(prediction - target))
    c = threshold.detach()
    l2_part = torch.sum((diff**2 + c**2))/(2*c)
    l1_part = torch.sum(torch.abs(diff))
    return l1_part+l2_part

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
