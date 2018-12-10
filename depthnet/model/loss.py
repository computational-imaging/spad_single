import torch
import torch.cuda
from torch.nn import MSELoss, L1Loss

##################
# Loss functions #
##################

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
        return (1./torch.sum(mask))*out
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
def delta(prediction, target, mask, threshold):
    """
    Given prediction and target, compute the fraction of indices i
    such that
    max(prediction[i]/target[i], target[i]/prediction[i]) < threshold
    """
    c = torch.max(prediction[mask > 0]/target[mask > 0], target[mask > 0]/prediction[mask > 0])
    return torch.sum((c < threshold).float())/(torch.sum(mask))

def rmse(prediction, target, mask):
    """
    Return the RMSE of prediction and target
    """
    # print("prediction nans (rmse): {}".format(torch.isnan(prediction).any()))
    # print("target nans (rmse): {}".format(torch.isnan(target).any()))
    diff = prediction - target
    squares = (diff[mask > 0]).pow(2)
    # rawmaxidx = squares.view(-1).max(0)[1]
    # idx = []
    # for adim in list(squares.size())[::-1]:
    #     idx.append((rawmaxidx%adim).item())
    #     rawmaxidx = rawmaxidx / adim
    # # print(idx)
    # idx.reverse()
    # idx = tuple(idx)
    # print(idx)
    # print(torch.max(squares))
    # print(squares[idx])
    # print(prediction[idx])
    # print(target[idx])
    sum_squares = torch.sum(squares)
    return torch.sqrt((1./torch.sum(mask))*sum_squares)

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
    sum_abs_rel = torch.sum(torch.abs(diff[mask > 0])/(target[mask > 0] + eps))
    return (1./torch.sum(mask))*sum_abs_rel

def rel_sqr_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative squared difference:

    1/N*sum(||prediction - target||**2/target)
    """
    diff = prediction - target
    sum_sqr_rel = torch.sum((diff[mask > 0]).pow(2)/(target[mask > 0] + eps))
    return (1./torch.sum(mask))*sum_sqr_rel

if __name__ == '__main__':
    print(test_rmse())
