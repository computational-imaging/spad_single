import torch
import numpy as np
from pdb import set_trace

def get_depth_index(model, input_, device):
    rgb = input_["rgb"].to(device)
    with torch.no_grad():
        x = model(rgb)
        log_probs, _ = model.to_logprobs(x)
        depth_index = torch.sum((log_probs >= np.log(0.5)), dim=1, keepdim=True).long()
    return depth_index


def img_to_hist(x, inv_squared_depths=None, albedo=None):
    """
    Assumes each pixel of x is normalized to sum to 1.
    x has shape N x C x H x W.
    """
    weights = torch.ones_like(x)
    if albedo is not None:
        assert albedo.shape[0] == x.shape[0]
        assert albedo.shape[1] == 1
        assert albedo.shape[-2:] == x.shape[-2:]
        weights = weights * albedo
    if inv_squared_depths is not None:
        assert inv_squared_depths.shape[:2] == x.shape[:2] and inv_squared_depths.shape[-2:] == (1,1)
        weights = weights * inv_squared_depths
#     x_hist = torch.sum(x, dim=(2, 3), keepdim=True) / (x.shape[2] * x.shape[3])
    x_hist = torch.sum(x*weights, dim=(2,3), keepdim=True)
    x_hist = x_hist / torch.sum(x_hist)

    return x_hist


def threshold_and_normalize_pixels(x, eps=1e-2):
    """
    Given an NxCxWxH tensor, first threshold off values smaller than eps, then normalizes along the C dimension so each pixel's histogram
    sums to 1.
    """
    x = torch.clamp(x, min=eps)
    x = x / torch.sum(x, dim=1, keepdim=True)
    return x


def kernel_density_estimation(x, sigma, n_bins, eps=1e-2):
    """
    Given x, a batch of 2D depth maps, (N x 1 x W x H),
    return a tensor of size N x {n_bins} x W x H where each pixel has been converted into
    a histogram using a gaussian kernel with standard deviation sigma.
    """
    N, _, W, H = x.shape
    device = x.device
    ind = torch.linspace(0, n_bins-1, n_bins).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, -1, W, H).to(device)
    y = torch.exp((-1./sigma**2)*(x - ind)**2)
    y = threshold_and_normalize_pixels(y, eps=eps)
    return y


def sinkhorn_dist(cost_mat, lam, hist_pred, gt_hist, num_iters=100, eps=1e-1):
    """
    cost_mat: C x C
    hist_pred: N x C x H x W Should all be nonzero!
    gt_hist: N x C x H x W
    """

    r = hist_pred.permute([0,2,3,1]).unsqueeze(-1) # N x W x H x C x 1
#     print("r min", torch.min(r))
    c = gt_hist.permute([0,2,3,1]).unsqueeze(-1)     # N x W x H x C x 1
    M = cost_mat.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1 x 1 x 1 x C x C
    K = torch.exp(-lam*M)
    x = torch.ones_like(r)/r.shape[-2] # Initialize histogram to be uniform
    K_T = K.transpose(3, 4)
    # set_trace()
    # print(r)
    # print(c)
    # print(lam)
    # print(eps)
    # set_trace()
    for i in range(num_iters):
        temp1 = K_T.matmul(1./x) # N x W x H x C x 1
        temp2 = c*(1./temp1)  # N x W x H x C x 1
        temp3 = K.matmul(temp2)
        # r_diag = torch.diag_embed(torch.transpose(1./r, -1, -2)).squeeze(-3)
        # x_temp = r_diag.matmul(temp3)
        x_temp = (1./r)*temp3
        if torch.sum(torch.abs(x - x_temp)) < eps:
            break
        # print("diff", torch.sum(torch.abs(x - x_temp))) # Inspect for convergence
        if torch.isnan(x_temp).any().item():
            print("iteration {}".format(i))
            print(x)
            break

        # set_trace()
        x = x_temp
    u = 1./x
    v = c*(1./K_T.matmul(u))
    # Extract the transport matrix as well
    u_diag = torch.diag_embed(torch.transpose(u, -1, -2)).squeeze(-3)
    v_diag = torch.diag_embed(torch.transpose(v, -1, -2)).squeeze(-3)
    P = u_diag.matmul(K).matmul(v_diag)
    return torch.sum(u*(K*M).matmul(v)), P


def entropy(p, dim = -1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim = dim, keepdim = keepdim) # can be a scalar, when PyTorch.supports it


def optimize_depth_map(x_index_init, sigma, n_bins,
                       cost_mat, lam, spad_hist,
                       lr, num_sgd_iters, num_sinkhorn_iters,
                       kde_eps=1e-5,
                       sinkhorn_eps=1e-2,
                       inv_squared_depths=None,
                       albedo=None):
    x = x_index_init.clone().detach().float().requires_grad_(True)
    # print(x)
    for i in range(num_sgd_iters):
        # with torch.autograd.detect_anomaly():

        # Per-pixel depth index to per-pixel histogram
        x_img = kernel_density_estimation(x, sigma, n_bins, eps=kde_eps)

        # per-pixel histogram to full-image histogram
        x_hist = img_to_hist(x_img, inv_squared_depths=inv_squared_depths, albedo=albedo)
        # set_trace()
        hist_loss, P = sinkhorn_dist(cost_mat, lam,
                                     x_hist, spad_hist,
                                     num_iters=num_sinkhorn_iters,
                                     eps=sinkhorn_eps)

        # print(hist_loss)
        # entropy_loss = torch.sum(entropy(x_img, dim=1))
        loss = hist_loss
        loss.backward()
        with torch.no_grad():
            if torch.isnan(x.grad).any().item():
                            # set_trace()
                print("nans detected in x.grad")
                break
            x -= lr*x.grad
            if torch.sum(torch.abs(lr*x.grad)) < sinkhorn_eps: # Reuse sinkhorn eps for sgd convergence.
                x = torch.clamp(x, min=0., max=n_bins).requires_grad_(True)
                return x, x_img, x_hist
            x.grad.zero_()
            x = torch.clamp(x, min=0., max=n_bins).requires_grad_(True)
        # print("warning: sgd exited before convergence.")
    return x, x_img, x_hist