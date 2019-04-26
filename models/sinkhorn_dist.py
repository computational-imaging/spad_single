import torch
import numpy as np

def get_depth_index(model, input_, device):
    rgb = input_["rgb"].to(device)
    with torch.no_grad():
        x = model(rgb)
        log_probs, _ = model.to_logprobs(x)
        depth_index = torch.sum((log_probs >= np.log(0.5)), dim=1, keepdim=True).long()
    return depth_index

def img_to_hist(x):
    """
    Assumes each pixel of x is normalized to sum to 1.
    x has shape N x C x W x H.
    """
    x_hist = torch.sum(x, dim=(0, 2, 3), keepdim=True) / (x.shape[2] * x.shape[3])
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

def sinkhorn_dist(cost_mat, lam, hist_pred, gt_hist, num_iters=100, eps=1e-3):
    """
    cost_mat: C x C
    hist_pred: N x C x W x H Should all be nonzero!
    gt_hist: N x C x W x H
    """
    r = hist_pred.permute([0,2,3,1]).unsqueeze(-1) # N x W x H x C x 1
#     print("r min", torch.min(r))
    c = gt_hist.permute([0,2,3,1]).unsqueeze(-1)     # N x W x H x C x 1
    M = cost_mat.unsqueeze(0).unsqueeze(0).unsqueeze(0) # 1 x 1 x 1 x C x C
    K = torch.exp(-lam*M)
    x = torch.ones_like(r)/r.shape[-2] # Initialize histogram to be uniform
    K_T = K.transpose(3, 4)
    for i in range(num_iters):
#         print("x nans", torch.isnan(x).any().item())
#         print("x min", torch.min(x))
        temp1 = K_T.matmul(1./x) # N x W x H x C x 1
#         print("temp1 nans", torch.isnan(temp1).any().item())
#         print(temp1)
        temp2 = c*(1./temp1)  # N x W x H x C x 1
#         print("temp2 nans", torch.isnan(temp2).any().item())
        temp3 = K.matmul(temp2)
#         print("temp3 nans", torch.isnan(temp3).any().item())
        if torch.isnan(temp3).any().item():
            print(temp3)
            print(temp2)
        r_diag = torch.diag_embed(torch.transpose(1./r, -1, -2)).squeeze(-3)
#         print("rdiag nans", torch.isnan(r_diag).any().item())
        x_temp = r_diag.matmul(temp3)
        # print(x_temp)
#         print("diff", torch.sum(torch.abs(x - x_temp)))
        x = x_temp
    u = 1./x
    v = c*(1./K_T.matmul(u))
    # Extract the transport matrix as well
#     print("u", u.shape)
    u_diag = torch.diag_embed(torch.transpose(u, -1, -2)).squeeze(-3)
#     print("u_diag", u_diag.shape)
#     print("v", v.shape)
    v_diag = torch.diag_embed(torch.transpose(v, -1, -2)).squeeze(-3)
#     print("v_diag", v_diag.shape)
    P = u_diag.matmul(K).matmul(v_diag)
    return torch.sum(u*(K*M).matmul(v)), P

def entropy(p, dim = -1, keepdim=False):
    return -torch.where(p > 0, p * p.log(), p.new([0.0])).sum(dim = dim, keepdim = keepdim) # can be a scalar, when PyTorch.supports it


def optimize_depth_map(x_index_init, sigma, n_bins,
                       cost_mat, lam, spad_hist,
                       lr, num_sgd_iters, num_sinkhorn_iters):
    x = x_index_init.clone().detach().float().requires_grad_(True)
    for i in range(num_sgd_iters):
        # Per-pixel depth index to per-pixel histogram
        x_img = kernel_density_estimation(x, sigma, n_bins, eps=1e-1)
        x_hist = img_to_hist(x_img)
        # print(x_hist)
        # break
        hist_loss, P = sinkhorn_dist(cost_mat, lam,
                                     x_hist, spad_hist,
                                     num_iters=num_sinkhorn_iters)
        loss = hist_loss
        loss.backward(retain_graph=True)
        # print(x.grad)
        with torch.no_grad():
            x -= lr*x.grad
            x.grad.zero_()
            if torch.isnan(x.grad).any().item():
                print("nans detected in x.grad")
                break
    return x, x_img, x_hist