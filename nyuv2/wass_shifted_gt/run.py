#! /usr/bin/env python3

import os
import torch
import torch.optim as optim
from models.data.data_utils.sid_utils import SIDTorch
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp


def wasserstein_loss(x, y, C, eps=1e-6):
    """
    Calculate the wasserstein loss between two histograms given the cost matrix.
    :param h1, h2: Shape (N,) histograms. Should be nonnegative and sum to 1.
    :param C: Cost matrix of shape (N,N).
    """
    assert np.abs(np.sum(x) - 1.) < eps and (x >= -eps).all()
    assert np.abs(np.sum(y) - 1.) < eps and (y >= -eps).all()
    assert x.shape == y.shape and len(x.shape) == 1 and len(y.shape) == 1
    assert len(C.shape) == 2 and C.shape[0] == x.shape[0] and C.shape[1] == y.shape[0]

    n = x.shape[0]
    T = cp.Variable((n, n))
    obj = cp.Minimize(cp.trace(T * C))
    constr = [
        T * np.ones(n) == x,
        T.T * np.ones(n) == y,
        T >= 0
    ]
    prob = cp.Problem(obj, constr)
    prob.solve(solver="OSQP", eps_abs=eps)
    return prob.value, T.value

def spad_forward(x_index, mask, sigma, n_bins, kde_eps=1e-2,
                 inv_squared_depths=None, scaling=None):
    """
    Converts image of per-pixel depth indices to a single histogram for the whole image.
    :param x_index: N x 1 x H x W tensor of per-pixel depth indices in [0, n_bins]
    :param mask: N x 1 x H x W tensor of valid depth pixels
    :param sigma: width parameter for kernel density estimation
    :param n_bins: Number of bin indices
    :param kde_eps: Epsilon used for KDE to prevent 0 values in histogram.
    :param inv_squared_depths: N x C x 1 x 1
    :param scaling: N x 1 x H x W
    :returns x_hist: N x n_bins histograms
    """
    per_pixel_hists = kernel_density_estimation(x_index, sigma, n_bins, eps=kde_eps)
    x = per_pixel_hists * mask
    weights = torch.ones_like(x)
    if scaling is not None:
        assert scaling.shape[0] == x.shape[0]
        assert scaling.shape[1] == 1
        assert scaling.shape[-2:] == x.shape[-2:]
        weights = weights * scaling
    if inv_squared_depths is not None:
        assert inv_squared_depths.shape[:2] == x.shape[:2] and inv_squared_depths.shape[-2:] == (1,1)
        weights = weights * inv_squared_depths
#     x_hist = torch.sum(x, dim=(2, 3), keepdim=True) / (x.shape[2] * x.shape[3])
    x_hist = torch.sum(x*weights, dim=(2,3), keepdim=True)
    x_hist = x_hist / torch.sum(x_hist, dim=1, keepdim=True)
    return x_hist.squeeze(-1).squeeze(-1)


def kernel_density_estimation(x, sigma, n_bins, eps=1e-2):
    """
    Given x, a batch of 2D depth maps, (N x 1 x H x W),
    return a tensor of size N x {n_bins} x H x W where each pixel has been converted into
    a histogram using a gaussian kernel with standard deviation sigma.
    """
    N, _, W, H = x.shape
    device = x.device
    ind = torch.linspace(0, n_bins, n_bins+1).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand(N, -1, W, H).to(device)
    y = torch.exp((-1./sigma**2)*(x - ind)**2)
    y = threshold_and_normalize_pixels(y, eps=eps)
    return y


def threshold_and_normalize_pixels(x, eps=1e-2):
    """
    Given an NxCxWxH tensor, first threshold off values smaller than eps, then normalizes along the C dimension so each pixel's histogram
    sums to 1.
    """
    x = torch.clamp(x, min=eps)
    x = x / torch.sum(x, dim=1, keepdim=True)
    return x

def sinkhorn_dist_single(cost_mat, lam, gt_hist, x_hist, num_iters=100, eps=1e-4):
    """
    Computes N sinkhorn distances, one for each histogram of length C.
    Works in pytorch
    :param cost_mat: C x C
    :param lam: Controls strength of entropy regularization
    :param gt_hist: N x C. We discard zero entries of this vector since indexing won't backprop well through the x_hist.
    :param x_hist: N x C
    :param num_iters: Number of sinkhorn iterations to run
    :param eps: Algorithm stops when difference between sinkhorn dist is less than this
                for two consecutive iterations.
    :return: sinkhorn_dist, transport_matrix
    """
    # print(gt_hist.shape)
    # print(x_hist.shape)
    assert gt_hist.shape == x_hist.shape and len(x_hist.shape) == 2
    assert cost_mat.shape[0] == x_hist.shape[1] and cost_mat.shape[0] == cost_mat.shape[1]

    r = gt_hist.transpose(0,1) # C x N
    r_mask = (r > eps).squeeze() # Get rid of small entries for numerical stability
    # print(r_mask.shape)
    r = r[r_mask,:]         # C' x N
    c = x_hist.transpose(0,1) # C x N
    M = cost_mat[r_mask, :] # C' x C
    K = torch.exp(-lam*M) # C' x C
    K_T = K.transpose(0,1)  # C x C'
    u = torch.ones_like(r)*r.shape[0] # C' x N
    for i in range(num_iters):
        temp1 = K_T.mm(u) # C x N
        # print(temp1)
        v = c/torch.clamp(temp1, min=eps)         # C x N
        # print(v)
        temp2 = K.mm(v) # C' x N
        # print(temp2)
        u_temp = r/torch.clamp(temp2, min=eps) # C' x N
        # print(u_temp)
        if torch.sum(torch.abs(u - u_temp)) < eps:
            # print("sinkhorn early stopping.")
            break
        if torch.isnan(u_temp).any().item():
            print("iteration {}".format(i))
            print(u)
            break
        u = u_temp
    u_diag = u.transpose(0,1).unsqueeze(-1) # N x C' x 1 # Element wise multiplication with this matrix is equivalent to multiplication by the diagonalization of it
    v_diag = v.transpose(0,1).unsqueeze(1) # N x 1 x C
    K_temp = K.unsqueeze(0) # 1 x C' x C
    P = u_diag * K_temp * v_diag # N x C' x C
    return torch.sum(P*M), P

def optimize_depth_map_masked_adam(x_index_init, mask, sigma, n_bins,
                                   cost_mat, lam, gt_hist,
                                   lr, num_sgd_iters, num_sinkhorn_iters,
                                   kde_eps=1e-5,
                                   sinkhorn_eps=1e-2,
                                   min_sgd_iters=50,
                                   inv_squared_depths=None,
                                   scaling=None,
                                   writer=None, gt=None, model=None):
    """

    :param x_index_init: Initial depth map. Each pixel is an index in [0, n_bins-1]. N x 1 x H x W
    :param mask: Mask off pixels with invalid depth. N x 1 x H x W
    :param sigma: Width parameter for Kernel Density Estimation
    :param n_bins: Number of bins for each histogram.
    :param cost_mat: Cost Matrix for sinkhorn computation. n_bins x n_bins
    :param lam: Controls strength of entropy regularization. Higher = closer to wasserstein.
    :param gt_hist: The histogram we are trying to match x_index_init to.
    :param lr: Learning rate for gradient descent.
    :param num_sgd_iters: Maximum number of gradient descent steps to take.
    :param num_sinkhorn_iters: Maximum number of sinkhorn iteration steps to take.
    :param kde_eps: Epsilon used for KDE to prevent 0 values in histogram.
    :param sinkhorn_eps: Epsilon used to control stopping criterion for the sinkhorn iterations.
    :param min_sgd_iters: Minimum number of SGD iterations to undergo before stopping.
    :param inv_squared_depths: 1/depth^2 for each bin in [0, n_bins-1]
    :param scaling: Per-pixel scaling image.
    :return:
    """
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt

    print("lr: ", lr)
    print("sigma: ", sigma)
    gt_hist = gt_hist.squeeze(-1).squeeze(-1)
    x_index = x_index_init.clone().detach().float().requires_grad_(True)
    x_best = x_index_init.clone().detach().float().requires_grad_(False)
    x_hist_best = spad_forward(x_best, mask, sigma, n_bins, kde_eps=kde_eps,
                          inv_squared_depths=inv_squared_depths,
                          scaling=scaling)

    loss = torch.tensor(float('inf'))
    best_loss = float('inf')
    optimizer = optim.Adam([x_index], lr=lr)
    print("optimizer: {}".format(type(optimizer).__name__))


    for i in range(num_sgd_iters):
        x_hist = spad_forward(x_index, mask, sigma, n_bins, kde_eps=kde_eps,
                              inv_squared_depths=inv_squared_depths,
                              scaling=scaling)
        hist_loss, P = sinkhorn_dist_single(cost_mat, lam,
                                            gt_hist, x_hist,
                                            num_iters=num_sinkhorn_iters,
                                            eps=sinkhorn_eps)

        if not i % 10:
            print("sinkhorn", hist_loss.item())
            print("\tbest so far", best_loss)
        prev_loss = loss.item()
        loss = hist_loss
        optimizer.zero_grad()
        loss.backward()
        with torch.no_grad():
            if loss.item() < best_loss:
                best_loss = loss.item()
                x_best = x_index.clone().detach().float().requires_grad_(False)
                x_hist_best = x_hist
            if torch.isnan(x_index.grad).any().item():
                print("nans detected in x.grad")
                break
            optimizer.step()
            x_index.clamp_(min=0., max=n_bins)
            rel_improvement = np.abs(prev_loss - loss.item())/loss.item()
            if rel_improvement < sinkhorn_eps and i >= min_sgd_iters:
                print("early stopping")
                return x_best, x_hist_best
    return x_best, x_hist_best


def get_extended_hist(img, sid_bin_edges):
    """Returns a numpy array of shape C
    """
    extended_bin_edges = np.append(sid_bin_edges.numpy(), float('inf'))
    img_hist, _ = np.histogram(img, bins=extended_bin_edges)
    return img_hist

if __name__ == '__main__':
    gt = torch.load(os.path.join("data", "test_1.pt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Pretend initialize GT shifted
    gt_shifted = [torch.clamp(gt + shift, min=0., max=10.).unsqueeze(0).to(device)
                  for shift in [0.01, 1., 2., 3.]]

    # Get GT depth histogram
    # sinkhorn_opt = SinkhornOpt(sgd_iters=250, sinkhorn_iters=40, sigma=0.75, lam=1e0, kde_eps=1e-5,
    #                  sinkhorn_eps=1e-7, dc_eps=1e-5,
    #                  remove_dc=False, use_intensity=False, use_squared_falloff=False,
    #                  lr=1e5, min_depth=0., max_depth=10., sid_bins=68,
    #                  alpha=0.6569154266167957, beta=9.972175646365525, offset=0)
    sid_obj = SIDTorch(sid_bins=68, alpha=0.6569154266167957, beta=9.972175646365525, offset=0)
    sid_bins = 68
    C = np.array([[(sid_obj.sid_bin_values[i] - sid_obj.sid_bin_values[j]).item() ** 2 for i in range(sid_bins + 1)]
                  for j in range(sid_bins + 1)])
    cost_mat = torch.from_numpy(C).float()

    mask = torch.ones_like(gt)
    gt_hist = get_extended_hist(gt.numpy(), sid_obj.sid_bin_edges)
    gt_hist = torch.from_numpy(gt_hist).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).float()
    gt_hist /= torch.sum(gt_hist)
    # Run sinkhorn optimization
    # gt_noisy_index = sinkhorn_opt.sid_obj.get_sid_index_from_value(gt_noisy)
    # pred_noisy, pred_noisy_hist = sinkhorn_opt.optimize_depth_map(gt_noisy, None, gt_hist, mask, gt)
    lam = 1e-3
    pred_shifteds_wass = []
    for i, gt_shift in enumerate(gt_shifted):
        gt_shift_index = sid_obj.get_sid_index_from_value(gt_shift).to(device)
        mask = mask.to(device)
        cost_mat = cost_mat.to(device)
        gt_hist = gt_hist.to(device)
        sid_obj.to(device)
        pred_shifted_index, _ = optimize_depth_map_masked_adam(gt_shift_index, mask, sigma=0.75, n_bins=68,
                                                               cost_mat=cost_mat, lam=lam, gt_hist=gt_hist,
                                                               lr=1e-2, num_sgd_iters=5000, num_sinkhorn_iters=40,
                                                               kde_eps=1e-5,
                                                               sinkhorn_eps=1e-7,
                                                               min_sgd_iters=50)
        pred_shifted = sid_obj.get_value_from_sid_index(torch.round(pred_shifted_index).long())
        pred_shifteds_wass.append(pred_shifted)
        ### TESTING
        # if i == 1:
        #     break
        ### END TESTING

    sid_obj.to(torch.device("cpu"))
    cost_mat = cost_mat.cpu()
    # Convert to numpy.
    pred_shifteds_wass = [a.cpu().numpy() for a in pred_shifteds_wass]
    # Save
    np.save("preds.npy", np.concatenate(pred_shifteds_wass, axis=0))
    print("Saved outputs to preds.npy")

    # Display
    # gt_shifted_img
    gt_np = gt.numpy().squeeze()
    fig, axs = plt.subplots(2, 1 + len(gt_shifted), figsize=(50, 15))
    axs[0, 0].imshow(gt_np, vmin=0., vmax=10.)

    gt_hist = get_extended_hist(gt_np, sid_obj.sid_bin_edges)
    axs[1, 0].bar(range(len(gt_hist)), gt_hist / np.sum(gt_hist))
    axs[1, 0].set_xlabel("RMSE: {:1.3f}, WASS: {:1.3f}".format(0, 0), fontsize=25)

    for i, pred_shift in enumerate(pred_shifteds_wass):
        im = axs[0, 1 + i].imshow(pred_shift.squeeze(), vmin=0., vmax=10.)
        shifted_hist = get_extended_hist(pred_shift.squeeze(), sid_obj.sid_bin_edges)
        axs[1, 1 + i].bar(range(len(shifted_hist)), shifted_hist / np.sum(shifted_hist))
        # Calculate and display metrics
        rmse = np.sqrt(np.mean((gt_np - pred_shift) ** 2))
        wass, _ = wasserstein_loss(gt_hist / np.sum(gt_hist),
                                   shifted_hist / np.sum(shifted_hist),
                                   cost_mat.numpy())
        sink, _ = sinkhorn_dist_single(cost_mat, lam,
                                       torch.from_numpy(gt_hist / np.sum(gt_hist)).unsqueeze(0).float(),
                                       torch.from_numpy(shifted_hist / np.sum(shifted_hist)).unsqueeze(0).float()
                                       )
        axs[1, 1 + i].set_xlabel("RMSE: {:1.3f}, WASS: {:1.3f},\nSINK: {:1.3f}".format(rmse, wass, sink.item()),
                                 fontsize=25)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.savefig("output.png")
    print("Saved figure to output.png")
