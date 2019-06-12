import torch
import numpy as np
from pdb import set_trace
from utils.inspect_results import add_hist_plot, log_single_gray_img, add_diff_map

mse = torch.nn.MSELoss()

def get_depth_index(model, input_, device):
    rgb = input_["rgb"].to(device)
    with torch.no_grad():
        x = model(rgb)
        log_probs, _ = model.to_logprobs(x)
        depth_index = torch.sum((log_probs >= np.log(0.5)), dim=1, keepdim=True).long()
    return depth_index


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


def sinkhorn_dist_single(cost_mat, lam, gt_hist, x_hist, num_iters=100, eps=1e-4):
    """
    Computes N sinkhorn distances, one for each histogram of length C.
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
    u = torch.ones_like(r)/r.shape[0] # C' x N
    for i in range(num_iters):
        temp1 = K_T.mm(u) # C x N
        # print(temp1)
        v = c/temp1         # C x N
        # print(v)
        temp2 = K.mm(v) # C' x N
        # print(temp2)
        u_temp = r/temp2 # C' x N
        # print(u_temp)
        if torch.sum(torch.abs(u - u_temp)) < eps:
            print("sinkhorn early stopping.")
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
                       cost_mat, lam, gt_hist,
                       lr, num_sgd_iters, num_sinkhorn_iters,
                       kde_eps=1e-5,
                       sinkhorn_eps=1e-2,
                       inv_squared_depths=None,
                       scaling=None,
                       regularizer=None):
    """

    :param x_index_init:
    :param mask:
    :param sigma:
    :param n_bins:
    :param cost_mat:
    :param lam:
    :param spad_hist:
    :param lr:
    :param num_sgd_iters:
    :param num_sinkhorn_iters:
    :param kde_eps:
    :param sinkhorn_eps:
    :param inv_squared_depths:
    :param scaling:
    :param regularizer:
    :return:
    """
    print("lr: ", lr)
    print("sigma: ", sigma)
    print("regularizer: ", regularizer)
    x0 = x_index_init.clone().detach().float().requires_grad_(False)
    x = x_index_init.clone().detach().float().requires_grad_(True)
    # print(x)
    for i in range(num_sgd_iters):
        # per-pixel histogram to full-image histogram
        x_hist = spad_forward(x, torch.ones_like(x), sigma, n_bins, kde_eps=kde_eps,
                              inv_squared_depths=inv_squared_depths, scaling=scaling)
        # set_trace()
        hist_loss, P = sinkhorn_dist(cost_mat, lam,
                                     x_hist, gt_hist,
                                     num_iters=num_sinkhorn_iters,
                                     eps=sinkhorn_eps)



        if not i % 10:
            print("sinkhorn", hist_loss.item())
            # TESTING: Print Wasserstein Loss
            print(P.shape)
            print(cost_mat.shape)
            print(torch.sum(P * cost_mat))
        # print(hist_loss)
        # entropy_loss = torch.sum(entropy(x_img, dim=1))
        loss = hist_loss
        if regularizer is not None:
            loss = loss + regularizer(x, x0)
        loss.backward()
        with torch.no_grad():
            if torch.isnan(x.grad).any().item():
                            # set_trace()
                print("nans detected in x.grad")
                break
            x -= lr*x.grad
            if torch.sum(torch.abs(lr*x.grad)) < sinkhorn_eps: # Reuse sinkhorn eps for sgd convergence.
                x = torch.clamp(x, min=0., max=n_bins).requires_grad_(True)
                return x, x_hist
            x.grad.zero_()
            x = torch.clamp(x, min=0., max=n_bins).requires_grad_(True)
        # print("warning: sgd exited before convergence.")
    return x, x_hist


def optimize_depth_map_masked(x_index_init, mask, sigma, n_bins,
                              cost_mat, lam, gt_hist,
                              lr, num_sgd_iters, num_sinkhorn_iters,
                              kde_eps=1e-5,
                              sinkhorn_eps=1e-2,
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
    :param inv_squared_depths: 1/depth^2 for each bin in [0, n_bins-1]
    :param scaling: Per-pixel scaling image.
    :return:
    """
    import torchvision.utils as vutils
    import matplotlib.pyplot as plt

    print("lr: ", lr)
    print("sigma: ", sigma)
    gt_hist = gt_hist.squeeze(-1).squeeze(-1)

    if writer is not None:
        # Histogram plot
        add_hist_plot(writer, "hist/gt_hist", gt_hist)

        # RMSE Reference
        # Standard deviation of gt depth
        baseline_rmse = gt[mask > 0].std()
        writer.add_scalar("data/baseline_rmse", baseline_rmse, 0)

    x_index = x_index_init.clone().detach().float().requires_grad_(True)
    x_best = x_index_init.clone().detach().float().requires_grad_(False)
    # if writer is not None:
    #     writer.add_image('depth_in',)
    loss = torch.tensor(float('inf'))
    best_loss = float('inf')
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


        ###
        if writer is not None:
            # Wasserstein Loss
            writer.add_scalar("data/sinkhorn_dist", hist_loss.item(), i)
            # Image itself
            pred_temp = model.sid_obj.get_value_from_sid_index(torch.floor(x_best).detach().long())
            log_single_gray_img(writer, "depth/pred", pred_temp, model.min_depth, model.max_depth, global_step=i)
            # Diff image
            add_diff_map(writer, "depth/diff", gt, pred_temp, i)

            # RMSE
            metrics = model.get_metrics(pred_temp, gt, mask)
            writer.add_scalar("data/rmse", metrics["rmse"], i)

            # Histogram plot
            add_hist_plot(writer, "hist/x_hist", x_hist, global_step=i)
            # Image itself

        ###
        prev_loss = loss.item()
        loss = hist_loss
        loss.backward()
        # Save previous loss for convergence criteria
        with torch.no_grad():
            if loss.item() < best_loss:
                best_loss = loss.item()
                x_best = x_index.clone().detach().float().requires_grad_(False)
            # Do gradient descent step
            if torch.isnan(x_index.grad).any().item():
                print("nans detected in x.grad")
                break
            x_index -= lr*x_index.grad
            # if torch.sum(torch.abs(lr*x_index.grad)) < sinkhorn_eps: # Reuse sinkhorn eps for sgd convergence.
            rel_improvement = np.abs(prev_loss - loss.item())/loss.item()
            print("rel_improvement", rel_improvement)
            # if loss.item() < sinkhorn_eps:
            if rel_improvement < sinkhorn_eps:
                x_index = torch.clamp(x_index, min=0., max=n_bins).requires_grad_(True)
                print("early stopping")
                return x_best, x_hist
            x_index.grad.zero_()
            x_index = torch.clamp(x_index, min=0., max=n_bins).requires_grad_(True)
        # print("warning: sgd exited before convergence.")
    return x_best, x_hist






if __name__ == "__main__":
    gaussian = lambda n, mu, sigma: np.exp(-1. / (2 * sigma ^ 2) * (np.linspace(0, n, n) - mu) ** 2)  # Ground truth depth
    eps = 1e-3
    lam = 1e1
    n = 68  # Number of entries in the histogram
    sigma = 6  # standard dev. of gaussian, in units of bins
    mu_y = 20  # mean, in the interval [0,n]
    y = gaussian(n, mu_y, sigma)
    y[y < eps] = eps
    y = y / np.sum(y)

    # plt.figure()
    # plt.plot(y, label="target")
    # plt.title("Histograms")

    mu_x = 40
    x = gaussian(n, mu_x, sigma)
    x[x < eps] = eps
    x = x / np.sum(x)
    # plt.plot(x, label="initial")
    # plt.legend()

    C = np.array([[(i - j)**2 for j in range(n)] for i in range(n)])/1000.


    ### TEST SINKHORN DISTANCE ###
    # Switch everything to torch and go
    # h1 = torch.from_numpy(x).unsqueeze(0)
    # h2 = torch.from_numpy(y).unsqueeze(0)
    # cost_mat = torch.from_numpy(C)
    # sinkhorn_dist_1, P_1 = sinkhorn_dist_single(cost_mat, lam, h1, h2)
    #
    # print("single", sinkhorn_dist_1, P_1)
    #
    # h1_spatial = h1.unsqueeze(-1).unsqueeze(-1)
    # h2_spatial = h2.unsqueeze(-1).unsqueeze(-1)
    # sinkhorn_dist_2, P_2 = sinkhorn_dist(cost_mat, lam, h1_spatial, h2_spatial)
    #
    # print("spatial", sinkhorn_dist_2, P_2)
    ### DONE ###

    ### TEST ON SMALL IMAGE ###
    kde_eps=1e-2
    n = 10
    C = np.array([[(i - j)**2 for j in range(n+1)] for i in range(n+1)])
    cost_mat = torch.from_numpy(C).float()
    x_index_init = torch.tensor([[3, 7, 9],
                                 [3, 9, 8],
                                 [3, 1, 1]]).unsqueeze(0).unsqueeze(0).float()
    init_hist = spad_forward(x_index_init, torch.ones_like(x_index_init), sigma=0.5, n_bins=n, kde_eps=kde_eps)

    x_truth = torch.tensor([[4, 6, 9],
                            [3, 8, 9],
                            [2, 2, 0]]).unsqueeze(0).unsqueeze(0).float()
    gt_hist = spad_forward(x_truth, torch.ones_like(x_truth), sigma=0.5, n_bins=n, kde_eps=kde_eps)

    # Add noise to enable perfect matching
    x_index_init = torch.clamp(x_index_init + 0.01*torch.randn_like(x_index_init), min=0., max=n)
    print("init", x_index_init)
    print(init_hist)
    x_index_pred, x_hist = optimize_depth_map_masked(x_index_init, torch.ones_like(x_index_init),
                                                     sigma=0.5, n_bins=10,
                                                     cost_mat=cost_mat, lam=1e1, gt_hist=gt_hist,
                                                     lr=1e0, num_sgd_iters=100, num_sinkhorn_iters=40,
                                                     kde_eps=1e-2, sinkhorn_eps=1e-5)
    print("gt", x_truth)
    print("gt_hist", gt_hist)
    print("pred", torch.round(x_index_pred))
    print("pred hist", x_hist)
    ### DONE ###



