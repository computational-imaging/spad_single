import numpy as np
from wasserstein import wasserstein_loss

def get_pixel_movements(init_hist, gt_hist, cost_mat, opt_eps=1e-4):
    """
    init_hist: In pixel counts. Shape (N,)
    gt_hist: In pixel counts Shape (M,)
    cost_mat: Shape (M, N)
    opt_eps: Threshold below which we zero out all entries of the T matrix.
    """
    _, T = wasserstein_loss(init_hist/np.sum(init_hist),
                            gt_hist/np.sum(gt_hist),
                            cost_mat, eps_rel=1e-10)
    
    T[T < opt_eps] = 0.
    T[:, np.where(init_hist == 0)] = 0.
    T_marginal = np.sum(T, axis=0)
    marginal_nonzero = np.nonzero(T_marginal)
    T_cond = np.zeros_like(T)
    T_cond[:, marginal_nonzero] = T[:,marginal_nonzero]/T_marginal[marginal_nonzero]

    T_count = T_cond * init_hist
    # Correct last nonzero entry in each column so that np.sum(T_count, axis=0) = init_hist
    for col in range(T_cond.shape[1]):   
        nonzero_rows = np.nonzero(T_cond[:, col])[0]
        if len(nonzero_rows) > 0:
            last_nonzero_row = np.sort(nonzero_rows)[-1]
            T_count[last_nonzero_row, col] += init_hist[col] - np.sum(T_count, axis=0)[col]
    return T_count


def move_pixels_according_to(T_count, init_index, weights):
    assert init_index.shape == weights.shape
    marginal_nonzero = np.nonzero(np.sum(T_count, axis=0))[0]
    pred_index = np.zeros_like(init_index)
    index_sets = [np.where(init_index == bin_index) for bin_index in range(T_count.shape[1])]
    for old_bin in marginal_nonzero: # Indexes columns
        curr = 0
        for new_bin in range(T_count.shape[0]): # Indexes rows
            weight_to_move = T_count[new_bin, old_bin]
            rows, cols = index_sets[old_bin]
            weight_moved = 0.
            while weight_moved < weight_to_move and curr < len(rows):
                weight_moved += weights[rows[curr], cols[curr]]
                pred_index[rows[curr], cols[curr]] = new_bin
                curr += 1
    return pred_index


def wasserstein_match(init_index, gt_hist, cost_mat, weights, sid_obj, opt_eps=1e-4):
    init_hist, _ = np.histogram(init_index, weights=weights, bins=range(sid_obj.sid_bins + 1))
    T_count = get_pixel_movements(init_hist, gt_hist, cost_mat, opt_eps)
    pred_index = move_pixels_according_to(T_count, init_index, weights)
    pred = sid_obj.get_value_from_sid_index(pred_index)
    pred_hist, _ = np.histogram(pred_index, weights=weights, bins=range(len(gt_hist) + 1))
    return pred_index, pred, pred_hist
