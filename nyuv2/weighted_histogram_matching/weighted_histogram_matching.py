import numpy as np


def find_movement(hist_from, hist_to):
    """Gives the movements from hist_from (column sum)
    to hist_to (row sum).
    
    Based on Morovic et. al 2002 A fast, non-iterative, and exact histogram matching algorithm.
    
    hist_from and hist_to should sum to the same value
    """
    movement = np.zeros((len(hist_from), len(hist_to)))
    for row in range(len(hist_from)):
        for col in range(len(hist_to)):
            pixels_rem = hist_from[row] - np.sum(movement[row, :col])
            pixels_req = hist_to[col] - np.sum(movement[:row, col])
            movement[row, col] = np.minimum(pixels_rem, pixels_req)
    return movement


def move_pixels(T, init_index, weights):
    assert init_index.shape == weights.shape
    pred_index = np.zeros_like(init_index)
#     marginal_nonzero = np.nonzero(np.sum(T, axis=1))[0]
    for row in range(init_index.shape[0]):
        for col in range(init_index.shape[1]):
            i = init_index[row, col]
            cpf = np.cumsum(T[i, :])
            p = np.random.uniform(0, cpf[-1])
            for j in range(len(cpf)):
                if cpf[j] >= p:
                    pred_index[row, col] = j
                    T[i, j] = np.maximum(T[i, j] - weights[row, col], 0.)
                    break
    return pred_index


def move_pixels_raster(T_count, init_index, weights):
    assert init_index.shape == weights.shape
    marginal_nonzero = np.nonzero(np.sum(T_count, axis=1))[0]
    pred_index = np.zeros_like(init_index)
    index_sets = [np.where(init_index == bin_index) for bin_index in range(T_count.shape[1])]
    for old_bin in marginal_nonzero:  # Indexes rows
        curr = 0
        for new_bin in range(T_count.shape[1]):  # Indexes columns
            weight_to_move = T_count[old_bin, new_bin]
            rows, cols = index_sets[old_bin]
            weight_moved = 0.
            while weight_moved < weight_to_move and curr < len(rows):
                weight_moved += weights[rows[curr], cols[curr]]
                pred_index[rows[curr], cols[curr]] = new_bin
                curr += 1
    return pred_index
    

def image_histogram_match(init_index, gt_hist, weights, sid_obj):
    init_hist, _ = np.histogram(init_index, weights=weights, bins=range(sid_obj.sid_bins + 1))
    T_count = find_movement(init_hist, gt_hist)
    ## Debugging
    unweighted_hist, _ = np.histogram(init_index, bins=range(sid_obj.sid_bins + 1))
    # pred_index = move_pixels(T_count, init_index, weights)
    pred_index = move_pixels_raster(T_count, init_index, weights)
    pred = sid_obj.get_value_from_sid_index(pred_index)
    pred_hist, _ = np.histogram(pred_index, weights=weights, bins=range(len(gt_hist) + 1))
    return pred_index, pred, pred_hist


def summarize_in_subplot(axs, col, img, hist, gt, gt_hist, title):
    axs[0, col].set_title(title, fontsize=24)
    axs[0, col].imshow(img, vmin=0., vmax=10.)
    axs[1, col].bar(range(len(hist)), hist)
    rmse = np.sqrt(np.mean((gt - img)**2))
    axs[1,col].set_xlabel("RMSE = {:1.3f}".format(rmse), fontsize=24)