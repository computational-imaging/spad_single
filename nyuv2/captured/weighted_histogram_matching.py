# %load weighted_histogram_matching.py
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
            # if np.minimum(pixels_rem, pixels_req) < 0:
            #     print(row, col)
            #     print(hist_from[row])
            #     print(hist_to[col])
            #     print(pixels_rem, pixels_req)
            #     raise Exception()
            movement[row, col] = np.clip(np.minimum(pixels_rem, pixels_req), a_min=0., a_max=None)
    return movement


def move_pixels(T, init_index, weights):
    assert init_index.shape == weights.shape
    pred_index = np.zeros_like(init_index)
    cpfs = np.cumsum(T, axis=1)

    
    for row in range(init_index.shape[0]):
        for col in range(init_index.shape[1]):
            i = init_index[row, col]
            cpf = cpfs[i]
            p = np.random.uniform(0, cpf[-1])
            for j in range(len(cpf)):
                if cpf[j] >= p:
                    pred_index[row, col] = j
                    if T[i,j] > weights[row, col]:
                        # Leave some small positive mass here so that future pixels don't clip to 0.
                        T[i, j] = T[i, j] - weights[row, col]
                    break
    return pred_index


def move_pixels_vectorized(T, init_index, weights):
    assert init_index.shape == weights.shape
    pred_index = np.zeros_like(init_index)
    cpfs = np.cumsum(T, axis=1)  # Sum across columns
    pixel_cpfs = cpfs[init_index, :]  # Per-pixel cpf, cpf goes along axis 2
    p = np.random.uniform(0., pixel_cpfs[..., -1], size=init_index.shape)  # Generate 1 random number for each pixel
    # Use argmax trick to get first index k where p[i,j] < pixel_cpfs[i,j,k] for all i,j
    pred_index = (np.expand_dims(p, 2) < pixel_cpfs).argmax(axis=2)
    return pred_index


def move_pixels_raster(T_count, init_index, weights):
    assert init_index.shape == weights.shape
    marginal_nonzero = np.nonzero(np.sum(T_count, axis=1))[0]
    pred_index = np.zeros_like(init_index)
    index_sets = [np.where(init_index == bin_index) for bin_index in range(T_count.shape[1])]
    for old_bin in marginal_nonzero:  # Indexes rows
        possible_bins = np.nonzero(T_count[old_bin, :])[0]
        curr = 0  # Indexes current pixel
        rows, cols = index_sets[old_bin]
        for new_bin in possible_bins[:-1]:  # Indexes columns
            weight_to_move = T_count[old_bin, new_bin]
            weight_moved = 0.
            while weight_moved < weight_to_move and curr < len(rows):
                pred_index[rows[curr], cols[curr]] = new_bin
                weight_moved += weights[rows[curr], cols[curr]]
                curr += 1
        while curr < len(rows):
            pred_index[rows[curr], cols[curr]] = possible_bins[-1]
            curr += 1
    return pred_index


def image_histogram_match(init, gt_hist, weights, sid_obj):
    weights = weights * (np.sum(gt_hist) / np.sum(weights))
    init_index = np.clip(sid_obj.get_sid_index_from_value(init), a_min=0, a_max=sid_obj.sid_bins - 1)
    init_hist, _ = np.histogram(init_index, weights=weights, bins=range(sid_obj.sid_bins + 1))
    if (gt_hist < 0).any():
        print("Negative values in gt_hist")
        raise Exception()
    T_count = find_movement(init_hist, gt_hist)
#     pred_index = move_pixels_raster(T_count, init_index, weights)
#     pred_index = move_pixels(T_count, init_index, weights)
#     pred_index = move_pixels_better(T_count, init_index, weights)
    pred_index = move_pixels_vectorized(T_count, init_index, weights)
    pred = sid_obj.get_value_from_sid_index(pred_index)
    pred_hist, _ = np.histogram(pred_index, weights=weights, bins=range(len(gt_hist) + 1))
    return pred, (init_index, init_hist, pred_index, pred_hist, T_count)


def image_histogram_match_variable_bin(init, gt_hist, weights, sid_obj_init, sid_obj_pred):
    weights = weights * (np.sum(gt_hist) / np.sum(weights))
    init_index = np.clip(sid_obj_init.get_sid_index_from_value(init),
                         a_min=0, a_max=sid_obj_init.sid_bins - 1)
    init_hist, _ = np.histogram(init_index, weights=weights, bins=range(sid_obj_init.sid_bins + 1))
    if (gt_hist < 0).any():
        print("Negative values in gt_hist")
        raise Exception()
    T_count = find_movement(init_hist, gt_hist)
    pred_index = move_pixels_vectorized(T_count, init_index, weights)
    pred = sid_obj_pred.get_value_from_sid_index(pred_index)
    pred_hist, _ = np.histogram(pred_index, weights=weights, bins=range(len(gt_hist) + 1))
    return pred, (init_index, init_hist, pred_index, pred_hist, T_count)


def image_histogram_match_lin(init, gt_hist, weights, min_depth, max_depth):
    weights = weights * (np.sum(gt_hist) / np.sum(weights))
    n_bins = len(gt_hist)
    bin_edges = np.linspace(min_depth, max_depth, n_bins + 1)
    bin_values = (bin_edges[1:] + bin_edges[:-1])/2
    init_index = np.clip(np.floor((init - min_depth)*n_bins/(max_depth - min_depth)).astype('int'),
                         a_min=0., a_max=n_bins-1)
    init_hist, _ = np.histogram(init_index, weights=weights, bins=range(n_bins+1))
    if (gt_hist < 0).any():
        print("Negative values in gt_hist")
        raise Exception()
    T_count = find_movement(init_hist, gt_hist)
#     pred_index = move_pixels_raster(T_count, init_index, weights)
#     pred_index = move_pixels(T_count, init_index, weights)
#     pred_index = move_pixels_better(T_count, init_index, weights)
    pred_index = move_pixels_vectorized(T_count, init_index, weights)
    pred = np.take(bin_values, pred_index)
    pred_hist, _ = np.histogram(pred_index, weights=weights, bins=range(n_bins+1))
    return pred, (init_index, init_hist, pred_index, pred_hist, T_count)


def summarize_in_subplot(axs, col, img, hist, gt, title):
    axs[0, col].set_title(title, fontsize=24)
    axs[0, col].imshow(img, vmin=0., vmax=10.)
    axs[0, col].axis('off')
    axs[1, col].bar(range(len(hist)), hist)
    rmse = np.sqrt(np.mean((gt - img) ** 2))
    axs[1, col].set_xlabel("RMSE = {:1.3f}".format(rmse), fontsize=24)
