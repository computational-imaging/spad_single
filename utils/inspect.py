from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

def show(img):
    """Displays a 3-channel RGB image or a 1-channel Grayscale image.
    Input should have 3 dimensions, with the first being the number of channels
    (as is standard in pytorch).
    """
    plt.figure()
    npimg = img.cpu().numpy()
    if npimg.shape[0] == 1:  # Single-channel image
        npimg = np.concatenate([npimg, npimg, npimg], axis=0)
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')


def show_hist(hist, min_depth, max_depth):
    """Input should be a 1d numpy array with the value for each
    bin specified.

    In particular, this does NOT compute a histogram on hist before displaying: it assumes
    that bar-graphing hist is what will produce the histogram.
    """
    fig = plt.figure(figsize=(5, 2))
    ax = fig.add_subplot(111)
    ax.bar(np.linspace(min_depth, max_depth, len(hist)), hist, width=(max_depth - min_depth) / len(hist))


def get_loss_diffs(nohints_losses, hints_losses):
    loss_names = set()
    loss_diffs = defaultdict(dict)
    for entry in nohints_losses:
        if entry in hints_losses:
            for loss_name in nohints_losses[entry]:
                loss_names.add(loss_name)
                loss_diffs[entry][loss_name] = hints_losses[entry][loss_name] - nohints_losses[entry][loss_name]
        else:
            raise KeyError("missing corresponding hints entry: {}".format(entry))
    return loss_diffs, loss_names


def find_max_differential(loss_diffs, loss_names):
    sorted_loss_diffs = defaultdict(dict)
    for loss_name in loss_names:
        sorted_loss_diffs[loss_name] = sorted(loss_diffs.items(), key=lambda x: x[1][loss_name])
    return sorted_loss_diffs
    # Sort by highest loss differential to lowest
    # delta1_sorted = sorted(loss_diffs.items(), key=lambda x: x[1]["delta1"], reverse=True)
    # delta2_sorted = sorted(loss_diffs.items(), key=lambda x: x[1]["delta2"], reverse=True)
    # delta3_sorted = sorted(loss_diffs.items(), key=lambda x: x[1]["delta3"], reverse=True)
    # rmse_sorted = sorted(loss_diffs.items(), key=lambda x: x[1]["rmse"], reverse=False)
    # rel_abs_diff_sorted = sorted(loss_diffs.items(), key=lambda x: x[1]["rel_abs_diff"], reverse=False)
    # rel_sqr_diff_sorted = sorted(loss_diffs.items(), key=lambda x: x[1]["rel_sqr_diff"], reverse=False)

#
# def find_pixelwise_diffs(img1, img2):
#     """
#     Return a heatmap showing where img1 and img2 differ the most from each other.
#     :param img1:
#     :param img2:
#     :return: absolute value of difference
#     """
#     return np.abs(img1 - img2)


