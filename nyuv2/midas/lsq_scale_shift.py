import numpy as np

def lsq_scale_shift(pred, target, mask):
    pred_vec= np.hstack((pred[mask > 0].reshape(-1, 1), np.ones((np.size(pred[mask > 0]), 1))))
    target_vec = target[mask > 0].reshape(-1, 1)
    scaleshift, _, _, _ = np.linalg.lstsq(pred_vec.T.dot(pred_vec), pred_vec.T.dot(target_vec), rcond=None)
    return scaleshift
