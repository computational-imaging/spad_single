import numpy as np


def delta(prediction, target, mask, threshold):
    """
    Given prediction and target, compute the fraction of indices i
    such that
    max(prediction[i]/target[i], target[i]/prediction[i]) < threshold
    """
    c = np.maximum(prediction[mask > 0]/target[mask > 0], target[mask > 0]/prediction[mask > 0])
    # print(c)
    if np.sum(mask) > 0:
        # print((c < threshold).astype(float))
        return np.sum((c < threshold).astype(float)) / (np.sum(mask))
    else:
        return 0.

def mse(prediction, target, mask):
    """
    Return the MSE of prediction and target
    """
    diff = prediction - target
    squares = (diff[mask > 0])**(2)
    out = np.sum(squares)
    total = np.sum(mask).item()
    if total > 0:
        return (1. / np.sum(mask)) * out
    else:
        return 0.

def test_rmse():
    prediction = 2*np.ones(3, 3, 3)
    target = np.zeros(3, 3, 3)
    err = mse(prediction, target, np.ones(3, 3, 3))
    return err

def rel_abs_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative absolute difference:

    1/N*sum(|prediction - target|/target)
    """
    diff = prediction - target
    out = np.sum(np.abs(diff[mask > 0])/(target[mask > 0] + eps))
    total = np.sum(mask).item()
    if total > 0:
        return (1. / np.sum(mask)) * out
    else:
        return 0.

def rel_sqr_diff(prediction, target, mask, eps=1e-6):
    """
    The average relative squared difference:

    1/N*sum(||prediction - target||**2/target)
    """
    diff = prediction - target
    out = np.sum((diff[mask > 0]**2)/(target[mask > 0] + eps))
    total = np.sum(mask).item()
    if total > 0:
        return (1. / np.sum(mask)) * out
    else:
        return 0.

if __name__ == '__main__':
    prediction = np.ones((3, 3))
    target = 0.5*np.ones((3, 3))
    mask = np.array([[1., 1, 1],
                     [0, 0, 0],
                     [1, 1, 1]])
    print("delta1", delta(prediction, target, mask, 1.25))
    print("mse", mse(prediction, target, mask))
    print("rel_abs_diff", rel_abs_diff(prediction, target, mask))
    print("rel_sqr_diff", rel_sqr_diff(prediction, target, mask))
