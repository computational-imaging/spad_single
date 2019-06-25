import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":
    gaussian = lambda n, mu, sigma: np.exp(-1./(2*sigma^2)*(np.linspace(0, n, n) - mu)**2) # Ground truth depth
    eps = 1e-3
    n = 68 # Number of entries in the histogram
    sigma = 6 # standard dev. of gaussian, in units of bins
    mu_y = 20 # mean, in the interval [0,n]
    y = gaussian(n, mu_y, sigma)
    y[y < eps] = 0.
    y = y / np.sum(y)

    plt.figure()
    plt.plot(y, label="target")
    plt.title("Histograms")

    mu_x = 40
    x = gaussian(n, mu_x, sigma)
    x[x < eps] = 0.
    x = x / np.sum(x)
    plt.plot(x, label="initial")
    plt.legend()