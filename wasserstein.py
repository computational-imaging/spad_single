import numpy as np
import cvxpy as cp

def wasserstein_loss(hist0, hist1, cost_mat, **solver_kwargs):
    """
    :param hist0: (N,) numpy array, summing to 1
    :param hist1: (N,) numpy array, summing to 1
    :param cost_mat: (N, N) numpy array
    """
    assert hist0.shape == hist1.shape
    assert len(hist0.shape) == 1
    assert len(cost_mat.shape) == 2
    assert cost_mat.shape[0] == cost_mat.shape[1]
    assert hist0.shape[0] == cost_mat.shape[0]
    N = hist0.shape[0]
    T = cp.Variable((N, N))
    obj = cp.Minimize(cp.trace(T.T*cost_mat))
    constr = [
        cp.sum(T, axis=0) == hist0,
        cp.sum(T, axis=1) == hist1,
        T >= 0.
    ]
    prob = cp.Problem(obj, constr)
    prob.solve(solver="OSQP", **solver_kwargs)
    return prob.value, T.value