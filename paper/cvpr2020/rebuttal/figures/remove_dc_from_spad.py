import numpy as np

def remove_dc_from_spad_edge(spad, ambient, grad_th=1e3, n_std=1.):
    """
    Create a "bounding box" that is bounded on the left and the right
    by using the gradients and below by using the ambient estimate.
    """
    # Detect edges:
    assert len(spad.shape) == 1
    edges = np.abs(np.diff(spad)) > grad_th
    print(np.abs(np.diff(spad)))
    first = np.nonzero(edges)[0][1] + 1  # Want the right side of the first edge
    last = np.nonzero(edges)[0][-1]      # Want the left side of the second edge
    below = ambient + n_std*np.sqrt(ambient)
    # Walk first and last backward and forward until we encounter a value below the threshold
    while first >= 0 and spad[first] > below:
        first -= 1
    while last < len(spad) and spad[last] > below:
        last += 1
    spad[:first] = 0.
    spad[last+1:] = 0.
    print(first)
    print(last)
    return np.clip(spad - ambient, a_min=0., a_max=None)
