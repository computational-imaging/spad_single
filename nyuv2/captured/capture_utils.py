import h5py
import numpy as np

def loadmat_h5py(file):
    output = {}
    with h5py.File(file, 'r') as f:
        for k, v in f.items():
            output[k] = np.array(v)
    return output


def z_to_r_kinect(z):
    fc = [1053.622, 1047.508]  # Focal length in pixels
    yy, xx = np.meshgrid(range(z.shape[0]), range(z.shape[1]), indexing="ij")
    x = (xx * z) / fc[0]
    y = (yy * z) / fc[1]
    r = np.sqrt(x**2 + y**2 + z**2)
    return r


def r_to_z_kinect(r):
    fc = [1053.622, 1047.508]  # Focal length in pixels
    yy, xx = np.meshgrid(range(r.shape[0]), range(r.shape[1]), indexing="ij")
    z = r / np.sqrt((xx/fc[0])**2 + (yy/fc[1])**2 + 1)
    return z


def rescale_bins(spad_counts, min_depth, max_depth, sid_obj):
    """
    Works in Numpy
    :param spad_counts: The histogram of spad counts to rescale.
    :param min_depth: The minimum depth of the histogram.
    :param max_depth: The maximum depth of the histogram.
    :param sid_obj: An object representing a SID.
    :return: A rescaled histogram in time to be according to the SID

    Assign photons to sid bins proportionally according to the amount of overlap between
    the sid bin range and the spad_count bin.
    """

    sid_bin_edges_m = sid_obj.sid_bin_edges - sid_obj.offset
#     print(sid_bin_edges_m)
    # Convert sid_bin_edges_m into units of spad bins
    sid_bin_edges_bin = sid_bin_edges_m * len(spad_counts) / (max_depth - min_depth)
    sid_bin_edges_bin -= sid_bin_edges_bin[0]  # Start at 0
    sid_bin_edges_bin[-1] = np.floor(sid_bin_edges_bin[-1])
    # print(sid_bin_edges_bin[-1])
    # Map spad_counts onto sid_bin indices
    print(sid_bin_edges_bin)
    sid_counts = np.zeros(sid_obj.sid_bins)
    for i in range(sid_obj.sid_bins):
#         print(i)
        left = sid_bin_edges_bin[i]
        right = sid_bin_edges_bin[i + 1]
        curr = left
        while curr != right:
#             print(curr)
            curr = np.min([right, np.floor(left + 1.)])  # Don't go across spad bins - stop at integers
            sid_counts[i] += (curr - left) * spad_counts[int(np.floor(left))]
            # Update window
            left = curr

    return sid_counts


def normals_from_depth(z):
    """
    Compute surface normals of a depth map.
    """
    # Get GT x and y coords
    fc = [758.2466, 791.2153]  # Focal length of SPAD in pixels
    yy, xx = np.meshgrid(range(z.shape[0]), range(z.shape[1]), indexing="ij")
    x = (xx * z) / fc[0]
    y = (yy * z) / fc[1]
    dzdx = (z[:, 2:] - z[:, :-2])/(x[:, 2:] - x[:, :-2])
    dzdy = (z[:-2, :] - z[2:, :])/(y[:-2, :] - y[2:, :])
    n = np.array((-dzdx[1:-1, :], -dzdy[:, 1:-1], np.ones_like(dzdx[1:-1,:])))
    n /= np.sqrt(np.sum(n**2, axis=0)) # Normalize each vector
    return n



