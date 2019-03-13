import numpy as np
import torch

def simulate_spad(depth_truth, albedo, spad_bins, photon_count, dc_count, fwhm_ps,
                  mask, min_depth, max_depth, sid_bins, device):
    """
    min_depth, max_depth in meters
    fwhm: given in picoseconds
    """
    #     spad_bin_edges = np.linspace(min_depth, max_depth, spad_bins + 1)
    weights = (albedo / (depth_truth ** 2 + 1e-6)) * mask
    depth_hist, _ = np.histogram(depth_truth, bins=spad_bins, range=(min_depth, max_depth), weights=weights)

    # Scale by number of photons
    #     print(spad_counts.shape)
    spad_counts = depth_hist * (photon_count / np.sum(depth_hist))
    # Add ambient/dark counts (dc_count)
    spad_counts += dc_count / spad_bins * np.ones(len(spad_counts))

    # Convolve with PSF
    bin_width_m = float(max_depth - min_depth) / spad_bins  # meters/bin
    bin_width_ps = 2 * bin_width_m * 1e12 / (
        3e8)  # ps/bin, speed of light = 3e8, x2 because light needs to travel there and back.
    fwhm_bin = fwhm_ps / bin_width_ps
    psf = makeGaussianPSF(len(spad_counts), fwhm=fwhm_bin)
    # print(psf)
    # print(spad_counts)
    spad_counts = fftconvolve(psf, spad_counts)[:len(spad_counts)]
    # print(spad_counts)

    # Apply poisson
    spad_counts = np.random.poisson(spad_counts)
    sid_counts = rescale_bins(spad_counts, sid_bins, min_depth, max_depth)
    sid_counts = sid_counts/np.sum(sid_counts)
    # return sid_counts#, spad_counts, depth_hist, psf
    sid_hist = torch.from_numpy(sid_counts).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).float()
    sid_hist = sid_hist.to(device)
    return sid_hist

def makeGaussianPSF(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    x0 = size // 2
    return np.roll(np.exp(-4 * np.log(2) * ((x - x0) ** 2) / fwhm ** 2), len(x) - x0)


def rescale_bins(spad_counts, sid_bins, min_depth, max_depth):
    """Use bin numbers to do sid discretization.

    Assign photons to sid bins proportionally according to the amount of overlap between
    the sid bin range and the spad_count bin.
    """
    alpha = 1.
    offset = 1.0 - min_depth
    beta = max_depth + offset

    # Get edges of sid bins in meters
    sid_bin_edges_m = np.array([beta ** (float(i) / sid_bins) for i in range(sid_bins + 1)]) - offset

    # Convert sid_bin_edges_m into units of spad bins
    sid_bin_edges_bin = sid_bin_edges_m * len(spad_counts) / (max_depth - min_depth)

    # Map spad_counts onto sid_bin indices
    sid_counts = np.zeros(sid_bins)
    for i in range(sid_bins):
        left = sid_bin_edges_bin[i]
        right = sid_bin_edges_bin[i + 1]
        curr = left
        while curr != right:
            curr = np.min([right, np.floor(left + 1.)])  # Don't go across spad bins - stop at integers
            sid_counts[i] += (curr - left) * spad_counts[int(np.floor(left))]
            # Update window
            left = curr
    return sid_counts