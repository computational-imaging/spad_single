import numpy as np
import torch

from scipy.signal import fftconvolve

from sacred import Experiment

spad_ingredient = Experiment("spad_config")

@spad_ingredient.config
def cfg():
    spad_bins = 1024                # Number of bins to capture
    photon_count = 1e6              # Number of real photons that we get
    dc_count = 0.1*photon_count     # Simulates ambient + dark count (additional to photon_count
    fwhm_ps = 70.                   # Full-width-at-half-maximum of (Gaussian) SPAD jitter, in picoseconds

    use_albedo = True
    use_squared_falloff = True


def simulate_spad(depth_truth, albedo, mask, min_depth, max_depth,
                  spad_bins, photon_count, dc_count, fwhm_ps,
                  use_albedo, use_squared_falloff):
    """

    :param depth_truth: The ground truth depth map (z, not distance...)
    :param albedo: The albedo map, aligned with the ground truth depth map.
    :param mask: The mask of valid pixels
    :param min_depth: The minimum depth value (used for the discretization).
    :param max_depth: The maximum depth value (used for the discretization).
    :param spad_bins: The number of spad bins to simulate
    :param photon_count: The number of photons to collect
    :param dc_count: The additional fraction of photons to add to account for dark count + ambient light
    :param fwhm_ps: The full-width-at-half-maximum of the laser pulse jitter
    :param use_albedo: Whether or not to take the albedo into account when simulating.
    :param use_squared_falloff: Whether or not to take the squared depth into account when simulating
    :return: A simulated spad.
    """
    # Only use the green channel to simulate
    weights = mask
    if use_albedo:
        weights = weights * albedo[..., 1]
    if use_squared_falloff:
        weights = weights / (depth_truth ** 2 + 1e-6)
    # weights = (albedo[..., 1] / (depth_truth ** 2 + 1e-6)) * mask
    depth_hist, _ = np.histogram(depth_truth, bins=spad_bins, range=(min_depth, max_depth), weights=weights)

    # Scale by number of photons
    spad_counts = depth_hist * (photon_count / np.sum(depth_hist))

    # Add ambient/dark counts (dc_count)
    spad_counts += np.ones(len(spad_counts)) * (dc_count / spad_bins)

    # Convolve with PSF
    bin_width_m = float(max_depth - min_depth) / spad_bins  # meters/bin
    bin_width_ps = 2 * bin_width_m * 1e12 / (3e8)  # ps/bin, speed of light = 3e8, x2 because light needs to travel there and back.
    fwhm_bin = fwhm_ps / bin_width_ps
    psf = makeGaussianPSF(len(spad_counts), fwhm=fwhm_bin)
    spad_counts = fftconvolve(psf, spad_counts)[:int(spad_bins)]
    spad_counts = np.clip(spad_counts, a_min=0., a_max=None)
    # Apply poisson
    # print(np.min(spad_counts))
    spad_counts = np.random.poisson(spad_counts)
    return spad_counts

# def simulate
#     sid_counts = rescale_bins(spad_counts, sid_bins, min_depth, max_depth)
#     sid_counts = sid_counts/np.sum(sid_counts)
#     # return sid_counts#, spad_counts, depth_hist, psf
#     sid_hist = torch.from_numpy(sid_counts).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).float()
#     sid_hist = sid_hist.to(device)
#     return sid_hist

def makeGaussianPSF(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    x0 = size // 2
    return np.roll(np.exp(-4 * np.log(2) * ((x - x0) ** 2) / fwhm ** 2), len(x) - x0)


def rescale_bins(spad_counts, min_depth, max_depth, sid_obj):
    """

    :param spad_counts: The histogram of spad counts to rescale.
    :param min_depth: The minimum depth of the histogram.
    :param max_depth: The maximum depth of the histogram.
    :param sid_obj: An object representing a SID.
    :return: A rescaled histogram in time to be according to the SID

    Assign photons to sid bins proportionally according to the amount of overlap between
    the sid bin range and the spad_count bin.
    """

    sid_bin_edges_m = sid_obj.sid_bin_edges

    # Convert sid_bin_edges_m into units of spad bins
    sid_bin_edges_bin = sid_bin_edges_m * len(spad_counts) / (max_depth - min_depth)

    # Map spad_counts onto sid_bin indices
    sid_counts = np.zeros(sid_obj.sid_bins)
    for i in range(sid_obj.sid_bins):
        left = sid_bin_edges_bin[i]
        right = sid_bin_edges_bin[i + 1]
        curr = left
        while curr != right:
            curr = np.min([right, np.floor(left + 1.)])  # Don't go across spad bins - stop at integers
            sid_counts[i] += (curr - left) * spad_counts[int(np.floor(left))]
            # Update window
            left = curr
    return sid_counts


class SimulateSpad:
    def __init__(self, depth_truth_key, albedo_key, mask_key, spad_key, min_depth, max_depth,
                 spad_bins, photon_count, dc_count, fwhm_ps, use_albedo, use_squared_falloff,
                 sid_obj=None):
        """

        :param depth_truth_key: Key for ground truth depth in sample.
        :param albedo_key: Key for albedo in sample.
        :param mask_key: Key for mask in sample.
        :param spad_key: Output key for spad counts.
        :param spad_bins: As in simulate_spad
        :param photon_count: As in simulate_spad
        :param dc_count: As in simulate_spad
        :param fwhm_ps: As in simulate_spad
        :param min_depth: As in simulate_spad
        :param max_depth: As in simulate_spad
        :param sid_obj: If not None, rescales histogram using the sid object.
        """
        self.depth_truth_key = depth_truth_key
        self.albedo_key = albedo_key
        self.mask_key = mask_key
        self.spad_key = spad_key
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.sid_obj = sid_obj
        self.use_albedo = use_albedo
        self.use_squared_falloff = use_squared_falloff
        self.simulate_spad_fn = \
            lambda d, a, m: simulate_spad(d, a, m, min_depth, max_depth, spad_bins, photon_count, dc_count,
                                          fwhm_ps, use_albedo, use_squared_falloff)

    def __call__(self, sample):
        spad_counts = self.simulate_spad_fn(sample[self.depth_truth_key],
                                            sample[self.albedo_key],
                                            sample[self.mask_key])
        if self.sid_obj is not None:
            spad_counts = rescale_bins(spad_counts, self.min_depth, self.max_depth, self.sid_obj)
        sample[self.spad_key] = spad_counts/np.sum(spad_counts)

        return sample


# class AddDepthHist: # pylint: disable=too-few-public-methods
#     """Takes a depth map and computes a histogram of depths as well"""
#     def __init__(self, use_albedo=True, use_squared_falloff=True):
#         """
#         kwargs - passthrough to np.histogram
#         """
#         self.use_albedo = use_albedo
#         self.use_squared_falloff = use_squared_falloff
#         self.hist_kwargs =
#
#     def __call__(self, sample):
#         depth = sample["depth"]
#         if "mask" in sample:
#             mask = sample["mask"]
#             depth = depth[mask > 0]
#         weights = np.ones(depth.shape)
#         if self.use_albedo:
#             weights = weights * np.mean(sample["albedo"]) # Attenuate by the average albedo
#         if self.use_squared_falloff:
#             weights[depth == 0] = 0.
#             weights[depth != 0] = weights[depth != 0] / (depth[depth != 0]**2)
#         if not self.use_albedo and not self.use_squared_falloff:
#             sample["hist"], _ = np.histogram(depth, weights= normalize=True)
#         else:
#             sample["hist"], _ = np.histogram(depth, weights=weights, normalize=True)
#         return sample

if __name__ == "__main__":
    pass
