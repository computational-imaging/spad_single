import numpy as np
import torch
import cvxpy as cp
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
    spad_comment = "use_albedo_{}".format(use_albedo) + "_" + \
                   "use_squared_falloff_{}".format(use_squared_falloff) + "_" + \
                   "dc_count_{}".format(dc_count)

@spad_ingredient.named_config
def rawhist():
    dc_count = 0.
    use_albedo = False
    use_squared_falloff = False


def bgr2gray(bgr):
    """
    Numpy / Torch version of cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).
    :param bgr: tensor with channels in (B, G, R) order
    :return: tensor with grayscale image
    """
    if len(bgr.shape) == 4:
        # Shape is N x 3 x H x W
        return 0.2989 * bgr[:,2:3,:,:] + 0.5870 * bgr[:,1:2,:,:] + 0.1140 * bgr[:,0:1,:,:]
    # Otherwise, shape is H x W x 3
    return 0.2989 * bgr[:,:,2:3] + 0.5870 * bgr[:,:,1:2] + 0.1140 * bgr[:,:,0:1]


def simulate_spad(depth_truth, intensity, mask, min_depth, max_depth,
                  spad_bins, photon_count, dc_count, fwhm_ps,
                  use_albedo, use_squared_falloff):
    """
    Works in numpy.
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
        weights = weights * intensity
    if use_squared_falloff:
        weights = weights / (depth_truth ** 2 + 1e-6)
    # weights = (albedo[..., 1] / (depth_truth ** 2 + 1e-6)) * mask
    # print(depth_truth.shape)
    # print(weights.shape)
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
    :return: A rescaled histogram in time to be according to the SIDgit

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


def get_rescale_layer(spad_bins, min_depth, max_depth, sid_obj):
    """
    Returns the linear layer that converts the bins and the rescaled bins.
    Works in torch.
    :param spad_bins: The number of spad_bins
    :param min_depth:
    :param max_depth:
    :param sid_obj:
    :return:
    """
    weights_matrix = torch.zeros(spad_bins, sid_obj.sid_bins)
    for i in range(spad_bins):
        e_i = np.zeros(spad_bins)
        e_i[i] = 1.
        out_i = rescale_bins(e_i, min_depth, max_depth, sid_obj)
        weights_matrix[i, :] = torch.from_numpy(out_i)
    rescale_layer = torch.nn.Linear(spad_bins, sid_obj.sid_bins, bias=False)
    rescale_layer.weight.data = weights_matrix
    return rescale_layer


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
            lambda d, i, m: simulate_spad(d, i, m, min_depth, max_depth, spad_bins, photon_count, dc_count,
                                          fwhm_ps, use_albedo, use_squared_falloff)

    def __call__(self, sample):
        spad_counts = self.simulate_spad_fn(sample[self.depth_truth_key],
                                            sample[self.albedo_key][..., 1],
                                            sample[self.mask_key])
        if self.sid_obj is not None:
            spad_counts = rescale_bins(spad_counts, self.min_depth, self.max_depth, self.sid_obj)
        sample[self.spad_key] = spad_counts/np.sum(spad_counts)
        return sample


class SimulateSpadIntensity:
    def __init__(self, depth_truth_key, rgb_key, mask_key, spad_key, min_depth, max_depth,
                 spad_bins, photon_count, dc_count, fwhm_ps, use_albedo, use_squared_falloff,
                 sid_obj=None):
        """

        :param depth_truth_key: Key for ground truth depth in sample.
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
        self.rgb_key = rgb_key
        self.mask_key = mask_key
        self.spad_key = spad_key
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.sid_obj = sid_obj
        self.use_albedo = use_albedo
        self.use_squared_falloff = use_squared_falloff
        self.simulate_spad_fn = \
            lambda d, i, m: simulate_spad(d, i, m, min_depth, max_depth, spad_bins, photon_count, dc_count,
                                          fwhm_ps, use_albedo, use_squared_falloff)

    def __call__(self, sample):
        print(sample[self.rgb_key].shape)
        spad_counts = self.simulate_spad_fn(sample[self.depth_truth_key],
                                            bgr2gray(sample[self.rgb_key]).squeeze(-1),
                                            sample[self.mask_key])
        if self.sid_obj is not None:
            spad_counts = rescale_bins(spad_counts, self.min_depth, self.max_depth, self.sid_obj)
        sample[self.spad_key] = spad_counts/np.sum(spad_counts)
        return sample


def remove_dc_from_spad_batched(noisy_spad, bin_edges, lam=1e-2, eps=1e-5):
    """
    Batched, operates of batches of size N
    WARNING: Batching generally produces noisier answers.
    Works in numpy.
    :param noisy_spad: length NxC array with the raw spad histogram to denoise.
    :param bin_edges: (C+1) array with the edges of the bins
    """
    # print(noisy_spad.shape)
    # print(bin_widths.shape)

    N, C = noisy_spad.shape
    assert bin_edges.shape == (C+1,)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    # Equalize everything so DC appears uniform
    #     for i in range(N):
    #     spad = noisy_spad[i,:]
    spad_equalized = noisy_spad / bin_widths
    x = cp.Variable((N, C), "signal")
    z = cp.Variable((N, 1), "noise")
    print((x + z).shape)
    obj = cp.Minimize(cp.sum_squares(spad_equalized - (x + z)) + lam * cp.norm(x, 1))
    constr = [
        x >= 0,
        z >= 0
    ]
    prob = cp.Problem(obj, constr)
    prob.solve(solver=cp.OSQP, verbose=True, eps_abs=eps)
    signal_hist = x.value
    signal_hist *= bin_widths
    return signal_hist


def remove_dc_from_spad(noisy_spad, bin_edges, lam=1e-2, eps=1e-5):
    """
    Not batched, solves N convex problems where N is the batch size.
    For some reason, this gives better results.
    Works in numpy.
    :param noisy_spad: NxC array with the raw spad histogram to denoise.
    :param bin_widths: length C array with the bin widths in meters of the original bins.
    :param lam: float value controlling strength of L1 regularization on the signal
    :param eps: float value controlling precision of solver
    """
    # print(noisy_spad.shape)
    # print(bin_widths.shape)
    assert len(noisy_spad.shape) == 2
    N, C = noisy_spad.shape
    assert bin_edges.shape == (C+1,)
    bin_widths = bin_edges[1:] - bin_edges[:-1]

    # Equalize everything so DC appears uniform
    denoised_spad = np.zeros_like(noisy_spad)
    for i in range(N):
        #     spad = noisy_spad[i,:]
        spad_equalized = noisy_spad / bin_widths
        x = cp.Variable((C,), "signal")
        z = cp.Variable((1,), "noise")
        #         print((x+z).shape)
        obj = cp.Minimize(cp.sum_squares(spad_equalized[i, :] - (x + z)) + lam * cp.norm(x, 1))
        constr = [
            x >= 0,
            z >= 0
        ]
        prob = cp.Problem(obj, constr)
        prob.solve(solver=cp.OSQP, eps_abs=eps)
        #         signal_hist = x.value
        denoised_spad[i, :] = x.value * bin_widths
    return denoised_spad


if __name__ == "__main__":
    min_depth = 0.
    max_depth = 10.

    sid_bins = 68   # Number of bins (network outputs 2x this number of channels)
    bin_edges = np.array(range(sid_bins + 1)).astype(np.float32)
    dorn_decode = np.exp((bin_edges - 1) / 25 - 0.36)
    d0 = dorn_decode[0]
    d1 = dorn_decode[1]
    # Algebra stuff to make the depth bins work out exactly like in the
    # original DORN code.
    alpha = (2 * d0 ** 2) / (d1 + d0)
    beta = alpha * np.exp(sid_bins * np.log(2 * d0 / alpha - 1))
    del bin_edges, dorn_decode, d0, d1
    offset = 0.

    from models.data.utils.sid_utils import SID
    sid_obj = SID(sid_bins, alpha, beta, offset)
    layer = get_rescale_layer(1024, min_depth, max_depth, sid_obj)
    print(layer.weight.data[:,0])
