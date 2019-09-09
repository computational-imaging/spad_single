import cvxpy as cp
import numpy as np

def remove_dc_from_spad(noisy_spad, bin_edges, bin_weight, lam=1e-2, eps_rel=1e-5):
    """
    Works in numpy.
    :param noisy_spad: Length C array with the raw spad histogram to denoise.
    :param bin_edges: Length C+1 array with the bin widths in meters of the original bins.
    :param bin_weight: Length C nonnegative array controlling relative strength of L1 regularization on each bin.
    :param lam: float value controlling strength of overall L1 regularization on the signal
    :param eps: float value controlling precision of solver
    """
    assert len(noisy_spad.shape) == 1
    C = noisy_spad.shape[0]
    assert bin_edges.shape == (C+1,)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    spad_equalized = noisy_spad / bin_widths
    x = cp.Variable((C,), "signal")
    z = cp.Variable((1,), "noise")
    print("lam", lam)
    print("eps_rel", eps_rel)
    # print("spad_equalized", spad_equalized)
    orig_sum = np.sum(spad_equalized)
    spad_normalized = spad_equalized/orig_sum
    obj = cp.Minimize(cp.sum_squares(spad_normalized - (x + z)) + lam * cp.sum(bin_weight*cp.abs(x)))
    constr = [
        x >= 0,
        # z >= 0
    ]
    prob = cp.Problem(obj, constr)
    prob.solve(solver=cp.OSQP, eps_rel=eps_rel)
    print("z.value", z.value)
    denoised_spad = np.clip(x.value * bin_widths, a_min=0., a_max=None)
    return denoised_spad


def remove_dc_from_spad_test(noisy_spad, bin_edges, bin_weight,
                                 use_anscombe, use_quad_over_lin,
                                 use_poisson, use_squared_falloff,
                                 lam1=1e-2, lam2=1e-1, eps_rel=1e-5):
    def anscombe(x):
        return 2*np.sqrt(x + 3./8)
    def inv_anscombe(x):
        return (x/2)**2 - 3./8
    assert len(noisy_spad.shape) == 1
    C = noisy_spad.shape[0]
    
    assert bin_edges.shape == (C+1,)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    spad_equalized = noisy_spad / bin_widths
    x = cp.Variable((C,), "signal")
    z = cp.Variable((1,), "dc")
    nx = cp.Variable((C,), "signal noise")
    nz = cp.Variable((C,), "dc noise")
    if use_poisson:
        # Need tricky stuff
        if use_anscombe:
#             plt.figure()
#             plt.bar(range(len(spad_equalized)), spad_equalized, log=True)
#             plt.title("Before")
    #         d_ans = cp.Variable((C,), "denoised anscombe")
            # Apply Anscombe Transform to data:
            spad_ans = anscombe(spad_equalized)
            # Apply median filter to remove Gaussian Noise
            spad_ans_filt = scipy.signal.medfilt(spad_ans, kernel_size=15)
            # Apply Inverse Anscombe Transform
            spad_equalized = inv_anscombe(spad_ans_filt)
#             plt.figure()
#             plt.bar(range(len(spad_equalized)), spad_equalized, log=True)
#             plt.title("After")

        if use_quad_over_lin:
            obj = \
                    cp.sum([cp.quad_over_lin(nx[i], x[i]) for i in range(C)]) + \
                    cp.sum([cp.quad_over_lin(nz[i], z) for i in range(C)]) + \
                    lam2 * cp.sum(bin_weight*cp.abs(x))
            constr = [
                x >= 0,
                x + nx >= 0,
                z >= cp.min(spad_equalized),
                z + nz >= cp.min(spad_equalized),
                x + nx + z + nz == spad_equalized
            ]
            prob = cp.Problem(cp.Minimize(obj), constr)
            prob.solve(solver=cp.ECOS, verbose=True, reltol=eps_rel)
        else:
            obj = cp.sum_squares(spad_equalized - (x + z)) + lam2 * cp.sum(bin_weight*cp.abs(x))
            constr = [
                x >= 0,
                z >= 0
            ]
            prob = cp.Problem(cp.Minimize(obj), constr)
            prob.solve(solver=cp.OSQP, verbose=True, eps_rel=eps_rel)
    else:
        # No need for tricky stuff
        obj = cp.sum_squares(spad_equalized - (x + z)) + 1e0 * cp.sum(bin_weight*cp.abs(x))
        constr = [
            x >= 0,
            z >= 0
        ]
        prob = cp.Problem(cp.Minimize(obj), constr)
        prob.solve(solver=cp.OSQP, eps_rel=eps_rel)
    denoised_spad = np.clip(x.value * bin_widths, a_min=0., a_max=None)
    print("z.value", z.value)
    return denoised_spad


def remove_dc_from_spad_poisson(noisy_spad, bin_edges,
                                lam=1e-1):
    assert len(noisy_spad.shape) == 1
    C = noisy_spad.shape[0]
    assert bin_edges.shape == (C+1,)
    bin_widths = bin_edges[1:] - bin_edges[:-1]
    spad_equalized = noisy_spad / bin_widths
    orig_sum = np.sum(spad_equalized)
    spad_normalized = spad_equalized/orig_sum
    x = cp.Variable((C,), "signal")
    z = cp.Variable((1,), "dc")
    obj = -spad_normalized*cp.log(z + x) + cp.sum(z + x) + \
          lam*cp.norm(x, 2)
          # lam*cp.sum_squares(x)
#           lam*cp.square(cp.norm(cp.multiply(bin_weight, x), 2))
#     obj = -spad_equalized*cp.log(z + x) + cp.sum(z + x) + lam*cp.sum_squares(x)
    constr = [
        z >= 0,
        x >= 0
    ]

    prob = cp.Problem(cp.Minimize(obj), constr)
    prob.solve(verbose=False, gp=False)
#     noise = np.exp(z.value)
#     signal = np.exp(x.value)
    noise = z.value
    signal = x.value
#     zero_out = x.value
#     zero_out[zero_out < 1e0] = 0.
    frac = orig_sum *np.sum(signal)/(np.sum(signal) + len(spad_equalized)*noise)
    denoised_spad = np.clip(frac * signal * bin_widths, a_min=0., a_max=None)
    return denoised_spad



def remove_dc_from_spad_ambient_estimate(spad, min_depth, max_depth, global_min_depth, n_std=1., axs=None):
    assert len(spad.shape) == 1
    assert global_min_depth > min_depth
    def anscombe(x):
        return 2*np.sqrt(x + 3./8)
    def inv_anscombe(x):
        return (x/2)**2 - 3./8
    N = len(spad)
    # Estimate ambient level from first few bins of spad
    t = (global_min_depth - min_depth)/(max_depth - min_depth)
    M = np.mean(spad[:int(np.floor(t*N))])
    # print(M)
#     spad = anscombe(spad)
#     spad = scipy.signal.medfilt(spad, kernel_size=5)
#     spad = inv_anscombe(spad)
    cutoff = M + n_std*np.sqrt(M)
    spad_denoised = np.clip((spad - cutoff), a_min=0., a_max=None)
    if axs is not None:
        axs[0].bar(range(len(spad)), spad, log=True)
        axs[0].axhline(y=cutoff, color='r', linewidth=0.5)
    return spad_denoised


import cv2
def sobel_edge_detection(x, thresh, ax=None):
    grad = cv2.Sobel(x, cv2.CV_64F, 1, 0)
    if ax is not None:
        ax.bar(range(len(grad.squeeze())), np.abs(grad.squeeze()) / 2, log=True)
        ax.axhline(y=thresh, color='r', linewidth=0.5)
    return np.abs(grad) / 2 >= thresh


# def remove_dc_from_spad_edge(spad, ambient, grad_th=1e3, n_std=1.):
#     """
#     Create a "bounding box" that is bounded on the left and the right
#     by using the gradients and below by using the ambient estimate.
#     """
#     # Detect edges:
#     assert len(spad.shape) == 1
#     spad_2d = spad.reshape(1, -1).astype("float")
#     edges = sobel_edge_detection(spad_2d, grad_th)
#     first = np.nonzero(edges)[1][0]
#     last = np.nonzero(edges)[1][-1]
#     below = ambient + n_std * np.sqrt(ambient)
#     # Walk first and last backward and forward until we encounter a value below the threshold
#     while first >= 0 and spad[first] > below:
#         first -= 1
#     while last < len(spad) and spad[last] > below:
#         last += 1
#     spad[:first] = 0.
#     spad[last + 1:] = 0.
#     return np.clip(spad - ambient, a_min=0., a_max=None)


def remove_dc_from_spad_edge(spad, ambient, grad_th=1e3, n_std=1.):
    """
    Create a "bounding box" that is bounded on the left and the right
    by using the gradients and below by using the ambient estimate.
    :param spad:
    :param ambient: per-bin estimate of how many photons we should see if the bin is all ambient
    :param grad_th:
    :param n_std:
    """
    # Detect edges:
    assert len(spad.shape) == 1
    edges = np.abs(np.diff(spad)) > grad_th
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
    return np.clip(spad - ambient, a_min=0., a_max=None)

import sklearn.mixture as skmix
def remove_dc_from_spad_gmm(h, n_components=4, weight_concentration_prior=1e0, depth_values=None, axs=None):
    assert len(h.shape) == 1
    h_denoised = h.copy().astype('float')
    h_nz = h[h > 0].copy()

    model = skmix.BayesianGaussianMixture(n_components=n_components,
                                          weight_concentration_prior=weight_concentration_prior)
    if depth_values is None:
        classes = model.fit_predict(np.log(h_nz).reshape(-1,1))
    else:
        depth_values_nz = depth_values[h > 0]
        classes = model.fit_predict(np.stack([np.log(h_nz), depth_values_nz], axis=-1))

    noise_class = min((np.mean(h_nz[classes == i]), i) for i in np.unique(classes))[1]
    cutoff = (np.max(h_nz[classes == noise_class]) + np.min(h_nz[classes != noise_class])) / 2
    if axs is not None:
        axs[0].bar(range(len(h)), h, log=True)
        axs[0].axhline(y=cutoff, color='r', linewidth=0.5)
        h_noise, _ = np.histogram(np.log(h_nz[classes == noise_class]), bins=200,
                                  range=(np.min(np.log(h_nz)), np.max(np.log(h_nz))))
        h_signal, _ = np.histogram(np.log(h_nz[classes != noise_class]), bins=200,
                                   range=(np.min(np.log(h_nz)), np.max(np.log(h_nz))))
        axs[1].bar(range(len(h_noise)), h_noise)
        axs[1].bar(range(len(h_signal)), h_signal)
    h_denoised[h_denoised <= cutoff] = 0.
    dc = np.mean(h_nz[classes == noise_class])
    h_denoised[h_denoised > cutoff] -= dc
    return h_denoised

# import scipy
# def remove_dc_from_spad_ambient_estimate(spad, min_depth, max_depth, global_min_depth, n_std=1., axs=None):
#     assert len(spad.shape) == 1
#     assert global_min_depth > min_depth
#
#     def anscombe(x):
#         return 2 * np.sqrt(x + 3. / 8)
#
#     def inv_anscombe(x):
#         return (x / 2) ** 2 - 3. / 8
#
#     N = len(spad)
#     # Estimate ambient level from first few bins of spad
#     t = (global_min_depth - min_depth) / (max_depth - min_depth)
#     M = np.mean(spad[:int(np.floor(t * N))])
#     spad = anscombe(spad)
#     spad = scipy.signal.medfilt(spad, kernel_size=5)
#     spad = inv_anscombe(spad)
#     cutoff = M + n_std * np.sqrt(M)
#     spad_denoised = np.clip((spad - cutoff), a_min=0., a_max=None)
#     if axs is not None:
#         axs[0].bar(range(len(spad)), spad, log=True)
#         axs[0].axhline(y=cutoff, color='r', linewidth=0.5)
#     return spad_denoised


def preprocess_spad_ambient_estimate(spad, min_depth, max_depth, correct_falloff, remove_dc, **opt_kwargs):
    if remove_dc:
        spad = remove_dc_from_spad_ambient_estimate(spad, min_depth, max_depth,
                                                    global_min_depth=opt_kwargs["global_min_depth"],
                                                    n_std=opt_kwargs["n_std"])
    if correct_falloff:
        spad = spad * (np.linspace(min_depth, max_depth, len(spad))) ** 2
    return spad


def preprocess_spad_sid_gmm(spad_sid, sid_obj, correct_falloff, remove_dc,
                            **opt_kwargs):
    if remove_dc:
#         bin_widths = sid_obj.sid_bin_edges[1:] - sid_obj.sid_bin_edges[:-1]
#         spad_sid = spad_sid / bin_widths
        spad_sid = remove_dc_from_spad_gmm(spad_sid, **opt_kwargs)
#         spad_sid = spad_eq_denoised * bin_widths
    if correct_falloff:
        spad_sid = spad_sid * sid_obj.sid_bin_values[:-2]**2
    return spad_sid


def preprocess_spad_gmm(spad, min_depth, max_depth, correct_falloff, remove_dc,
                        **opt_kwargs):
    if remove_dc:
        spad = remove_dc_from_spad_gmm(spad, #depth_values=np.linspace(min_depth, max_depth, len(spad)),
                                       axs=opt_kwargs["axs"])
    if correct_falloff:
        spad = spad * (np.linspace(min_depth, max_depth, len(spad)))**2
    return spad


def preprocess_spad_sid_poisson(spad_sid, sid_obj, correct_falloff, remove_dc,
                                **opt_kwargs):
    if remove_dc:
        spad_sid = remove_dc_from_spad_poisson(spad_sid,
                                               sid_obj.sid_bin_edges,
                                               sid_obj.sid_bin_values[:-2]**2,
                                               **opt_kwargs)
    if correct_falloff:
        spad_sid = spad_sid * sid_obj.sid_bin_values[:-2]**2
    return spad_sid


def preprocess_spad_poisson(spad_sid, min_depth, max_depth, correct_falloff, remove_dc,
                            **opt_kwargs):
    if remove_dc:
        spad_sid = remove_dc_from_spad_poisson(spad_sid,
                                               np.array(range(len(spad) + 1)).astype('float'),
                                               np.linspace(min_depth, max_depth, len(spad))**2+1e-4,
                                               **opt_kwargs)
    if correct_falloff:
        spad_sid = spad_sid * (np.linspace(min_depth, max_depth, len(spad)))**2
    return spad_sid
# def remove_dc_from_spad(noisy_spad, bin_edges, bin_weight, lam=1e-2, eps_rel=1e-5):


def preprocess_spad_sid(spad_sid, sid_obj, correct_falloff, remove_dc, 
                        **opt_kwargs):
    if remove_dc:
        spad_sid = remove_dc_from_spad(spad_sid,
                                       sid_obj.sid_bin_edges,
                                       sid_obj.sid_bin_values[:-2]**2,
                                       **opt_kwargs)
    if correct_falloff:
        spad_sid = spad_sid * sid_obj.sid_bin_values[:-2]**2
    return spad_sid

def preprocess_spad(spad, min_depth, max_depth, correct_falloff, remove_dc,
                    use_anscombe, use_quad_over_lin, use_poisson, **opt_kwargs):
    if remove_dc:
        spad = remove_dc_from_spad_test(spad,
                                        np.array(range(len(spad) + 1)).astype('float'),
                                        np.linspace(min_depth, max_depth, len(spad))**2+1e-4,
                                        use_anscombe,
                                        use_quad_over_lin,
                                        use_poisson,
                                        correct_falloff,
                                        **opt_kwargs)
    if correct_falloff:
        spad = spad * (np.linspace(min_depth, max_depth, len(spad)))**2
    return spad
