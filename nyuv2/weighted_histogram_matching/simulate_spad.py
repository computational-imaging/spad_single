#! /usr/bin/env python3
import numpy as np
import os
import torch
from torch.utils.data._utils.collate import default_collate
import cvxpy as cp
from scipy.signal import fftconvolve
from sacred import Experiment
from nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data
from spad_utils import simulate_spad

spad_ingredient = Experiment("spad_config", ingredients=[nyuv2_labeled_ingredient])

@spad_ingredient.config
def cfg(data_config):
    spad_bins = 1024                # Number of bins to capture
    photon_count = 1e6              # Number of real photons that we get
    dc_count = 0.1*photon_count     # Simulates ambient + dark count (additional to photon_count
    fwhm_ps = 70.                   # Full-width-at-half-maximum of (Gaussian) SPAD jitter, in picoseconds

    use_poisson = True
    use_intensity = True
    use_squared_falloff = True
    use_jitter = True

    min_depth = data_config["min_depth"]
    max_depth = data_config["max_depth"]

    # "test" or "train"
    dataset_type = "test"

    # Output directory
    output_dir = "data"

@spad_ingredient.capture
def simulate_spad_passthrough(depth_truth, intensity, mask, min_depth, max_depth,
                              spad_bins, photon_count, dc_count, fwhm_ps,
                              use_poisson, use_intensity, use_squared_falloff,
                              use_jitter):
    return simulate_spad(depth_truth, intensity, mask, min_depth, max_depth,
                  spad_bins, photon_count, dc_count, fwhm_ps,
                  use_poisson, use_intensity, use_squared_falloff,
                  use_jitter)

def rgb2gray(img):
    return 0.2989 * img[:, 0:1, ...] + 0.5870 * img[:, 1:2, ...] + 0.1140 * img[:, 2:3, ...]

@spad_ingredient.automain
def run(dataset_type,
        output_dir,
        use_intensity,
        use_squared_falloff,
        dc_count,
        use_jitter,
        use_poisson,
        _config):  # The entire config dict for this experiment
    print("dataset_type: {}".format(dataset_type))
    dataset = load_data(dataset_type)
    all_spad_counts = []
    all_intensities = []
    for i in range(len(dataset)):
        print("Simulating SPAD for entry {}".format(i))
        data = default_collate([dataset[i]])

        intensity = rgb2gray(data["rgb_cropped"].numpy()/255.)
        # print(intensity.shape)
        depth_truth = data["depth_cropped"].numpy()
        spad_counts = simulate_spad_passthrough(depth_truth=depth_truth,
                                                intensity=intensity,
                                                mask=np.ones_like(depth_truth))
        all_spad_counts.append(spad_counts)
        all_intensities.append(intensity)

    output = {
        "config": _config,
        "spad": np.array(all_spad_counts),
        "intensity": np.concatenate(all_intensities),
    }

    print("saving {}_int_{}_fall_{}_dc_{}_jit_{}_poiss_{}_spad.npy to {}".format(dataset_type,
                                                                 use_intensity, use_squared_falloff, dc_count,
                                                                 use_jitter, use_poisson,
                                                                 output_dir))
    np.save(os.path.join(output_dir, "{}_int_{}_fall_{}_dc_{}_jit_{}_poiss_{}_spad.npy".format(dataset_type,
                                                                               use_intensity,
                                                                               use_squared_falloff,
                                                                               dc_count, use_jitter,
                                                                               use_poisson)), output)
