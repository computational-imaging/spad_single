import torch
import numpy as np
import os
import json

from models.data.data_utils.sid_utils import SID

def get_latest_config(results_dir):
    """
    Given a results directory, collects the config for the latest run (highest index).
    """
    runs_dir = os.path.join(results_dir, "runs")
    latest = str(max(int(s) for s in os.listdir(runs_dir) if not s.startswith("_")))
    config_file = os.path.join(runs_dir, latest, "config.json")
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config


def load_dataset_from_config(config, load_hints, load_nohints):
    """
    Given a config, load the dataset corresponding to the (hints/nohints) dataset we want.
    """
    data_config = config["data_config"]
    # Future: Check data_name to choose the right dataset
    if "data_name" in data_config:
        del data_config["data_name"]
    spad_config = None
    if "spad_config" in config:
        # With hints (probably)
        spad_config = config["spad_config"]
        train, val, test = load_hints(**data_config, spad_config=spad_config)
    else:
        # No hints (probably)
        train, val, test = load_nohints(**data_config)
    if config["dataset_type"] == "val":
        dataset = val
    elif config["dataset_type"] == "test":
        dataset = test
    else:
        raise ValueError("Unknown dataset type: {}".format(config["dataset_type"]))
    return dataset


def load_entry_from_results_dir(entry, results_dir, load_hints, load_nohints, config=None):
    if config is None:
        config = get_latest_config(results_dir)
    dataset = load_dataset_from_config(config, load_hints, load_nohints)

    input_ = dataset.get_item_by_id(entry)
    print(input_.keys())
    sid_obj = SID(config["data_config"]["sid_bins"],
                  config["data_config"]["alpha"],
                  config["data_config"]["beta"],
                  config["data_config"]["offset"])
    data = {"rgb": input_["rgb_orig"].numpy().transpose(1,2,0)/255.,
            "gt": input_["rawdepth_orig"].numpy().transpose(1,2,0).squeeze(),
#             "gt": input_["depth"].numpy().transpose(1,2,0).squeeze(),
            "mask": input_["mask_orig"].numpy().squeeze(),
            "sid": sid_obj,
            "min_depth": config["data_config"]["min_depth"],
            "max_depth": config["data_config"]["max_depth"],
            "gt_hist": np.histogram(input_["rawdepth_orig"], bins=sid_obj.sid_bin_edges)
    }

    if "spad" in input_:
        # Show this one too
        data["spad"] = input_["spad"].numpy().squeeze()
        data["albedo"] = input_["albedo_orig"].numpy().transpose(1,2,0).squeeze()/255.

    # Load the outputs to the network from the file
    outfile = os.path.join(results_dir, entry + "_out.pt")
    output = torch.load(outfile)
    data["pred"] = output["pred"].numpy().squeeze()
    data["pred_hist"] = np.histogram(output["pred"], bins=sid_obj.sid_bin_edges)
    return data

# Simulated histogram from ground truth depth plus albedo:
from models.sinkhorn_dist import kernel_density_estimation
def simulate_spad_from_depth_map(depth, albedo, mask, sid, sigma=0.5, n_bins=68, kde_eps=1e-4):
    """
    Do a feed-foward differentiable approximation to what the SPAD would see.
    Uses Kernel Density Estimation with parameter sigma.
    :param depth: HxW numpy array
    :param albedo: HxWx3 numpy array
    :param mask: HxW numpy array
    
    :returns: numpy array with simulated spad
    """
    one_over_depth_squared = torch.from_numpy(1./(sid.sid_bin_values[:-2] ** 2)).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).float()
    albedo = torch.from_numpy(albedo.transpose(2, 0, 1)[1,:,:]).unsqueeze(0).unsqueeze(0)
    index = torch.from_numpy(sid.get_sid_index_from_value(depth)).unsqueeze(0).unsqueeze(0).float()
    mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0)
    per_pixel_hist_masked = kernel_density_estimation(index, sigma, n_bins, kde_eps) * mask
    albedo_falloff_hist = img_to_hist(per_pixel_hist_masked, inv_squared_depths=one_over_depth_squared, albedo=albedo)
    return albedo_falloff_hist.numpy().squeeze()
