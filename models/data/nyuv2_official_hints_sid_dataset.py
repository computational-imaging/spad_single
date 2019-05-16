import os

import numpy as np

from torchvision import transforms

from models.data.utils.transforms import (Save, ResizeAll, RandomHorizontalFlipAll, Normalize,
                                          AddDepthMask, ToTensorAll)
from models.data.utils.sid_utils import AddSIDDepth, SID
from models.data.nyuv2_official_nohints_dataset import NYUDepthv2Dataset
from models.data.utils.spad_utils import spad_ingredient, SimulateSpad


from sacred import Experiment

nyuv2_hints_sid_ingredient = Experiment('data_config', ingredients=[spad_ingredient])


@nyuv2_hints_sid_ingredient.config
def cfg():
    data_name = "nyu_depth_v2"
    # Paths should be specified relative to the train script, not this file.
    root_dir = os.path.join("data", "nyu_depth_v2_scaled16")
    train_file = os.path.join(root_dir, "train.json")
    train_dir = root_dir
    val_file = os.path.join(root_dir, "val.json")
    val_dir = root_dir
    test_file = os.path.join(root_dir, "test.json")
    test_dir = root_dir
    del root_dir

    # Indices of images to exclude from the dataset.
    # Set relative to the directory from which the dataset is being loaded.
    blacklist_file = "blacklist.txt"

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

    # Complex procedure to calculate min and max depths
    # to conform to DORN standards
    # i.e. make it so that doing exp(i/25 - 0.36) is the right way to decode depth from a bin value i.
    min_depth = 0.
    max_depth = 10.
    normalization = "dorn" # {"dorn", "none", "train"} # Sets specific normalization if using DORN network.
                                  # If False, defaults to using the empirical mean and variance from train set.

# @nyuv2_hints_sid_ingredient.capture
# def load_config(spad_config):
#     return config

#############
# Load data #
#############
@nyuv2_hints_sid_ingredient.capture
def load_data(train_file, train_dir,
              val_file, val_dir,
              test_file, test_dir,
              min_depth, max_depth, normalization,
              sid_bins, alpha, beta, offset,
              blacklist_file,
              spad_config):
    """Generates training and validation datasets from
    text files and directories. Sets up datasets with transforms.py.
    *_file - string - a text file containing info for DepthDataset to load the images
    *_dir - string - the folder containing the images to load
    min_depth - the minimum depth for this dataset
    max_depth - the maximum depth for this dataset
    normalization - The type of normalization to use.
    sid_bins - the number of Spacing Increasing Discretization bins to add.
    blacklist_file - string - a text file listing, on each line, an image_id of an image to exclude
                              from the dataset.

    test_loader - bool - whether or not to test the loader and not set the dataset-wide mean and
                         variance.

    Returns
    -------
    train, val, test - torch.utils.data.Dataset objects containing the relevant splits
    """
    # print(spad_config)
    train = NYUDepthv2Dataset(train_file, train_dir, transform=None,
                              file_types=["rgb", "albedo", "rawdepth"],
                              min_depth=min_depth, max_depth=max_depth,
                              blacklist_file=blacklist_file)

    train.rgb_mean, train.rgb_var = train.get_mean_and_var()

    # Transform:
    # Size is set to (353, 257) to conform to DORN conventions
    # If normalization == "dorn":
    # Mean is set to np.array([[[103.0626, 115.9029, 123.1516]]]).astype(np.float32) to conform to DORN conventions
    # Var is set to np.ones((1,1,3)) to conform to DORN conventions
    if normalization == "dorn":
        # Use normalization as in the github code for DORN.
        print("Using DORN normalization.")
        transform_mean = np.array([[[103.0626, 115.9029, 123.1516]]]).astype(np.float32)
        transform_var = np.ones((1, 1, 3))
    elif normalization == "none":
        print("No normalization.")
        transform_mean = np.zeros((1, 1, 3))
        transform_var = np.ones((1, 1, 3))
    else:
        transform_mean = train.rgb_mean
        transform_var = train.rgb_var

    train_transform = transforms.Compose([
        AddDepthMask(min_depth, max_depth, "rawdepth"),
        Save(["rgb", "mask", "albedo", "rawdepth"], "_orig"),
        Normalize(transform_mean, transform_var, key="rgb"),
        ResizeAll((353, 257), keys=["rgb", "albedo", "rawdepth"]),
        RandomHorizontalFlipAll(flip_prob=0.5, keys=["rgb", "albedo", "rawdepth"]),
        AddDepthMask(min_depth, max_depth, "rawdepth"), # "mask"
        AddSIDDepth(sid_bins, alpha, beta, offset, "rawdepth"), # "rawdepth_sid"  "rawdepth_sid_index"
        SimulateSpad("rawdepth", "albedo", "mask", "spad", min_depth, max_depth,
                     spad_config["spad_bins"],
                     spad_config["photon_count"],
                     spad_config["dc_count"],
                     spad_config["fwhm_ps"],
                     spad_config["use_albedo"],
                     spad_config["use_squared_falloff"],
                     sid_obj=SID(sid_bins, alpha, beta, offset)),
        ToTensorAll(keys=["rgb", "rgb_orig", "rawdepth", "rawdepth_orig", "albedo", "albedo_orig",
                          "rawdepth_sid", "rawdepth_sid_index", "mask", "mask_orig", "spad"])
        ]
    )

    val_transform = transforms.Compose([
        AddDepthMask(min_depth, max_depth, "rawdepth"),
        Save(["rgb", "mask", "albedo", "rawdepth"], "_orig"),
        Normalize(transform_mean, transform_var, key="rgb"),
        ResizeAll((353, 257), keys=["rgb", "albedo", "rawdepth"]),
        AddDepthMask(min_depth, max_depth, "rawdepth"),
        AddSIDDepth(sid_bins, alpha, beta, offset, "rawdepth"),
        SimulateSpad("rawdepth", "albedo", "mask", "spad", min_depth, max_depth,
                     spad_config["spad_bins"],
                     spad_config["photon_count"],
                     spad_config["dc_count"],
                     spad_config["fwhm_ps"],
                     spad_config["use_albedo"],
                     spad_config["use_squared_falloff"],
                     sid_obj=SID(sid_bins, alpha, beta, offset)),
        ToTensorAll(keys=["rgb", "rgb_orig", "rawdepth", "rawdepth_orig", "albedo", "albedo_orig",
                          "rawdepth_sid", "rawdepth_sid_index", "mask", "mask_orig", "spad"])
        ]
    )

    test_transform = transforms.Compose([
        AddDepthMask(min_depth, max_depth, "rawdepth"),
        Save(["rgb", "mask", "albedo", "rawdepth"], "_orig"),
        Normalize(transform_mean, transform_var, key="rgb"),
        ResizeAll((353, 257), keys=["rgb", "albedo", "rawdepth"]),
        AddDepthMask(min_depth, max_depth, "rawdepth"),
        AddSIDDepth(sid_bins, alpha, beta, offset, "rawdepth"),
        SimulateSpad("rawdepth", "albedo", "mask", "spad", min_depth, max_depth,
                     spad_config["spad_bins"],
                     spad_config["photon_count"],
                     spad_config["dc_count"],
                     spad_config["fwhm_ps"],
                     spad_config["use_albedo"],
                     spad_config["use_squared_falloff"],
                     sid_obj=SID(sid_bins, alpha, beta, offset)),
        ToTensorAll(keys=["rgb", "rgb_orig", "rawdepth", "rawdepth_orig", "albedo", "albedo_orig",
                          "rawdepth_sid", "rawdepth_sid_index", "mask", "mask_orig", "spad"])
        ]
    )
    train.transform = train_transform
    print("Loaded training dataset from {} with size {}.".format(train_file, len(train)))
    val = None
    if val_file is not None:
        val = NYUDepthv2Dataset(val_file, val_dir, transform=val_transform,
                                file_types = ["rgb", "albedo", "rawdepth"],
                                min_depth=min_depth, max_depth=max_depth)
        val.rgb_mean, val.rgb_var = train.rgb_mean, train.rgb_var
        print("Loaded val dataset from {} with size {}.".format(val_file, len(val)))
    test = None
    if test_file is not None:
        test = NYUDepthv2Dataset(test_file, test_dir, transform=test_transform,
                                 file_types = ["rgb", "albedo", "rawdepth"],
                                 min_depth=min_depth, max_depth=max_depth)
        test.rgb_mean, test.rgb_var = train.rgb_mean, train.rgb_var
        print("Loaded test dataset from {} with size {}.".format(test_file, len(test)))

    return train, val, test



###########
# Testing #
###########
@nyuv2_hints_sid_ingredient.automain
def test_load_data(min_depth, max_depth):
    train, val, test = load_data()
    sample = train.get_item_by_id("dining_room_0001a/0001")
    print(sample["spad"])

