import os
import random
import json

import numpy as np
import torch
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import cv2

from torchvision import transforms, utils
from models.data.transforms import (ResizeAll, RandomHorizontalFlipAll, Normalize,
                                    AddDepthMask, ToTensorAll)
from models.data.sid_utils import AddSIDDepth

from sacred import Experiment

from .nyuv2_official_nohints_dataset import NYUDepthv2Dataset

nyuv2_nohints_sid_ingredient = Experiment('data_config')


@nyuv2_nohints_sid_ingredient.config
def cfg():
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
    min_depth = 0.  # Minimum depth
    max_depth = 10. # Maximum depth
    use_dorn_normalization = True # Sets specific normalization if using DORN network.
                                  # If False, defaults to using the empirical mean and variance from train set.
    sid_bins = 68   # Number of bins (network outputs 2x this number of channels)


#############
# Load data #
#############
@nyuv2_nohints_sid_ingredient.capture
def load_data(train_file, train_dir,
              val_file, val_dir,
              test_file, test_dir,
              min_depth, max_depth, use_dorn_normalization, sid_bins,
              blacklist_file):
    """Generates training and validation datasets from
    text files and directories. Sets up datasets with transforms.py.
    *_file - string - a text file containing info for DepthDataset to load the images
    *_dir - string - the folder containing the images to load
    min_depth - the minimum depth for this dataset
    max_depth - the maximum depth for this dataset
    use_dorn_normalization - Whether or not to use the normalization from the original DORN nyuv2 network.
    sid_bins - the number of Spacing Increasing Discretization bins to add.
    blacklist_file - string - a text file listing, on each line, an image_id of an image to exclude
                              from the dataset.

    test_loader - bool - whether or not to test the loader and not set the dataset-wide mean and
                         variance.

    Returns
    -------
    train, val, test - torch.utils.data.Dataset objects containing the relevant splits
    """
    train = NYUDepthv2Dataset(train_file, train_dir, transform=None,
                              file_types=["rgb", "rawdepth"],
                              min_depth=min_depth, max_depth=max_depth)

    train.rgb_mean, train.rgb_var = train.get_mean_and_var()

    # Transform:
    # Size is set to (353, 257) to conform to DORN conventions
    # If use_dorn_normalization is true:
    # Mean is set to np.array([[[103.0626, 115.9029, 123.1516]]]).astype(np.float32) to conform to DORN conventions
    # Var is set to np.ones((1,1,3)) to conform to DORN conventions
    if use_dorn_normalization:
        transform_mean = np.array([[[103.0626, 115.9029, 123.1516]]]).astype(np.float32)
        transform_var = np.ones((1, 1, 3))
    else:
        transform_mean = train.rgb_mean
        transform_var = train.rgb_var
    train_transform = transforms.Compose([
        ResizeAll((353, 257), keys=["rgb", "rawdepth"]),
        RandomHorizontalFlipAll(flip_prob=0.5, keys=["rgb", "rawdepth"]),
        AddDepthMask(min_depth, max_depth, "rawdepth"), # introduces "mask"
        Normalize(transform_mean, transform_var, key="rgb"), # introduces "rgb_orig"
        AddSIDDepth(),
        ToTensorAll(keys=["rgb", "rgb_orig", "rawdepth", "mask"])
        ]
    )

    val_transform = transforms.Compose([
        ResizeAll((353, 257), keys=["rgb", "rawdepth"]),
        AddDepthMask(min_depth, max_depth, "rawdepth"),
        Normalize(transform_mean, transform_var, key="rgb"),
        AddSIDDepth(),
        ToTensorAll(keys=["rgb", "rgb_orig", "rawdepth", "mask"])
        ]
    )

    test_transform = transforms.Compose([
        ResizeAll((353, 257), keys=["rgb", "rawdepth"]),
        AddDepthMask(min_depth, max_depth, "rawdepth"),
        Normalize(transform_mean, transform_var, key="rgb"),
        ToTensorAll(keys=["rgb", "rgb_orig", "rawdepth", "mask"])
        ]
    )
    train.transform = train_transform
    print("Loaded training dataset from {} with size {}.".format(train_file, len(train)))
    val = None
    if val_file is not None:
        val = NYUDepthv2Dataset(val_file, val_dir, transform=val_transform,
                                file_types = ["rgb", "rawdepth"],
                                min_depth=min_depth, max_depth=max_depth)
        val.rgb_mean, val.rgb_var = train.rgb_mean, train.rgb_var
        print("Loaded val dataset from {} with size {}.".format(val_file, len(val)))
    test = None
    if test_file is not None:
        test = NYUDepthv2Dataset(test_file, test_dir, transform=test_transform,
                                 file_types = ["rgb", "rawdepth"],
                                 min_depth=min_depth, max_depth=max_depth)
        test.rgb_mean, test.rgb_var = train.rgb_mean, train.rgb_var
        print("Loaded test dataset from {} with size {}.".format(test_file, len(test)))

    return train, val, test



###########
# Testing #
###########
@nyuv2_nohints_sid_ingredient.automain
def test_load_data(min_depth, max_depth):
    train, val, test = load_data()
    sample = train.get_item_by_id("dining_room_0001a/0001")
    print(sample["rgb"].size())
