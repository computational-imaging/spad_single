import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

from torchvision import transforms
from models.data.utils.transforms import (ResizeAll, Save, Normalize,
                                          AddDepthMask, ToTensorAll)

from sacred import Experiment

nyuv2_test_split_ingredient = Experiment('data_config')

@nyuv2_test_split_ingredient.config
def cfg():
    data_name = "nyu_depth_v2_test_split"
    root_dir = os.path.join("data", "nyu_depth_v2_wonka")
    dorn_mode = False # Default is Wonka Mode

    min_depth = 0.
    max_depth = 10.

class NYUDepthv2TestDataset(Dataset):
    """
    The split of the nyu_depth_v2 dataset given by
    https://github.com/ialhashim/DenseDepth
    """
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.depth = np.load(os.path.join(rootdir, "eigen_test_depth.npy"))
        self.rgb = np.load(os.path.join(rootdir, "eigen_test_rgb.npy"))
        self.crop = np.load(os.path.join(rootdir, "eigen_test_crop.npy"))
        self.depth_cropped = self.depth[:,
                                        self.crop[0]:self.crop[1]+1,
                                        self.crop[2]:self.crop[3]+1
                                       ]
        self.rgb_cropped = self.rgb[:,
                                      self.crop[0]:self.crop[1]+1,
                                      self.crop[2]:self.crop[3]+1,
                                      :
                                      ]
        self.transform = transform

    def __len__(self):
        return self.depth.shape[0]

    def __getitem__(self, i):
        # Convert to torch tensor
        # Flip channel order
        sample = {
            "depth_cropped": self.depth_cropped[i, :, :],
            "rgb_cropped": self.rgb_cropped[i, :, :, :],
            "depth": self.depth[i, :, :],
            "rgb": self.rgb[i, :, :, :],
            "crop": self.crop,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


@nyuv2_test_split_ingredient.capture
def load_data(root_dir, min_depth, max_depth, dorn_mode):
    """

    :param root_dir:  The root directory from which to load the dataset
    :param use_dorn_normalization: Whether or not to normalize the rgb images according to DORN statistics.
    :return: test: a NYUDepthv2TestDataset object.
    """
    test = NYUDepthv2TestDataset(root_dir, transform=None)

    if dorn_mode:
        # Give data entries as DORN expects
        transform_mean = np.array([[[103.0626, 115.9029, 123.1516]]]).astype(np.float32)
        transform_var = np.ones((1, 1, 3))
        test_transform = transforms.Compose([
            AddDepthMask(min_depth, max_depth, "depth"),
            Save(["rgb", "mask", "depth"], "_orig"),
            ResizeAll((353, 257), keys=["rgb", "depth"]),
            Normalize(transform_mean, transform_var, key="rgb"),
            ToTensorAll(keys=["rgb", "rgb_orig", "depth", "depth_orig", "mask", "mask_orig"])
        ])
        # TODO: Determine if BGR or RGB

    else:
        # Give data entries as Wonka et. al. expects.
        test_transform = None # Leaves numpy data as-is

    test.transform = test_transform
    return test

###########
# Testing #
###########
@nyuv2_test_split_ingredient.automain
def test_load_data(min_depth, max_depth):
    test = load_data(dorn_mode=True)
    for i in range(10):
        sample = test[i]
        # print(sample["rgb"])
        print([(key, sample[key].shape) for key in sample])
        print(torch.min(sample["depth"]))
        print(torch.max(sample["depth"]))





