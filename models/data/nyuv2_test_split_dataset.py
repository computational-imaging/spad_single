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
    def __init__(self, rootdir, transform=None, dorn_mode=False):
        self.rootdir = rootdir
        self.dorn_mode = dorn_mode # set to True if you're loading data for DORN.
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
        if dorn_mode:
            self.rgb = np.flip(self.rgb, axis=3).copy() # N x H x W x C
            self.rgb_cropped = np.flip(self.rgb_cropped, axis=3).copy()

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
    test = NYUDepthv2TestDataset(root_dir, transform=None, dorn_mode=dorn_mode)

    if dorn_mode:
        print("Using dataset in DORN mode.")
        # Give data entries as DORN expects
        transform_mean = np.array([[[103.0626, 115.9029, 123.1516]]]).astype(np.float32)
        transform_var = np.ones((1, 1, 3))
        test_transform = transforms.Compose([
            AddDepthMask(min_depth, max_depth, "depth_cropped"),
            Save(["rgb_cropped", "mask", "depth_cropped"], "_orig"),
            ResizeAll((353, 257), keys=["rgb_cropped", "depth_cropped"]),
            Normalize(transform_mean, transform_var, key="rgb_cropped"),
            ToTensorAll(keys=["rgb", "rgb_cropped", "rgb_cropped_orig","depth", "depth_cropped",
                              "depth_cropped_orig", "mask", "mask_orig"])
        ])
        # TODO: Determine if BGR or RGB
        # Answer: RGB - need to flip for DORN.
    else:
        print("Using dataset in Wonka mode.")
        # Give data entries as Wonka et. al. expects.
        test_transform = None # Leaves numpy data as-is

    test.transform = test_transform
    return test

###########
# Testing #
###########
@nyuv2_test_split_ingredient.automain
def test_load_data(min_depth, max_depth):
    test = load_data(dorn_mode=False)
    for i in range(1):
        sample = test[300]
        # print(sample["rgb"])
        print([(key, sample[key].shape) for key in sample])
        print(np.min(sample["depth"]))
        print(np.max(sample["depth"]))

        print(sample["rgb"][30, 30, :])
        # cv2 imwrite presumes BGR order.
        cv2.imwrite("flip.png", sample["rgb"].astype('uint8'))
        cv2.imwrite("noflip.png", sample["rgb"][:,:,::-1].astype('uint8'))

    test = load_data(dorn_mode=True)
    for i in range(1):
        sample = test[i]
        # print(sample["rgb"])
        print([(key, sample[key].shape) for key in sample])
        print(torch.min(sample["depth"]))
        print(torch.max(sample["depth"]))

        print(sample["rgb"][:, 30, 30])






