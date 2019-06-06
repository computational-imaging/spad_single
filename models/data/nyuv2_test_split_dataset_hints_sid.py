import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

from torchvision import transforms
from models.data.utils.transforms import (ResizeAll, Save, Normalize,
                                          AddDepthMask, ToTensorAll)
from models.data.utils.sid_utils import SID
from models.data.utils.spad_utils import SimulateSpadIntensity, spad_ingredient

from sacred import Experiment

nyuv2_test_split_ingredient = Experiment('data_config', ingredients=[spad_ingredient])

@nyuv2_test_split_ingredient.config
def cfg():
    data_name = "nyu_depth_v2_test_split"
    root_dir = os.path.join("data", "nyu_depth_v2_wonka")
    dorn_mode = False # Default is Wonka Mode

    min_depth = 0.
    max_depth = 10.

    # SID stuff (for DORN)
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
        # Unnecessary because wonka is already in BGR order.
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
            "entry": i,
        }
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_item_by_id(self, id):
        return self[int(id)]


@nyuv2_test_split_ingredient.capture
def load_data(root_dir,
              min_depth, max_depth, dorn_mode,
              sid_bins, alpha, beta, offset,
              spad_config):
    """

    :param root_dir:  The root directory from which to load the dataset
    :param use_dorn_normalization: Whether or not to normalize the rgb images according to DORN statistics.
    :return: test: a NYUDepthv2TestDataset object.
    """
    test = NYUDepthv2TestDataset(root_dir, transform=None, dorn_mode=dorn_mode)

    if dorn_mode:
        transform_mean = np.array([[[103.0626, 115.9029, 123.1516]]]).astype(np.float32)
    else:
        transform_mean = np.zeros((1, 1, 3)).astype(np.float32)
    transform_var = np.ones((1, 1, 3)).astype(np.float32)

    transform_list = [
        AddDepthMask(min_depth, max_depth, "depth_cropped"),
        Save(["rgb_cropped", "mask", "depth_cropped"], "_orig"),
    ]
    if dorn_mode:
        transform_list.append(ResizeAll((353, 257), keys=["rgb_cropped", "depth_cropped"]))
    transform_list += [
        Normalize(transform_mean, transform_var, key="rgb_cropped"),
        AddDepthMask(min_depth, max_depth, "depth_cropped"),
        SimulateSpadIntensity("depth_cropped", "rgb_cropped", "mask", "spad", min_depth, max_depth,
                     spad_config["spad_bins"],
                     spad_config["photon_count"],
                     spad_config["dc_count"],
                     spad_config["fwhm_ps"],
                     spad_config["use_intensity"],
                     spad_config["use_squared_falloff"],
                     sid_obj=SID(sid_bins, alpha, beta, offset)),

    ]
    # TODO: Determine if BGR or RGB
    # Answer: RGB - need to flip for DORN.
    if dorn_mode:
        print("Using dataset in DORN mode.")
        transform_list.append(ToTensorAll(keys=["rgb_cropped", "rgb_cropped_orig", "depth_cropped", "depth_cropped_orig",
                                                "mask", "mask_orig", "spad"]))
    else:
        print("Using dataset in Wonka mode.")
        # Only convert the SPAD to pytorch - everything else can stay in numpy.
        transform_list.append(ToTensorAll(keys=["spad"]))
    test.transform = transforms.Compose(transform_list)
    return test

###########
# Testing #
###########
@nyuv2_test_split_ingredient.automain
def test_load_data(root_dir,
              min_depth, max_depth, dorn_mode,
              sid_bins, alpha, beta, offset,
              spad_config):


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
        print(torch.min(sample["depth_cropped"]))
        print(torch.max(sample["depth_cropped"]))

        print(sample["rgb_cropped"][:, 30, 30])






