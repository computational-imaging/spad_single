import os
import json

import numpy as np
from torch.utils.data import Dataset
import cv2

from torchvision import transforms
from models.data.utils.transforms import (ResizeAll, RandomHorizontalFlipAll, Normalize,
                                          AddDepthMask, ToTensorAll)

from sacred import Experiment

nyuv2_wonka_nohints_ingredient = Experiment('data_config')

@nyuv2_wonka_nohints_ingredient.config
def cfg():
    data_name = "nyu_depth_v2_wonka"
    root_dir = os.path.join("data", "nyu_depth_v2_wonka")

class NYUDepthv2WonkaDataset(Dataset):
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
        self.rgb_cropped = self.depth[:,
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
            # GT Depth
            # RGB
        }
        if self.transform is not None:
            data = self.transf
