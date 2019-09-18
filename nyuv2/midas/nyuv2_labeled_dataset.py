import os
import torch
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms
from models.data.data_utils.transforms import (ResizeAll, Save, Normalize,
                                               AddDepthMask, ToTensorAll)

from sacred import Experiment

nyuv2_labeled_ingredient = Experiment('data_config')

@nyuv2_labeled_ingredient.config
def cfg():
    data_name = "nyu_depth_v2_labeled"
    root_dir = os.path.join("data", "nyu_depth_v2_labeled_numpy")
    train_files = {
        "rgb": "train_images.npy",
        "rgb_cropped": "train_images_cropped.npy",
        "depth": "train_depths.npy",
        "depth_cropped": "train_depths_cropped.npy",
        "rawdepth": "train_rawDepths.npy",
        "rawdepth_cropped": "train_rawDepths_cropped.npy"
    }
    test_files = {
        "rgb": "test_images.npy",
        "rgb_cropped": "test_images_cropped",
        "depth": "test_depths.npy",
        "depth_cropped": "test_depths_cropped.npy",
        "rawdepth": "test_rawDepths.npy",
        "rawdepth_cropped": "test_rawDepths_cropped.npy"
    }
    crop = (20, 460,  24, 616)

    min_depth = 0.
    max_depth = 10.


class NYUDepthv2LabeledDataset(Dataset):
    """
    The official NYUv2 Labeled dataset, loaded from numpy files.
    Evaluation crop specified in a separate numpy file.
    """
    def __init__(self, rootdir, rgb_file, depth_file, rawdepth_file, crop, transform=None,
                 bgr_mode=False):
        """

        :param rootdir:
        :param rgb_file:
        :param depth_file:
        :param rawdepth_file:
        :param crop_file:
        :param transform:
        :param bgr_mode:
        :param channels_first: True for tensorflow, False for pytorch
        """
        self.rootdir = rootdir
        self.bgr_mode = bgr_mode # set to True if you're loading data for DORN.
        self.depth = np.load(os.path.join(rootdir, depth_file))        # H x W x N
        self.rawdepth = np.load(os.path.join(rootdir, rawdepth_file))  # H x W x N
        self.rgb = np.load(os.path.join(rootdir, rgb_file))            # H x W x C x N
        self.crop = crop

        self.depth_cropped = self.depth[self.crop[0]:self.crop[1],
                                        self.crop[2]:self.crop[3],
                                        :
                                        ]
        self.rawdepth_cropped = self.rawdepth[self.crop[0]:self.crop[1],
                                              self.crop[2]:self.crop[3],
                                              :
                                              ]
        self.rgb_cropped = self.rgb[self.crop[0]:self.crop[1],
                                    self.crop[2]:self.crop[3],
                                    :,
                                    :
                                    ]

        channel_axis = 2
        if bgr_mode:
            self.bgr = np.flip(self.rgb, axis=channel_axis).copy()
            self.bgr_cropped = np.flip(self.rgb_cropped, axis=channel_axis).copy()
        self.transform = transform

    def __len__(self):
        return self.depth.shape[-1]

    def __getitem__(self, i):
        # Convert to torch tensor
        sample = {
            "depth_cropped": self.depth_cropped[..., i],
            "depth": self.depth[..., i],
            "rawdepth_cropped": self.rawdepth_cropped[..., i],
            "rawdepth": self.rawdepth[..., i],
            "crop": self.crop,
            "entry": str(i)
        }
        if self.bgr_mode:
            sample.update({
                "bgr": self.bgr[..., i],
                "bgr_cropped": self.bgr_cropped[..., i]
            })
        else:
            sample.update({
                "rgb": self.rgb[..., i],
                "rgb_cropped": self.rgb_cropped[..., i]
            })
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_item_by_id(self, entry):
        return self[int(entry)]


@nyuv2_labeled_ingredient.capture
def load_data(channels_first, dataset_type, root_dir, train_files, test_files, crop, min_depth, max_depth):
    """
    DORN:
    Input: Resized version of bgr_cropped
    Output size: Same as unresized bgr_cropped
    Comapre to: rawdepth_cropped, mask

    :param root_dir:  The root directory from which to load the dataset
    :param use_dorn_normalization: Whether or not to normalize the rgb images according to DORN statistics.
    :return: test: a NYUDepthv2TestDataset object.
    """
    if dataset_type == "train":
        dataset = NYUDepthv2LabeledDataset(root_dir, train_files["rgb"],
                                           train_files["depth"],
                                           train_files["rawdepth"],
                                           crop,
                                           transform=None, bgr_mode=False)
    elif dataset_type == "test":
        dataset = NYUDepthv2LabeledDataset(root_dir, test_files["rgb"],
                                           test_files["depth"],
                                           test_files["rawdepth"],
                                           crop,
                                           transform=None, bgr_mode=False)
    transform_list = [
        AddDepthMask(min_depth, max_depth, "rawdepth_cropped", "mask_cropped"),
        AddDepthMask(min_depth, max_depth, "rawdepth", "mask"),
        ToTensorAll(keys=["rgb", "rgb_cropped", "depth_cropped"],
                    channels_first=channels_first)
    ]
    dataset.transform = transforms.Compose(transform_list)
    return dataset


###########
# Testing #
###########
@nyuv2_labeled_ingredient.automain
def test_load_data(min_depth, max_depth):
    from torch.utils.data._utils.collate import default_collate

    dataset = load_data(dataset_type="test", channels_first=True)
    data = default_collate([dataset[0]])
    print(data.keys())
    print(data["rgb"].shape)
    print(data["depth_cropped"])







