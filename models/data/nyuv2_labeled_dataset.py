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
    crop_file = "crop.npy"

    # True for DORN
    bgr_mode = False

    # True for DORN
    channels_first=False

    min_depth = 0.
    max_depth = 10.

class NYUDepthv2LabeledDataset(Dataset):
    """
    The official NYUv2 Labeled dataset, loaded from numpy files.
    Evaluation crop specified in a separate numpy file.
    """
    def __init__(self, rootdir, rgb_file, depth_file, rawdepth_file, crop_file, transform=None,
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


        self.crop = np.load(os.path.join(rootdir, crop_file))
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
def load_data(root_dir, train_files, test_files, crop_file, min_depth, max_depth, dorn_mode):
    """
    Wonka:
    Input: rgb
    Output size: Same as *_cropped
    Compare to: rawdepth_cropped, mask

    DORN:
    Input: Resized version of bgr_cropped
    Output size: Same as unresized bgr_cropped
    Comapre to: rawdepth_cropped, mask

    :param root_dir:  The root directory from which to load the dataset
    :param use_dorn_normalization: Whether or not to normalize the rgb images according to DORN statistics.
    :return: test: a NYUDepthv2TestDataset object.
    """

    train = NYUDepthv2LabeledDataset(root_dir, train_files["rgb"],
                                     train_files["depth"],
                                     train_files["rawdepth"],
                                     crop_file,
                                     transform=None, bgr_mode=dorn_mode)
    test = NYUDepthv2LabeledDataset(root_dir, test_files["rgb"],
                                    test_files["depth"],
                                    test_files["rawdepth"],
                                    crop_file,
                                    transform=None, bgr_mode=dorn_mode)
    transform_list = [
        AddDepthMask(min_depth, max_depth, "rawdepth_cropped", "mask_cropped"),
        AddDepthMask(min_depth, max_depth, "rawdepth", "mask")
    ]
    if dorn_mode:
        print("Using dataset in DORN mode.")
        transform_mean = np.array([[[103.0626, 115.9029, 123.1516]]]).astype(np.float32)
        transform_var = np.ones((1, 1, 3)).astype(np.float32)
        transform_list += [
            Save(["bgr"], "_orig"),
            ResizeAll((353, 257), keys=["bgr"]),
            Normalize(transform_mean, transform_var, key="bgr"),
            ToTensorAll(keys=["bgr", "bgr_orig", "depth_cropped"],
                        channels_first=dorn_mode)
        ]
    else:
        print("Using dataset in Wonka mode.")
        transform_list.append(ToTensorAll(keys=["rgb", "depth_cropped"], channels_first=dorn_mode))
    train.transform = transforms.Compose(transform_list)
    test.transform = transforms.Compose(transform_list)
    return train, test


###########
# Testing #
###########
@nyuv2_labeled_ingredient.automain
def test_load_data(min_depth, max_depth):
    train, test = load_data(dorn_mode=False)

    sample = test[300]
    # print(sample["rgb"])
    print([(key, sample[key].shape) for key in sample if isinstance(sample[key], torch.Tensor)])
    print(np.min(sample["depth"]))
    print(np.max(sample["depth"]))
    print(sample["rgb"].shape)
    print(sample["rgb"][30, 30, :]) # Channels should be last
    # cv2 imwrite presumes BGR order.
    # cv2.imwrite("flip.png", sample["rgb"].astype('uint8'))
    # cv2.imwrite("noflip.png", sample["rgb"][:,:,::-1].astype('uint8'))

    train, test = load_data(dorn_mode=True)
    sample = test[300]
    # print(sample["rgb"])
    print([(key, sample[key].shape) for key in sample if isinstance(sample[key], torch.Tensor)])
    # print(torch.min(sample["depth"]))
    # print(torch.max(sample["depth"]))
    print(sample["bgr"].shape)
    print(sample["bgr"][:, 30, 30]) # Channels should be first

    # Find entries where even inpainted depth has invalid entries
    for i in range(len(test)):
        depth = test[i]["depth_cropped"]
        less_than_zero = torch.sum((depth < 0).float())
        greater_than_ten = torch.sum((depth > 10.).float())
        if less_than_zero + greater_than_ten > 0:
            print("invalid depth entries for image {}".format(i))
            break






