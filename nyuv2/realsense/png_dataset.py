import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

from torchvision import transforms
from models.data.data_utils.transforms import (ResizeAll, Save, Normalize,
                                               AddDepthMask, ToTensorAll)

from sacred import Experiment

<<<<<<< HEAD
png_labeled_ingredient = Experiment('data_config')


@png_labeled_ingredient.config
=======
png_ingredient = Experiment('data_config')


@png_ingredient.config
>>>>>>> ac23c73116b7fdd108ca4951a414a00ea9b25df3
def cfg():
    data_name = "realsense_packard"
    rootdir = "data"

    # scene_folder: number_of_images
    scene_dict = {
        "conf_room": 4,
        "couches": 3,
        "kitchen": 3,
        "kitchen2": 4,
        "office": 4,
        "office2": 4,
        "third_floor": 5,
    }

    crop = (20, 460,  24, 616)

    min_depth = 0.
    max_depth = 10.


class PNGDataset(Dataset):
    """
    Dataset of images in a 1-deep directory structure.
    """
    def __init__(self, rootdir, scene_dict, min_depth, max_depth, crop, transform=None, bgr_mode=False):
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
        def process_depth_img(img, min_depth, max_depth):
            if img.dtype == np.uint16:
                img = img * (max_depth - min_depth) / (2 ** 16 - 1) + min_depth
            elif img.dtype == np.uint8:
                img = img * (max_depth - min_depth) / (2 ** 8 - 1) + min_depth
            else:
                raise TypeError("PNGDataset: Unknown image data type: {}".format(img.dtype))
            return img

        self.rootdir = rootdir
        self.scene_dict = scene_dict
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.crop = crop
        self.transform = transform
        self.bgr_mode = bgr_mode # set to True if you're loading data for DORN.

        # Load Images
        depth = []
        rawDepth = []
        imgs = []
<<<<<<< HEAD
=======
        self.i_to_entry = []
        self.entry_to_i = {}
>>>>>>> ac23c73116b7fdd108ca4951a414a00ea9b25df3
        for scene, n_img in scene_dict.items():
            for i in range(n_img):
                # print("loading {}[{}]".format(scene, i))
                depthFile = os.path.join(rootdir, scene, "{}_depth.png".format(i))
                depth.append(process_depth_img(cv2.imread(depthFile, cv2.IMREAD_UNCHANGED),
                                               self.min_depth,
                                               self.max_depth))
                rawDepthFile = os.path.join(rootdir, scene, "{}_rawDepth.png".format(i))
                rawDepth.append(process_depth_img(cv2.imread(rawDepthFile, cv2.IMREAD_UNCHANGED),
                                                  self.min_depth,
                                                  self.max_depth))
                imgFile = os.path.join(rootdir, scene, "{}_rgb.png".format(i))
                img = cv2.imread(imgFile, cv2.IMREAD_UNCHANGED)
                if not bgr_mode:
                    # Flip color channel
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
<<<<<<< HEAD
=======
                self.i_to_entry.append("{}_{}".format(scene, i))
                self.entry_to_i["{}_{}".format(scene, i)] = len(imgs)
>>>>>>> ac23c73116b7fdd108ca4951a414a00ea9b25df3
                imgs.append(img)
        self.depth = np.array(depth)
        self.rawDepth = np.array(rawDepth)
        self.imgs = np.array(imgs)

        self.depth_cropped = self.depth[:,
                                        self.crop[0]:self.crop[1],
                                        self.crop[2]:self.crop[3]
                                        ]
        self.rawDepth_cropped = self.rawDepth[:,
                                              self.crop[0]:self.crop[1],
                                              self.crop[2]:self.crop[3]
                                              ]
        self.imgs_cropped = self.imgs[:,
                                      self.crop[0]:self.crop[1],
                                      self.crop[2]:self.crop[3],
                                      :,
                                      ]

    def __len__(self):
        return self.depth.shape[0]

    def __getitem__(self, i):
        # Convert to torch tensor
        sample = {
            "depth_cropped": self.depth_cropped[i, ...],
            "depth": self.depth[i, ...],
            "rawDepth_cropped": self.rawDepth_cropped[i, ...],
            "rawDepth": self.rawDepth[i, ...],
            "crop": self.crop,
<<<<<<< HEAD
            "entry": str(i)
=======
            "entry": self.i_to_entry[i]
>>>>>>> ac23c73116b7fdd108ca4951a414a00ea9b25df3
        }
        if self.bgr_mode:
            sample.update({
                "bgr": self.imgs[i, ...],
                "bgr_cropped": self.imgs_cropped[i, ...]
            })
        else:
            sample.update({
                "rgb": self.imgs[i, ...],
                "rgb_cropped": self.imgs_cropped[i, ...]
            })
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_item_by_id(self, entry):
<<<<<<< HEAD
        return self[int(entry)]


@png_labeled_ingredient.capture
=======
        return self[self.entry_to_i[entry]]


@png_ingredient.capture
>>>>>>> ac23c73116b7fdd108ca4951a414a00ea9b25df3
def load_data(channels_first, rootdir, scene_dict, min_depth, max_depth, crop):
    """
    Load the dataset from files and initialize transforms accordingly.
    :return: The dataset.
    """
    dataset = PNGDataset(rootdir, scene_dict, min_depth, max_depth, crop)
    transform_list = [
        AddDepthMask(min_depth, max_depth, "rawDepth_cropped", "mask_cropped"),
        AddDepthMask(min_depth, max_depth, "rawDepth", "mask"),
        ToTensorAll(keys=["rgb", "rgb_cropped", "depth_cropped"],
                    channels_first=channels_first)
    ]
    dataset.transform = transforms.Compose(transform_list)
    return dataset


###########
# Testing #
###########
<<<<<<< HEAD
@png_labeled_ingredient.automain
=======
@png_ingredient.automain
>>>>>>> ac23c73116b7fdd108ca4951a414a00ea9b25df3
def test_load_data(rootdir, scene_dict, min_depth, max_depth, crop):
    # from torch.utils.data._utils.collate import default_collate
    #
    # dataset = load_data(dataset_type="test")
    # data = default_collate([dataset[0]])
    # print(data.keys())
    # print(data["rgb"].shape)
    # print(data["depth_cropped"])

<<<<<<< HEAD
    dataset = load_data(rootdir, scene_dict, min_depth, max_depth, crop)
    # print(dataset.depth.shape)
    # print(dataset.rawDepth.shape)
    # print(dataset.rgb.shape)
    print(dataset[0]["rgb"].shape)
    print(dataset[1]["depth"])
=======
    dataset = load_data(channels_first=False)
    # print(dataset.depth.shape)
    # print(dataset.rawDepth.shape)
    # print(dataset.rgb.shape)
    print(dataset[2])
    print(dataset[0]["rgb"].shape)
    print(dataset.get_item_by_id("couches[2]")["depth"])
>>>>>>> ac23c73116b7fdd108ca4951a414a00ea9b25df3





