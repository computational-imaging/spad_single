import os
import json

import numpy as np
from torch.utils.data import Dataset
import cv2

from torchvision import transforms
from models.data.utils.transforms import (ResizeAll, RandomHorizontalFlipAll, Normalize,
                                          AddDepthMask, ToTensorAll)

from sacred import Experiment

nyuv2_nohints_ingredient = Experiment('data_config')


@nyuv2_nohints_ingredient.config
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


class NYUDepthv2Dataset(Dataset):  # pylint: disable=too-few-public-methods
    """Class for reading and storing image and depth data together.
    """

    def __init__(self, splitfile, data_dir, transform, file_types, min_depth, max_depth,
                 blacklist_file="blacklist.txt"):
        """
        :param splitfile: string: json file mapping |global_id| to a dictionary of resource files.
        :param data_dir: string - the root directory from which the resource files are specified
                                  (via relative path)
        :param transform - torchvision.transform - preprocessing applied to the data.
        :param file_types - list of string - the keys for the dictionary of resource files
                            provided by each entry in splitfile
        :param blacklist_file - list of string - keys in splitfile that should not be used.

        """
        super(NYUDepthv2Dataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.file_types = file_types
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.index = {}
        self.data = []
        self.info = {}
        self.blacklist = []

        if blacklist_file is not None:
            print("Loading blacklist from {}".format(os.path.join(data_dir, blacklist_file)))
            with open(os.path.join(data_dir, blacklist_file), "r") as f:
                self.blacklist = [line.strip() for line in f.readlines()]

        with open(splitfile, "r") as f:
            self.index = json.load(f)
        for entry in self.index:
            if entry in self.blacklist:
                continue  # Exclude this entry.
            self.data.append(entry)

        self.rgb_mean, self.rgb_var = np.zeros(3), np.ones(3) # Default initialization
        self.transform = transform

    def get_mean_and_var(self, cache="mean_var.npy", write_cache=True):
        """Calculate mean and variance of each rgb channel.
        Optionally caches the result of this calculation in outfile so
        it doesn't need to be done each time the dataset is loaded.

        Does everything in numpy.
        """
        if cache is not None:
            cache_file = os.path.join(self.data_dir, cache)
            try:
                mean_var = np.load(cache_file)
                mean = mean_var[()]["mean"]
                var = mean_var[()]["var"]
                print("loaded stats cache at {}".format(cache_file))
                print(mean, var)
                return mean, var
            except IOError:
                print("failed to load stats cache at {}".format(cache_file))

        print("creating new stats cache (this may take a while...) at {} ".format(cache_file))
        S = np.zeros(3)
        S_sq = np.zeros(3)
        npixels = 0.

        for entry in self.data:
            rgb_img = self.load_all_images(entry)["rgb"]
            npixels += rgb_img.shape[0] * rgb_img.shape[1]
            # for channel in range(rgb_img.shape[2]):
            S += np.sum(rgb_img, axis=(0, 1))
            S_sq += np.sum(rgb_img ** 2, axis=(0, 1))
        mean = S / npixels
        var = S_sq / npixels - mean ** 2

        if write_cache:
            try:
                output = {"mean": mean, "var": var}
                cache_file = os.path.join(self.data_dir, cache)
                np.save(cache_file, output)
                print("wrote stats cache to {}".format(cache_file))
            except IOError:
                print("failed to write stats cache to {}".format(cache_file))
        return mean, var

    def load_all_images(self, image_id):
        """Given an image id, load the image as a
        numpy array using cv2 from the path given in the index.
        """
        imgs = {}
        for file_type in self.file_types:
            relpath = self.index[image_id][file_type]
            imgs[file_type] = cv2.imread(os.path.join(self.data_dir, relpath),
                                          cv2.IMREAD_UNCHANGED)
            if file_type == "depth" or file_type == "rawdepth":
                if imgs[file_type].dtype == np.uint16:
                    imgs[file_type] = imgs[file_type] * (self.max_depth - self.min_depth)/(2 ** 16 - 1) + self.min_depth
                elif imgs[file_type].dtype == np.uint8:
                    imgs[file_type] = imgs[file_type] * (self.max_depth - self.min_depth)/(2 ** 8 - 1) + self.min_depth
                else:
                    raise TypeError("DepthDataset: Unknown image data type: {}".format(str(imgs[file_type])))
            imgs[file_type] = imgs[file_type].astype(np.float32)
        return imgs


    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = dict()
        sample["entry"] = self.data[i]
        sample.update(self.load_all_images(self.data[i]))
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def get_item_by_id(self, image_id):
        """Different way of getting an item that goes by image_id
        instead of index i
        """
        return self.__getitem__(self.data.index(image_id))


#############
# Load data #
#############
@nyuv2_nohints_ingredient.capture
def load_data(train_file, train_dir,
              val_file, val_dir,
              test_file, test_dir,
              min_depth, max_depth, use_dorn_normalization,
              blacklist_file):
    """Generates training and validation datasets from
    text files and directories. Sets up datasets with transforms.py.
    *_file - string - a text file containing info for DepthDataset to load the images
    *_dir - string - the folder containing the images to load
    min_depth - the minimum depth for this dataset
    max_depth - the maximum depth for this dataset
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
                              min_depth=min_depth, max_depth=max_depth,
                              blacklist_file=blacklist_file)

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
        AddDepthMask(min_depth, max_depth, "rawdepth"), # introduces "mask"
        RandomHorizontalFlipAll(flip_prob=0.5, keys=["rgb", "rawdepth", "mask"]),
        Normalize(transform_mean, transform_var, key="rgb"), # introduces "rgb_orig"
        ToTensorAll(keys=["rgb", "rgb_orig", "rawdepth", "mask"])
        ]
    )

    val_transform = transforms.Compose([
        ResizeAll((353, 257), keys=["rgb", "rawdepth"]),
        AddDepthMask(min_depth, max_depth, "rawdepth"),
        Normalize(transform_mean, transform_var, key="rgb"),
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
@nyuv2_nohints_ingredient.automain
def test_load_data(min_depth, max_depth):
    train, val, test = load_data()
    sample = train.get_item_by_id("dining_room_0001a/0001")
    print(sample["rgb"].size())

