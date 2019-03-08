import os
from collections import defaultdict
import random
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from torchvision import transforms, utils
from depthnet.transforms import (CropPowerOf2All, DepthProcessing, AddRawDepthMask, AddSIDDepth,
                                 AddDepthHist, ClipMinMax, NormalizeRGB, ToFloat, ToTensor,
                                 RandomCropAll, RandomHorizontalFlipAll, ResizeAll)

from sacred import Ingredient, Experiment

nyuv2_nohints_ingredient = Experiment('data_config')


@nyuv2_nohints_ingredient.config
def cfg():
    root_dir = os.path.join("data", "nyu_depth_v2_scaled16")
    train_file = os.path.join(root_dir, "train.json")
    train_dir = root_dir
    val_file = os.path.join(root_dir, "val.json")
    val_dir = root_dir
    test_file = os.path.join(root_dir, "test.json")
    test_dir = root_dir
    del root_dir

    # Indices of images to exclude from the dataset.
    blacklist_file = os.path.join("blacklist.txt")

    train_keywords = None
    val_keywords = None
    test_keywords = None


    min_depth = 0.
    max_depth = 10.
    # For testing how data loads. Turns off normalization.
    test_loader = False


@data_ingredient.named_config
def raw_hist():
    hist_use_albedo = False
    hist_use_squared_falloff = False


def worker_init(worker_id):
    cudnn.deterministic = True
    random.seed(1 + worker_id)
    np.random.seed(1 + worker_id)
    torch.manual_seed(1 + worker_id)
    torch.cuda.manual_seed(1 + worker_id)


class DepthDataset(Dataset):  # pylint: disable=too-few-public-methods
    """Class for reading and storing image and depth data together.
    """

    def __init__(self, splitfile, data_dir, keywords=None,
                 file_types=["rgb", "depth", "rawdepth", "albedo"],
                 info_file="info.json", blacklist_file="blacklist.txt",
                 augment=None, transform=None):
        """
        Parameters
        ----------
        splitfile - string - json file mapping |global_id| to a dictionary of resource files.
        data_dir - string - the root directory from which the resource files are specified
                            (via relative path)
        keywords - string - a set of 
        file_types - list of string - the keys for the dictionary of resource files provided by each entry in splitfile
        info_file - string - path to json file containing a dictionary with metadata about each image, indexed by image_id
        blacklist_file - list of string - keys in splitfile that should not be used.
        augment - torchvision.transform - preprocessing applied to raw PIL images to augment the dataset.
        transform - torchvision.transform - preprocessing applied to the data before it is output.

        """
        super(DepthDataset, self).__init__()
        self.data_dir = data_dir
        self.augment = augment
        self.transform = transform
        self.file_types = file_types
        self.index = {}
        self.data = []
        self.info = {}
        self.blacklist = []

        info = {}
        if info_file is not None:
            print("Loading info file from {}".format(os.path.join(data_dir, info_file)))
            with open(os.path.join(data_dir, info_file), "r") as f:
                info = json.load(f)

        if blacklist_file is not None:
            print("Loading blacklist from {}".format(os.path.join(data_dir, blacklist_file)))
            with open(os.path.join(data_dir, blacklist_file), "r") as f:
                self.blacklist = [line.strip() for line in f.readlines()]

        with open(splitfile, "r") as f:
            # local_index = 0
            self.index = json.load(f)
        for image_id in self.index:
            # Check blacklist
            if image_id in self.blacklist:
                continue  # Exclude this entry.
            # elif self.check_blank_albedo(image_id):
            #     print("found blank albedo: {}".format(image_id))
            #     continue # Exclude this entry
            if info_file is not None and keywords is not None:
                for word in keywords:
                    # Keyword match obtained
                    if word in info[image_id]["keywords"]:
                        self.data.append(image_id)
                        break
            else:
                self.data.append(image_id)

    def get_global_stats(self, cache="stats_cache.txt", write_cache=True):
        """Calculate mean and variance of each rgb channel.
        Optionally caches the result of this calculation in outfile so
        it doesn't need to be done each time the dataset is loaded.

        Does everything in numpy.
        """
        if cache is not None:
            cache_file = os.path.join(self.data_dir, cache)
            try:
                with open(cache_file, "r") as f:
                    mean = np.array([float(a) for a in f.readline().strip().split(",")])
                    var = np.array([float(a) for a in f.readline().strip().split(",")])
                    print(mean)
                    print(var)
                    stats = {"rgb_mean": mean,
                             "rgb_var": var}
                    return stats
                print("loaded stats cache at {}".format(cache_file))
            except IOError:
                print("failed to load stats cache at {}".format(cache_file))

        print("creating new stats cache (this may take a while...) at {} ".format(cache_file))
        S = np.zeros(3)
        S_sq = np.zeros(3)
        npixels = 0.

        for image_id in self.data:
            rgb_img = self.load_from_file(image_id)["rgb"]
            rgb_img = np.asarray(rgb_img, dtype=np.uint16)

            npixels += rgb_img.shape[0] * rgb_img.shape[1]
            for channel in range(rgb_img.shape[2]):
                S[channel] += np.sum(rgb_img[:, :, channel])
                S_sq[channel] += np.sum((rgb_img[:, :, channel]) ** 2)
        mean = S / npixels
        var = S_sq / npixels - mean ** 2

        if write_cache:
            try:
                cache_file = os.path.join(self.data_dir, cache)
                with open(cache_file, "w") as f:
                    f.write(",".join(str(m) for m in mean) + "\n")
                    f.write(",".join(str(v) for v in var) + "\n")
                print("wrote stats cache to {}".format(cache_file))
            except IOError:
                print("failed to write stats cache to {}".format(cache_file))

        stats = {"rgb_mean": mean,
                 "rgb_var": var}
        return stats

    def check_blank_albedo(self, image_id, albedo_key="albedo"):
        """Check if a given image_id has an albedo image that is all blank.
        Use the albedo_key to index into the index to find the file path
        to the albedo image.
        """
        if albedo_key not in self.index[image_id]:
            print("no albedo file: {}".format(image_id))
            return False  # Entry doesn't have an albedo file associated with it.
        # Load the albedo file
        relpath = self.index[image_id][albedo_key]
        albedo = np.asarray(Image.open(os.path.join(self.data_dir, relpath)))
        return np.all(albedo == 0)

    def load_all_from_files(self, image_id):
        """Given an image id, load the image as a
        PIL.Image from the path given in the index.
        """
        files = {}
        for file_type in self.file_types:
            relpath = self.index[image_id][file_type]
            files[file_type] = Image.open(os.path.join(self.data_dir, relpath))
        return files

    def configure_transform(self,
                            depth_format,
                            min_depth,
                            max_depth,
                            rgb_mean,
                            rgb_var,
                            hist_len=None,
                            hist_use_squared_falloff=None,
                            hist_use_albedo=None,
                            sid_bins=None,
                            **kwargs):
        """
        Configure the data transforms of this dataset using external parameters.
        :param depth_format: From dataset spec - the type of depth data {NYU, SUNRGBD} to load.
        :param min_depth: From dataset spec - the minimum image depth to estimate.
        :param max_depth: From dataset spec - the maximum image depth to estimate.
        :param stats: A dictionary of global statistics to use to normalize or otherwise transform the dataset.
        :param hist_len: From model spec - the number of bins in the histogram. None if no histogram
        :param hist_use_squared_falloff: From dataset spec - whether or not to take the squared
               distance falloff into account when simulating the histogram.
        :param hist_use_albedo: From dataset spec - whether or not to use the albedo image to simulate the
               histogram
        :param sid_bins: From model spec - the number of discrete depth bins in the output.
        :return:
        """
        resize = [CropPowerOf2All(4)]
        # resize = [ResizeAll((257, 353))]
        PIL_transforms = [DepthProcessing("depth", depth_format),
                          DepthProcessing("rawdepth", depth_format),
                          AddRawDepthMask(min_depth, max_depth),
                          ClipMinMax("depth", min_depth, max_depth),  # Note: Need to add depth mask before clipping.
                          ToFloat("rgb"),
                          ToFloat("albedo"),
                          ]
        float_transforms = []
        if hist_len is not None:  # Use depth histogram.
            float_transforms.append(AddDepthHist(use_albedo=hist_use_albedo,
                                                 use_squared_falloff=hist_use_squared_falloff,
                                                 bins=hist_len, range=(min_depth, max_depth), density=True))
        if sid_bins is not None:  # Use spacing increasing discretization loss - requires depth_sid entry
            float_transforms.append(AddSIDDepth(sid_bins=sid_bins, sid_range=(min_depth, max_depth)))
        float_transforms += [NormalizeRGB(rgb_mean, rgb_var),
                             ToTensor(),
                             ]
        # float_transforms += [
        #     SubtractMeanRGB(rgb_mean)
        #
        # ]
        self.transform = transforms.Compose(resize + PIL_transforms + float_transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = {}
        sample["image_id"] = self.data[i]
        sample.update(self.load_all_from_files(self.data[i]))
        if self.augment:
            sample = self.augment(sample)
        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_item_by_id(self, image_id):
        """Different way of getting an item that goes by image_id
        instead of index i
        """
        return self.__getitem__(self.data.index(image_id))


###########
# Caching #
###########
def write_cache(obj, cache_file):
    """Saves the object to the given cache file."""
    with open(cache_file, "w") as f:
        json.dump(obj, f)
    print("wrote cache to {}.".format(cache_file))


def read_cache(cache_file):
    """Reads list from the cache file. If the cache file does not exist,
    returns None.
    """
    try:
        with open(cache_file, "r") as f:
            cache = json.load(f)
            return cache
    except IOError:
        print("failed to load cache at {}".format(cache_file))
        return None


#############
# Load data #
#############
@data_ingredient.capture
# def load_depth_data(train_file, train_dir, train_keywords=None,
#                     val_file=None, val_dir=None, val_keywords=None,
#                     test_file=None, test_dir=None, test_keywords=None,
#                     blacklist_file=None, depth_format="SUNRGBD",
#                     min_depth=None, max_depth=None,
#                     hist_bins=None, hist_range=None,
#                     sid_bins=None, sid_range=None, sid_offset=None,
#                     hist_use_albedo=True, hist_use_squared_falloff=True,
#                     test_loader=False):
def load_depth_data(train_file, train_dir, train_keywords=None,
                    val_file=None, val_dir=None, val_keywords=None,
                    test_file=None, test_dir=None, test_keywords=None,
                    blacklist_file=None, test_loader=False):
    """Generates training and validation datasets from
    text files and directories. Sets up datasets with transforms.py.
    *_file - string - a text file containing info for DepthDataset to load the images
    *_dir - string - the folder containing the images to load
    *_keywords - list of string - a collection of strings used to decide which subset of the
                          data to load.
    blacklist_file - string - a text file listing, on each line, an image_id of an image to exclude
                              from the dataset.

    test_loader - bool - whether or not to test the loader and not set the dataset-wide mean and
                         variance.

    Returns
    -------
    train, val, test - torch.utils.data.Dataset objects containing the relevant splits
    """
    train = DepthDataset(train_file, train_dir, train_keywords, blacklist_file=blacklist_file)
    if test_loader:
        mean = None  # Don't normalize
        var = None
    else:
        mean, var = train.get_global_stats()
    train.rgb_mean, train.rgb_var = mean, var
    augment = [RandomCropAll(300),
               RandomHorizontalFlipAll(0.5),
               ]
    resize = [ResizeAll((208, 176))]

    train.augment = transforms.Compose(augment + resize)

    print("Loaded training dataset from {} with size {}.".format(train_file, len(train)))
    val = None
    if val_file is not None:
        val = DepthDataset(val_file, val_dir, val_keywords, blacklist_file=blacklist_file,
                           augment=transforms.Compose(resize),
                           transform=None)
        val.rgb_mean, val.rgb_var = mean, var

        print("Loaded val dataset from {} with size {}.".format(val_file, len(val)))
    test = None
    if test_file is not None:
        test = DepthDataset(test_file, test_dir, test_keywords, blacklist_file=blacklist_file,
                            augment=None,
                            transform=None)
        test.rgb_mean, test.rgb_var = mean, var
        print("Loaded test dataset from {} with size {}.".format(test_file, len(test)))

    return train, val, test


###########
# Testing #
###########
@data_ingredient.automain
def test_load_data(min_depth, max_depth):
    # train_loader, _, _ = get_depth_loaders()
    # batch = next(iter(train_loader))
    # print(batch["rgb"][0, :, :, :])
    # print(batch["depth"][0, :, :, :])
    # print(batch["mask"][0, :, :, :])
    # print(batch["image_id"])
    # train, _, _ = load_depth_data()
    # batch = train[0]
    # print(batch["rgb"])

    # Save a histogram
    # with open("test_hist.pkl", "w") as f:
    # Save RGB
    # from itertools import permutations
    # for perm in permutations([0, 1, 2]):
    #     utils.save_image(torch.tensor(batch["rgb"][0, torch.LongTensor(perm)], dtype=torch.uint8), 
    #                      "test_rgb_{}.png".format("_".join([str(i) for i in perm])))
    # Save Depth
    # utils.save_image(batch["depth"][0, :, :, :], "test_depth.png", range=(1e-3, 8.))
    # utils.save_image(batch["rawdepth"][0, :, :], "test_rawdepth.png", range=(1e-3, 8.))
    # Save Albedo
    # utils.save_image(batch["albedo"][0, :, :, :], "test_albedo.png")

    train, _, _ = load_depth_data(hist_bins=10, hist_range=(min_depth, max_depth),
                                  sid_bins=40, sid_range=(min_depth, max_depth))

    files = train.dataset.get_item_by_id("dining_room_0001a/0001")
    depth = files["depth"][0, :, :]
    print(depth)
    depth_sid = files["depth_sid"]
    print(depth_sid)
    # print(depth.dim())
    utils.save_image(depth, "0001_depth_test.png", range=(min_depth, max_depth), normalize=True)
    # test_out = utils.make_grid(depth, range=(min_depth, max_depth), normalize=True)
    # print(test_out)
