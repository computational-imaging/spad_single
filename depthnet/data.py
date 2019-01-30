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

from sacred import Ingredient, Experiment

data_ingredient = Experiment('data_config')

@data_ingredient.config
def cfg():
    train_file = os.path.join("data", "sunrgbd_all", "train.txt")
    train_dir = os.path.join("data", "sunrgbd_all")
    val_file = os.path.join("data", "sunrgbd_all", "val.txt")
    val_dir = os.path.join("data", "sunrgbd_all")
    test_file = os.path.join("data", "sunrgbd_all", "test.txt")
    test_dir = os.path.join("data", "sunrgbd_all")

    # Indices of images to exclude from the dataset.
    blacklist_file = os.path.join("data", "sunrgbd_all", "blacklist.txt")

    batch_size = 20              # Number of training examples per iteration
    train_keywords = ["SUNRGBD"] # Keywords that control which data points are used.
    val_keywords = ["SUNRGBD"]   # A value of None means no restrictions on data points.
    test_keywords = ["SUNRGBD"]
    depth_format = "SUNRGBD"
    hist_use_albedo = True
    hist_use_squared_falloff = True

    min_depth = 1e-3
    max_depth = 8.

    sid_bins = 80               # Number of bins to use for the spacing-increasing discretization
    # For testing how data loads. Turns off normalization.
    test_loader = False

@data_ingredient.named_config
def nyu_depth_v2():
    root_dir = os.path.join("data", "nyu_depth_v2_processed")
    train_file = os.path.join(root_dir, "train.json")
    train_dir = root_dir
    val_file = os.path.join(root_dir, "test.json")
    val_dir = root_dir
    test_file = None
    test_dir = None
    del root_dir



    # Indices of images to exclude from the dataset.
    blacklist_file = os.path.join("blacklist.txt")

    batch_size = 20             # Number of training examples per iteration
    train_keywords = None
    val_keywords = None
    test_keywords = None
    depth_format = "NYU"
    # hist_use_albedo = True
    # hist_use_squared_falloff = True

    min_depth = 1e-3
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


class DepthDataset(Dataset): # pylint: disable=too-few-public-methods
    """Class for reading and storing image and depth data together.
    """
    def __init__(self, splitfile, data_dir, keywords=None,
                 file_types=["rgb", "depth", "rawdepth", "albedo"],
                 info_file="info.json", blacklist_file="blacklist.txt", 
                 rgb_mean=None, rgb_var=None, transform=None):
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
        transform - torchvision.transform - preprocessing applied to the data before it is output.

        """
        super(DepthDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.file_types = file_types
        self.index = {}
        self.data = []
        self.info = {}
        self.blacklist = []
        if rgb_mean is None:
            self.rgb_mean = (0, 0, 0)
        else:
            self.rgb_mean = rgb_mean
        if rgb_var is None:
            self.rgb_var = (1, 1, 1)
        else:
            self.rgb_var = rgb_var
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
                continue # Exclude this entry.
            # elif self.check_blank_albedo(image_id):
            #     print("found blank albedo: {}".format(image_id))
            #     continue # Exclude this entry
            if info_file is not None and keywords is not None:
                for word in keywords:
                    # Keyword match obtained
                    if word in info[image_id]["keywords"]:
                        self.data.append(image_id)
                        # Add some extra metadata
                        self.info[local_index] = info[image_id]
                        self.info[local_index]["image_id"] = image_id
                        local_index += 1
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
                    return mean, var
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

            npixels += rgb_img.shape[0]*rgb_img.shape[1]
            for channel in range(rgb_img.shape[2]):
                S[channel] += np.sum(rgb_img[:, :, channel])
                S_sq[channel] += np.sum((rgb_img[:, :, channel])**2)
        mean = S/npixels
        var = S_sq/npixels - mean**2

        if write_cache:
            try:
                cache_file = os.path.join(self.data_dir, cache)
                with open(cache_file, "w") as f:
                    f.write(",".join(str(m) for m in mean)+"\n")
                    f.write(",".join(str(v) for v in var)+"\n")
                print("wrote stats cache to {}".format(cache_file))
            except IOError:
                print("failed to write stats cache to {}".format(cache_file))
        self.rgb_mean = mean
        self.rgb_var = var
        return mean, var

    def check_blank_albedo(self, image_id, albedo_key="albedo"):
        """Check if a given image_id has an albedo image that is all blank.
        Use the albedo_key to index into the index to find the file path
        to the albedo image.
        """
        if albedo_key not in self.index[image_id]:
            print("no albedo file: {}".format(image_id))
            return False # Entry doesn't have an albedo file associated with it.
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        sample = {}
        sample["image_id"] = self.data[i]
        sample.update(self.load_all_from_files(self.data[i]))
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
def load_depth_data(train_file, train_dir, train_keywords=None,
                    val_file=None, val_dir=None, val_keywords=None,
                    test_file=None, test_dir=None, test_keywords=None,
                    blacklist_file=None, depth_format="SUNRGBD",
                    min_depth=None, max_depth=None,
                    hist_bins=None, hist_range=None,
                    sid_bins=None, sid_range=None, sid_offset=None,
                    hist_use_albedo=True, hist_use_squared_falloff=True,
                    test_loader=False):
    """Generates training and validation datasets from
    text files and directories. Sets up datasets with transforms.
    *_file - string - a text file containing info for DepthDataset to load the images
    *_dir - string - the folder containing the images to load
    *_keywords - list of string - a collection of strings used to decide which subset of the
                          data to load.
    blacklist_file - string - a text file listing, on each line, an image_id of an image to exclude
                              from the dataset.
    depth_format - string - {SUNRGBD, NYU} the format for reading the depth from the file.
    min_depth - float - the minimum depth to clip the images to
    max_depth - float - the maximum depth to clip the images to
    hist_bins - int - the number of bins in the histogram
    hist_range - (float, float) the range of values for the histogram
    hist_use_albedo - bool - whether or not to take albedo into account
    hist_use_squared_albedo - bool - whether or not to take the squared distance falloff into
                                     account
    test_loader - bool - whether or not to test the loader and not set the dataset-wide mean and
                         variance.
    """
    train = DepthDataset(train_file, train_dir, train_keywords, blacklist_file=blacklist_file)
    if test_loader:
        mean = None # Don't normalize
        var = None
    else:
        mean, var = train.get_global_stats()
    train.rgb_mean, train.rgb_var = mean, var
    augment = [RandomCropAll(300),
               RandomHorizontalFlipAll(0.5),
              ]
    # resize = [ResizeAll((224, 256))]
    resize = [CropPowerOf2All(4)]
    # resize = [] # No resizing operation
    PIL_transforms = [DepthProcessing("depth", depth_format),
                      DepthProcessing("rawdepth", depth_format),
                      AddDepthMask(),
                      ClipMinMax("depth", min_depth, max_depth), # Note: Need to add depth mask before clipping.
                      ToFloat("rgb"),
                      ToFloat("albedo"),
                     ]
    float_transforms = [AddDepthHist(use_albedo=hist_use_albedo,
                                     use_squared_falloff=hist_use_squared_falloff,
                                     bins=hist_bins, range=hist_range, density=True),
                        AddSIDDepth(sid_bins=sid_bins, sid_range=hist_range),
                        ToTensor(),
                       ]
    train.transform = transforms.Compose(augment + resize + PIL_transforms + float_transforms)

    print("Loaded training dataset from {} with size {}.".format(train_file, len(train)))
    val = None
    if val_file is not None:
        val = DepthDataset(val_file, val_dir, val_keywords, blacklist_file=blacklist_file,
                           rgb_mean=mean, rgb_var=var,
                           transform=transforms.Compose(resize + PIL_transforms + float_transforms))
        val.rgb_mean, val.rgb_var = mean, var

        print("Loaded val dataset from {} with size {}.".format(val_file, len(val)))
    test = None
    if test_file is not None:
        test = DepthDataset(test_file, test_dir, test_keywords, blacklist_file=blacklist_file,
                            rgb_mean=mean, rgb_var=var,
                            transform=transforms.Compose(resize + PIL_transforms + float_transforms))
        test.rgb_mean, test.rgb_var = mean, var
        print("Loaded test dataset from {} with size {}.".format(test_file, len(test)))
    # Incorporate global stats

    return train, val, test

@data_ingredient.capture
def get_depth_loaders(batch_size, **data_kwargs):
    """Wrapper for getting the loaders at training time."""
    train, val, test = load_depth_data(**data_kwargs)
    print(data_kwargs)
    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              worker_init_fn=worker_init
                             )
    val_loader = None
    if val is not None:
        val_loader = DataLoader(val,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=1,
                                pin_memory=True,
                                worker_init_fn=worker_init
                               )
    test_loader = None
    if test is not None:
        test_loader = DataLoader(test,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=1,
                                 pin_memory=True,
                                 worker_init_fn=worker_init
                                )
    return train_loader, val_loader, test_loader

##############
# Transforms #
##############
# Resizing:
class ResizeAll():
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.resize = transforms.Resize((224, 256), Image.NEAREST)

    def __call__(self, sample):
        seed = np.random.randint(2**32-1)
        for key in sample:
            if isinstance(sample[key], Image.Image):
                random.seed(seed)
                sample[key] = self.resize(sample[key])
        return sample

class CropPowerOf2All(): # pylint: disable=too-few-public-methods
    """Crop to a size where both dimensions are divisible by the given power of 2
    Note that for an Image.Image, the size attribute is given as (width, height) as is standard
    for images and displays (e.g. 640 x 480), but which is NOT standard for most arrays, which
    list the vertical direction first.
    """
    def __init__(self, power, rgb_key="rgb"):
        self.pow_of_2 = 2**power
        self.rgb_key = rgb_key

    def __call__(self, sample):
        rgb = sample[self.rgb_key]
        new_h, new_w = (rgb.size[1]//self.pow_of_2)*self.pow_of_2, \
                       (rgb.size[0]//self.pow_of_2)*self.pow_of_2
        crop = transforms.CenterCrop((new_h, new_w))
        for key in sample:
            if isinstance(sample[key], Image.Image):
                sample[key] = crop(sample[key])
        return sample

# for data augmentation
class RandomCropAll(): # pylint: disable=too-few-public-methods
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.random_crop = transforms.RandomCrop(self.output_size)

    def __call__(self, sample):
        seed = np.random.randint(2**32-1)
        for key in sample:
            if isinstance(sample[key], Image.Image):
                random.seed(seed)
                sample[key] = self.random_crop(sample[key])
        return sample

class RandomHorizontalFlipAll(): # pylint: disable=too-few-public-methods
    """Flip the image horizontally with probability flip_prob.
    """
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob
        self.random_horiz_flip = transforms.RandomHorizontalFlip(self.flip_prob)

    def __call__(self, sample):
        seed = np.random.randint(2**32-1)
        for key in sample:
            if isinstance(sample[key], Image.Image):
                random.seed(seed)
                sample[key] = self.random_horiz_flip(sample[key])
        return sample



class ToFloat(): # pylint: disable=too-few-public-methods
    def __init__(self, key):
        self.key = key
    def __call__(self, sample):
        sample[self.key] = np.asarray(sample[self.key]).astype(np.float32)
        return sample


class DepthProcessing(): # pylint: disable=too-few-public-methods
    """Performs the necessary transform to convert
    depth maps to floats."""
    def __init__(self, depthkey, depth_format="SUNRGBD"):
        self.format = depth_format
        self.key = depthkey

    def __call__(self, sample):
        depth = sample[self.key]
        if self.format == "SUNRGBD":
            x = np.asarray(depth, dtype=np.uint16)
            y = (x >> 3) | (x << 16-3)
            z = y.astype(np.float32)/1000
            z[z > 8.] = 8. # Clip to maximum depth of 8m.
            sample[self.key] = z
        else:  # Just read and divide by 1000.
            depth = np.asarray(depth, dtype=np.float32)
            x = depth/1000
            sample[self.key] = x
        return sample

class ToTensor(): # pylint: disable=too-few-public-methods
    """Convert ndarrays in sample to Tensors.
    Outputs should have 3 dimensions i.e. len(sample[key].size()) == 3
    for key in {'hist', 'mask', 'depth', 'rgb', 'eps'}

    If using a DataLoader, the DataLoader will prepend the batch dimension.
    """

    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
#         depth = depth.transpose((2, 0, 1))
        rgb = rgb.transpose((2, 0, 1))
        # depth = depth.transpose((1, 0))
        # output = {}
        if 'hist' in sample:
            sample['hist'] = torch.from_numpy(sample['hist']).unsqueeze(-1).unsqueeze(-1).float()
        if 'mask' in sample:
            sample['mask'] = torch.from_numpy(sample['mask']).unsqueeze(0).float()
            sample['eps'] = torch.from_numpy(sample['eps']).unsqueeze(-1).unsqueeze(-1).float()
        if 'depth_sid' in sample:
            sample["depth_sid"] = torch.from_numpy(sample['depth_sid']).transpose(2, 0, 1).float()
            sample["depth_sid_index"] = torch.from_numpy(sample["depth_sid_index"]).unsqueeze(0).float()
#         print(output)
        sample.update({'depth': torch.from_numpy(depth).unsqueeze(0).float(),
                       'rgb': torch.from_numpy(rgb).float()})
        return sample

class AddDepthHist(): # pylint: disable=too-few-public-methods
    """Takes a depth map and computes a histogram of depths as well"""
    def __init__(self, use_albedo=True, use_squared_falloff=True, **kwargs):
        """
        kwargs - passthrough to np.histogram
        """
        self.use_albedo = use_albedo
        self.use_squared_falloff = use_squared_falloff
        self.hist_kwargs = kwargs

    def __call__(self, sample):
        depth = sample["depth"]
        if "mask" in sample:
            mask = sample["mask"]
            depth = depth[mask > 0]
        weights = np.ones(depth.shape)
        if self.use_albedo:
            weights = weights * np.mean(sample["albedo"]) # Attenuate by the average albedo TODO
        if self.use_squared_falloff:
            weights[depth == 0] = 0.
            weights[depth != 0] = weights[depth != 0] / (depth[depth != 0]**2)
        if not self.use_albedo and not self.use_squared_falloff:
            sample["hist"], _ = np.histogram(depth, **self.hist_kwargs)
        else:
            sample["hist"], _ = np.histogram(depth, weights=weights, **self.hist_kwargs)
        return sample

class AddDepthMask(): # pylint: disable=too-few-public-methods
    """Creates a mask that is 1 where actual depth values were recorded and 0 where
    the inpainting algorithm failed to inpaint depth.

    eps - small positive number to assign to places with missing depth.
    """
    def __call__(self, sample, eps=1e-6):
        closest = (sample["depth"] == np.min(sample["depth"]))
        zero_depth = (sample["rawdepth"] == 0.)
        mask = (zero_depth & closest)
        # print(sample["rawdepth"])
        sample["mask"] = 1. - mask.astype(np.float32) # Logical NOT
        # Set all unknown depths to be a small positive number.
        sample["depth"] = sample["depth"]*sample["mask"] + (1 - sample["mask"])*eps
        sample["eps"] = np.array([eps])
        return sample

class AddSIDDepth():
    """Creates a copy of the depth image where the depth value has been replaced
    by the SID-discretized index

    Discretizes depth into |sid_bins| number of bins, where the edges of the bins are
    given by

    t_i = exp(log(start) + i/K*log(end/start))

    for i in {0,...,K}.

    According to the DORN paper, we add an offset |sid_offset| such that
    start = sid_range[0] + sid_offset = 1.0
    end = sid_range[1] + sid_offset

    Works in numpy.
    """
    def __init__(self, sid_bins, sid_range):
        self.sid_bins = sid_bins
        self.sid_range = sid_range
        self.offset = 1.0 - sid_range[0]

    def __call__(self, sample):
        """Computes an array with indices, and also an array with 
        0's and 1's that makes computing the ordinal regression loss easier later.
        """
        depth = sample["depth"]
        sample["depth_sid_index"] = self.get_depth_sid(depth)
        K = np.zeros(depth.shape + (self.sid_bins,))
        for i in range(self.sid_bins):
            K[:, :, i] = K[:, :, i] + i * np.ones(depth.shape)
        sample["depth_sid"] = (K < sample["depth_sid_index"])
        return sample

    def get_depth_sid(self, depth):
        """Given a depth image as a numpy array, computes the per-pixel
        bin index for a SID with range |self.sid_range| = (min_depth, max_depth)
        and a |self.sid_bins| number of bins.
        """
        start = 1.0
        end = self.sid_range[1] + self.offset
        depth_sid = self.sid_bins * np.log(depth / start) / \
                                     np.log(end / start)
        depth_sid = depth_sid.astype(np.int32)
        return depth_sid


class ClipMinMax(): # pylint: disable=too-few-public-methods
    def __init__(self, key, min_val, max_val):
        """Setting min_val or max_val equal to None removes clipping
        for the min or the max, respectively.
        """
        self.key = key
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, sample):
        x = sample[self.key]
        if self.min_val is not None:
            x[x < self.min_val] = self.min_val
        if self.max_val is not None:
            x[x > self.max_val] = self.max_val
        sample[self.key] = x
        return sample


class NormalizeRGB(object): # pylint: disable=too-few-public-methods
    """Subtract the mean and divide by the variance of the dataset."""
    def __init__(self, mean, var):
        """
        mean - np.array of size 3 - the means of the three color channels over the train set
        var - np.array of size 3 - the variances of the three color channels over the train set
        """
        self.mean = mean
        self.var = var
    def __call__(self, sample):
        sample["rgb"] -= self.mean
        sample["rgb"] /= np.sqrt(self.var)
#         print(sample["rgb"][0:10, 0:10, 0])
        return sample

### Old ###

class CenterCrop(): # pylint: disable=too-few-public-methods
    """Center crop the image

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']
        h, w = depth.shape
        new_h, new_w = self.output_size

        top = h//2 - new_h//2
        bottom = h//2 + new_h//2 + (1 if new_h % 2 else 0)
        left = w//2 - new_w//2
        right = w//2 + new_w//2 + (1 if new_w % 2 else 0)

        new_depth_rgb = {"depth": depth[top:bottom, left:right],
                         "rgb": rgb[top:bottom, left:right, :]}
        sample.update(new_depth_rgb)
        return sample



class CropSmall(): # pylint: disable=too-few-public-methods
    """Make a small patch for testing purposes"""
    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']
        h, w = depth.shape[:2]
        x = 16
        return {"depth": depth[h//2-x:h//2+x, w//2-x:w//2+x],
                "rgb": rgb[h//2-x:h//2+x, w//2-x:w//2+x, :]}

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

    # train, _, _ = load_depth_data(hist_bins=10, hist_range=(minxdepth, max_depth),
    #                               sid_bins=40, sid_range=(min_depth, max_depth))
    train, _, _ = get_depth_loaders(hist_bins=10, hist_range=(min_depth, max_depth),
                                    sid_bins=40, sid_range=(min_depth, max_depth))
    files = train.dataset.get_item_by_id("dining_room_0001a/0001")
    depth = files["depth"][0,:,:]
    print(depth)
    depth_sid = files["depth_sid"]
    print(depth_sid)
    # print(depth.dim())
    utils.save_image(depth, "0001_depth_test.png", range=(min_depth, max_depth), normalize=True)
    # test_out = utils.make_grid(depth, range=(min_depth, max_depth), normalize=True)
    # print(test_out)

