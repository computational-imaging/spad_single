import os
from collections import defaultdict
import random
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn

from torchvision import transforms

from sacred import Ingredient

data_ingredient = Ingredient('data_config')

@data_ingredient.config
def cfg():
    train_file = os.path.join("data", "sunrgbd_all", "train.txt")
    train_dir = os.path.join("data", "sunrgbd_all")
    val_file = os.path.join("data", "sunrgbd_all", "val.txt")
    val_dir = os.path.join("data", "sunrgbd_all")
    test_file = os.path.join("data", "sunrgbd_all", "test.txt")
    test_dir = os.path.join("data", "sunrgbd_all")
    batch_size = 20             # Number of training examples per iteration
    keywords = {                # Words that should be in the keywords set of dict
        "train": ["SUNRGBD"],
        "val": ["SUNRGBD"],
        "test": ["SUNRGBD"],
    }

def worker_init(worker_id):
    cudnn.deterministic = True
    random.seed(1 + worker_id)
    np.random.seed(1 + worker_id)
    torch.manual_seed(1 + worker_id)
    torch.cuda.manual_seed(1 + worker_id)


class DepthDataset(Dataset): # pylint: disable=too-few-public-methods
    """Class for reading and storing image and depth data together.
    """
    def __init__(self, splitfile, data_dir, keywords, transform=None):
        """
        Parameters
        ----------
        images : list of (string, string)
            list of (depth_map_path, rgb_path) filepaths to depth maps and their rgb images.
        load_depth_map : function
            the function for loading this particular kind of depth_map
        load_rgb : function
            the function for loading this particular kind of image.
        """
        super(DepthDataset, self).__init__()
        self.data_dir = data_dir
        self.transform = transform
        self.data = []
        self.info = {}
        with open(os.path.join(data_dir, "info.json"), "r") as f:
            info = json.load(f)

        with open(splitfile, "r") as f:
            local_index = 0
            for line in f.readlines():
                counter, rawdepth, depth, rgb = line.strip().split(",")
                for word in keywords:
                    # If we specified a keyword that matches a keyword for this entry, include it.
                    if word in info[counter]["keywords"]:
                        self.data.append((rawdepth, depth, rgb))
                        # Add some extra metadata
                        self.info[local_index] = info[counter]
                        self.info[local_index]["global_index"] = int(counter)
                        local_index += 1
                        break

#         print(self.data)

    def get_global_stats(self, cache="stats_cache.txt"):
        """Calculate mean and variance of each rgb channel.
        Optionally caches the result of this calculation in outfile so
        it doesn't need to be done each time the dataset is loaded.

        Does everything in numpy.
        """
        if cache is not None:
            try:
                with open(cache, "r") as f:
                    mean = np.array([float(a) for a in f.readline().strip().split(",")])
                    var = np.array([float(a) for a in f.readline().strip().split(",")])
                    print(mean)
                    print(var)
                    return mean, var
            except IOError:
                print("failed to load cache at {}".format(cache))

        S = np.zeros(3)
        S_sq = np.zeros(3)
        npixels = 0.
        for _, rgb_file in self.data:
            rgb_img = Image.open(os.path.join(self.data_dir, rgb_file))
            rgb_img = np.asarray(rgb_img, dtype=np.uint16)
#             print(rgb_img[0:10, 0:10, :])

            npixels += rgb_img.shape[0]*rgb_img.shape[1]
            for channel in range(rgb_img.shape[2]):
                S[channel] += np.sum(rgb_img[:, :, channel])
                S_sq[channel] += np.sum((rgb_img[:, :, channel])**2)
        mean = S/npixels
        var = S_sq/npixels - mean**2

        # Load full dataset (memory-intensive)
#         full = []
#         for depthFile, rgbFile in self.data:
#             rgb_img = Image.open(os.path.join(self.dataDir, rgbFile))
#             rgb_img = np.asarray(rgb_img, dtype=np.uint16)
#             full.append(rgb_img)

#         a = np.array(full)
#         mean_true = np.mean(a, axis=(0, 1, 2))
#         var_true = np.var(a, axis=(0, 1, 2))
#         print("actual mean and variance: {} {}".format(mean_true, var_true))
#         print(a.shape)
        try:
            with open(cache, "w") as f:
                f.write(",".join(str(m) for m in mean)+"\n")
                f.write(",".join(str(v) for v in var)+"\n")
        except IOError:
            print("failed to write cache to {}".format(cache))
        return mean, var

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rawdepth_file, depth_file, rgb_file = self.data[idx]
        rawdepth_img = Image.open(os.path.join(self.data_dir, rawdepth_file))
        depth_img = Image.open(os.path.join(self.data_dir, depth_file))
        rgb_img = Image.open(os.path.join(self.data_dir, rgb_file))
        sample = {"rawdepth": rawdepth_img, "depth": depth_img, "rgb": rgb_img}
        if self.transform:
            sample = self.transform(sample)
        return sample

#############
# Load data #
#############
def load_depth_data(train_file, train_dir, train_keywords,
                    val_file=None, val_dir=None, val_keywords=None,
                    test_file=None, test_dir=None, test_keywords=None):
    """Generates training and validation datasets from
    text files and directories. Sets up datasets with transforms."""
    train = DepthDataset(train_file, train_dir, train_keywords)
    mean, var = train.get_global_stats(os.path.join(train_dir, "stats_cache.txt"))
    augment = [RandomCropAll(300),
               RandomHorizontalFlipAll(0.5),
              ]
    resize = [ResizeAll((224, 256))]
    PIL_transforms = [SUNRGBDDepthProcessing("depth"),
                      SUNRGBDDepthProcessing("rawdepth"),
                      AddDepthMask(),
                      ToFloat("rgb")
                     ]
    float_transforms = [AddDepthHist(bins=800//3, range=(0, 8)),
                        NormalizeRGB(mean, var),
                        ToTensor(),
                       ]
    train.transform = transforms.Compose(augment + resize + PIL_transforms + float_transforms)

    print("Loaded training dataset from {} with size {}.".format(train_file, len(train)))
    val = None
    if val_file is not None:
        val = DepthDataset(val_file, val_dir, val_keywords,
                           transform=transforms.Compose(resize + PIL_transforms + float_transforms))

        print("Loaded val dataset from {} with size {}.".format(val_file, len(val)))
    test = None
    if test_file is not None:
        test = DepthDataset(test_file, test_dir, test_keywords,
                            transform=transforms.Compose(resize + PIL_transforms + float_transforms))

        print("Loaded test dataset from {} with size {}.".format(test_file, len(test)))
    return train, val, test

@data_ingredient.capture
def get_depth_loaders(train_file, train_dir,
                      val_file, val_dir,
                      test_file, test_dir,
                      batch_size, keywords):
    """Wrapper for getting the loaders at training time."""
    train, val, test = load_depth_data(train_file,
                                       train_dir,
                                       keywords["train"],
                                       val_file,
                                       val_dir,
                                       keywords["val"],
                                       test_file,
                                       test_dir,
                                       keywords["test"])

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
            random.seed(seed)
            sample[key] = self.resize(sample[key])
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
            random.seed(seed)
            sample[key] = self.random_horiz_flip(sample[key])
        return sample

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

class Crop8(): # pylint: disable=too-few-public-methods
    """Crop to a size where both dimensions are divisible by 8"""
    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']
        new_h, new_w = (depth.shape[0]//8)*8, (depth.shape[1]//8)*8
        new_depth_rgb = {"depth": depth[:new_h, :new_w],
                         "rgb": rgb[:new_h, :new_w, :]}
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

class ToFloat(): # pylint: disable=too-few-public-methods
    def __init__(self, key):
        self.key = key
    def __call__(self, sample):
        sample[self.key] = np.asarray(sample[self.key]).astype(np.float32)
        return sample


class SUNRGBDDepthProcessing(): # pylint: disable=too-few-public-methods
    """Performs the necessary transform to convert SUNRGBD
    depth maps to floats."""
    def __init__(self, depthkey):
        self.key = depthkey

    def __call__(self, sample):
        depth = sample[self.key]
        x = np.asarray(depth, dtype=np.uint16)
        y = (x >> 3) | (x << 16-3)
        z = y.astype(np.float32)/1000
        z[z > 8.] = 8. # Clip to maximum depth of 8m.
        sample[self.key] = z
        return sample

class ToTensor(): # pylint: disable=too-few-public-methods
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
#         depth = depth.transpose((2, 0, 1))
        rgb = rgb.transpose((2, 0, 1))
        output = {}
        if 'hist' in sample:
            output['hist'] = torch.from_numpy(sample['hist']).unsqueeze(-1).unsqueeze(-1).float()
        if 'mask' in sample:
            output['mask'] = torch.from_numpy(sample['mask']).unsqueeze(0).float()
#         print(output)
        output.update({'depth': torch.from_numpy(depth).unsqueeze(0).float(),
                       'rgb': torch.from_numpy(rgb).float()})
        return output

class AddDepthHist(): # pylint: disable=too-few-public-methods
    """Takes a depth map and computes a histogram of depths as well"""
    def __init__(self, **kwargs):
        """
        kwargs - passthrough to np.histogram
        """
        self.hist_kwargs = kwargs

    def __call__(self, sample):
        depth = sample["depth"]
        sample["hist"], _ = np.histogram(depth, **self.hist_kwargs)
#         print(hist)
#         print(sample["depth"])
        return sample

class AddDepthMask(): # pylint: disable=too-few-public-methods
    """Creates a mask that is 1 where actual depth values were recorded and 0 where
    the inpainting algorithm failed to inpaint depth.
    """
    def __call__(self, sample):
        closest = (sample["depth"] == np.min(sample["depth"]))
        zero_depth = (sample["rawdepth"] == 0.)

        mask = (zero_depth & closest)
        sample["mask"] = 1 - mask.astype(np.float32) # Logical NOT
        sample["depth"] = sample["depth"]*sample["mask"] # Set all unknown depths to 0.
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
