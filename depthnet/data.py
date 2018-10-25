import os
from collections import defaultdict

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class DepthDataset(Dataset): # pylint: disable=too-few-public-methods
    """Class for reading and storing image and depth data together.
    """
    def __init__(self, splitfile, dataDir, transform=None):
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
        self.data_dir = dataDir
        self.transform = transform
        self.data = []
        with open(splitfile, "r") as f:
            for line in f.readlines():
                self.data.append(line.strip().split(","))
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
        depth_file, rgb_file = self.data[idx]
        depth_img = Image.open(os.path.join(self.data_dir, depth_file))
        rgb_img = Image.open(os.path.join(self.data_dir, rgb_file))
        sample = {"depth": depth_img, "rgb": rgb_img}
        if self.transform:
            resize = transforms.Resize((224, 256), Image.NEAREST)
            sample["depth"] = resize(sample["depth"])
            sample["rgb"] = resize(sample["rgb"])
            sample = self.transform(sample)
        return sample

#############
# Load data #
#############

def load_data(train_file, train_dir, val_file, val_dir):
    """Generates training and validation datasets from
    text files and directories. Sets up datasets with transforms."""
    train = DepthDataset(train_file, train_dir)
    mean, var = train.get_global_stats(os.path.join(train_dir, "stats_cache.txt"))
    train.transform = transforms.Compose([ToFloat(),
                                          # RandomCrop((400, 320)),
                                          # CenterCrop((320, 400)),
                                          AddDepthHist(bins=800//3, range=(0, 8)),
                                          NormalizeRGB(mean, var),
                                          ToTensor()
                                         ])
    print("Loaded training dataset from {} with size {}.".format(train_file, len(train)))
    val = None
    if val_file is not None:
        val = DepthDataset(val_file, val_dir,
                           transform=transforms.Compose([ToFloat(),
                                                         # CenterCrop((320, 400)),
                                                         AddDepthHist(bins=800//3, range=(0, 8)),
                                                         NormalizeRGB(mean, var),
                                                         ToTensor(),
                                                        ])
                          )

        print("Loaded val dataset from {} with size {}.".format(val_file, len(val)))
    return train, val

def get_loaders(train_file, train_dir, val_file, val_dir, batch_size):
    """Wrapper for getting the loaders at training time."""
    train, val = load_data(train_file,
                           train_dir,
                           val_file,
                           val_dir)

    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    val_loader = None
    if val is not None:
        val_loader = DataLoader(val,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4,
                                pin_memory=True)
    return train_loader, val_loader


##############
# Transforms #
##############
# for data augmentation
class RandomCrop(): # pylint: disable=too-few-public-methods
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

    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']

        h, w = depth.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        depth = depth[top: top + new_h,
                      left: left + new_w]

        rgb = rgb[top: top + new_h,
                  left: left + new_w]

        return {'depth': depth, 'rgb': rgb}

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

        return {"depth": depth[top:bottom, left:right],
                "rgb": rgb[top:bottom, left:right, :]}

class Crop8(): # pylint: disable=too-few-public-methods
    """Crop to a size where both dimensions are divisible by 8"""
    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']
        new_h, new_w = (depth.shape[0]//8)*8, (depth.shape[1]//8)*8
        return {"depth": depth[:new_h, :new_w],
                "rgb": rgb[:new_h, :new_w, :]}

class CropSmall(): # pylint: disable=too-few-public-methods
    """Make a small patch for testing purposes"""
    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']
        h, w = depth.shape[:2]
        x = 16
        return {"depth": depth[h//2-x:h//2+x, w//2-x:w//2+x],
                "rgb": rgb[h//2-x:h//2+x, w//2-x:w//2+x, :]}

class ToFloat(): # pylint: disable=too-few-public-methods
    """Also parses the depth info for sunrgbd."""
    def __call__(self, sample):
        depth = sample['depth']
        x = np.asarray(depth, dtype=np.uint16)
        y = (x >> 3) | (x << 16-3)
        z = y.astype(np.float32)/1000
        z[z > 8.] = 8. # Clip to maximum depth of 8m.
        return {"depth": z,
                "rgb": np.asarray(sample['rgb']).astype(np.float32)}

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
            output['hist'] = torch.from_numpy(sample['hist']).unsqueeze(-1).unsqueeze(-1)

#         print(output)
        output.update({'depth': torch.from_numpy(depth).unsqueeze(0),
                       'rgb': torch.from_numpy(rgb)})
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
        hist, _ = np.histogram(depth, **self.hist_kwargs)
#         print(hist)
#         print(sample["depth"])
        return {"depth": sample["depth"],
                "rgb": sample["rgb"],
                "hist": hist}

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
