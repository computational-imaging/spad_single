import torch
import numpy as np
import random
from PIL import Image
from torchvision import transforms, utils


##############
# Transforms #
##############
# Resizing:
class ResizeAll():
    def __init__(self, output_size):
        """output_size is a tuple (width, height)"""
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
        if "rgb_orig" in sample:
            rgb_orig = sample["rgb_orig"]
            rgb_orig = rgb_orig.transpose(2, 0, 1)
            sample["rgb_orig"] = torch.from_numpy(rgb_orig).float()
        # depth = depth.transpose((1, 0))
        # output = {}
        if 'hist' in sample:
            sample['hist'] = torch.from_numpy(sample['hist']).unsqueeze(-1).unsqueeze(-1).float()
        if 'mask' in sample:
            sample['mask'] = torch.from_numpy(sample['mask']).unsqueeze(0).float()
            sample['eps'] = torch.from_numpy(sample['eps']).unsqueeze(-1).unsqueeze(-1).float()
        if 'depth_sid' in sample:
            sample["depth_sid"] = torch.from_numpy(sample['depth_sid'].transpose(2, 0, 1)).float()
            sample["depth_sid_index"] = torch.from_numpy(sample["depth_sid_index"]).unsqueeze(0).long()
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

        Index array gives the per-pixel bin index of the depth value.
        0's and 1's array has a vector of length |sid_bins| for each pixel that is
        1.0 up to (but not including) the index of the depth value, and 0.0 for the rest.
        Example:
             If depth_sid_index assigns some pixel to be bin 4 (out of 7 bins), then the
             vector for the same pixel in depth_sid is
              0 1 2 3 4 5 6
             [1 1 1 1 0 0 0]
             Note: The most 1's possible is n-1, where n is the number of bins:
             [1 1 1 1 1 1 0]
             The fewest is 0.
        """
        depth = sample["depth"]
        sample["depth_sid_index"] = self.get_depth_sid(depth)
        K = np.zeros(depth.shape + (self.sid_bins,))
        for i in range(self.sid_bins):
            K[:, :, i] = K[:, :, i] + i * np.ones(depth.shape)
        sample["depth_sid"] = (K < sample["depth_sid_index"][:, :, np.newaxis]).astype(np.int32)
        return sample

    def get_depth_sid(self, depth):
        """Given a depth image as a numpy array, computes the per-pixel
        bin index for a SID with range |self.sid_range| = (min_depth, max_depth)
        and a |self.sid_bins| number of bins.
        """
        start = 1.0
        end = self.sid_range[1] + self.offset
        depth_sid = self.sid_bins * np.log(depth + self.offset) / np.log(end)
        depth_sid = depth_sid.astype(np.int32) # Performs rounding
        depth_sid[depth_sid >= self.sid_bins] = self.sid_bins - 1 # Clip so as not to be out of range of the max index.
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
        sample["rgb_orig"] = np.copy(sample["rgb"])
        sample["rgb"] = (sample["rgb"] - self.mean)/np.sqrt(self.var)
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


class CropSmall(): # pylint: disable=too-few-public-methods
    """Make a small patch for testing purposes"""
    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']
        h, w = depth.shape[:2]
        x = 16
        return {"depth": depth[h//2-x:h//2+x, w//2-x:w//2+x],
                "rgb": rgb[h//2-x:h//2+x, w//2-x:w//2+x, :]}
