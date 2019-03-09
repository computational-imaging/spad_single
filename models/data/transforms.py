import cv2
import torch
import numpy as np
import random
from torchvision import transforms, utils
from .sid_utils import SID


##############
# Transforms #
##############
# Resizing:
class ResizeAll():
    def __init__(self, output_size, keys):
        """output_size is a tuple (width, height)
        keys is a list of keys into |sample| for images to resize.
        """
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.keys = keys
    def __call__(self, sample):
        for key in self.keys:
            sample[key] = cv2.resize(sample[key], self.output_size, cv2.INTER_LINEAR)
        return sample


# class CropPowerOf2All(): # pylint: disable=too-few-public-methods
#     """Crop to a size where both dimensions are divisible by the given power of 2
#     Note that for an Image.Image, the size attribute is given as (width, height) as is standard
#     for images and displays (e.g. 640 x 480), but which is NOT standard for most arrays, which
#     list the vertical direction first.
#     """
#     def __init__(self, power, rgb_key="rgb"):
#         self.pow_of_2 = 2**power
#         self.rgb_key = rgb_key
#
#     def __call__(self, sample):
#         rgb = sample[self.rgb_key]
#         new_h, new_w = (rgb.size[1]//self.pow_of_2)*self.pow_of_2, \
#                        (rgb.size[0]//self.pow_of_2)*self.pow_of_2
#         crop = transforms.CenterCrop((new_h, new_w))
#         for key in sample:
#             if isinstance(sample[key], Image.Image):
#                 sample[key] = crop(sample[key])
#         return sample


# for data augmentation
# class RandomCropAll(): # pylint: disable=too-few-public-methods
#     """Crop randomly the image in a sample.
#
#     Args:
#         output_size (tuple or int): Desired output size. If int, square crop
#             is made.
#     """
#     def __init__(self, output_size):
#         assert isinstance(output_size, (int, tuple))
#         if isinstance(output_size, int):
#             self.output_size = (output_size, output_size)
#         else:
#             assert len(output_size) == 2
#             self.output_size = output_size
#         self.random_crop = transforms.RandomCrop(self.output_size)
#
#     def __call__(self, sample):
#         seed = np.random.randint(2**32-1)
#         for key in sample:
#             if isinstance(sample[key], Image.Image):
#                 random.seed(seed)
#                 sample[key] = self.random_crop(sample[key])
#         return sample


class RandomHorizontalFlipAll(): # pylint: disable=too-few-public-methods
    """Flip the image horizontally with probability flip_prob.
    """
    def __init__(self, flip_prob, keys):
        self.flip_prob = flip_prob
        self.keys = keys

    def __call__(self, sample):
        flip = np.random.random() <= self.flip_prob
        for key in self.keys:
            if flip:
                sample[key] = np.copy(np.fliplr(sample[key]))
        return sample


class Normalize(object): # pylint: disable=too-few-public-methods
    """Subtract the mean and divide by the variance for one of the objects in the dataset."""
    def __init__(self, mean, var, key):
        """
        mean - np.array of size 3 - the means of the three color channels over the train set
        var - np.array of size 3 - the variances of the three color channels over the train set
        """
        self.mean = mean
        self.var = var
        self.key = key

    def __call__(self, sample):
        sample[self.key + "_orig"] = np.copy(sample[self.key])
        sample[self.key] = (sample[self.key] - self.mean)/self.var
        return sample

class AddDepthMask(): # pylint: disable=too-few-public-methods
    """Creates a mask that is 1 where actual depth values were recorded
    (i.e. depth != min_depth and depth != max_depth)

    Adds the mask based on the depth map in sample[key].
    """
    def __init__(self, min_depth, max_depth, key):
        self.key = key
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __call__(self, sample, eps=1e-6):
        sample["mask"] = ((sample[self.key] > self.min_depth) & (sample[self.key] < self.max_depth)).astype(np.float32)
        return sample


class ToTensorAll(): # pylint: disable=too-few-public-methods
    """Convert ndarrays in sample to Tensors.
    Outputs should have 3 dimensions i.e. len(sample[key].size()) == 3

    Behavior:
    1D arrays [N]: Append two dimensions to make shape [N, 1, 1]
    2D arrays [M, N]: Prepend dimension of size 1 to make shape [1, M, N]
    3D arrays [M, N, C]: transpose(2, 0, 1) to put channels first to make shape [C, M, N]

    """
    def __init__(self, keys):
        self.keys = keys

    def __call__(self, sample):
        for key in self.keys:
            arr = sample[key]
            if len(arr.shape) == 1:
                sample[key] = torch.from_numpy(arr).unsqueeze(-1).unsqueeze(-1).float()
            elif len(arr.shape) == 2:
                sample[key] = torch.from_numpy(arr).unsqueeze(0).float()
            elif len(arr.shape) == 3:
                sample[key] = torch.from_numpy(arr.transpose(2, 0, 1)).float()
            else:
                raise TypeError("Array with key {} has {} > 3 dimensions".format(key, len(arr.shape)))
        # if 'depth_sid' in sample:
        #     sample["depth_sid"] = torch.from_numpy(sample['depth_sid'].transpose(2, 0, 1)).float()
        #     sample["depth_sid_index"] = torch.from_numpy(sample["depth_sid_index"]).unsqueeze(0).long()
#         print(output)
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
            weights = weights * np.mean(sample["albedo"]) # Attenuate by the average albedo
        if self.use_squared_falloff:
            weights[depth == 0] = 0.
            weights[depth != 0] = weights[depth != 0] / (depth[depth != 0]**2)
        if not self.use_albedo and not self.use_squared_falloff:
            sample["hist"], _ = np.histogram(depth, **self.hist_kwargs)
        else:
            sample["hist"], _ = np.histogram(depth, weights=weights, **self.hist_kwargs)
        return sample



class AddRawDepthMask(): # pylint: disable=too-few-public-methods
    """Creates a mask according to the original paper Eigen et. al.
    Assigns a 0 to pixels where the minimum or maximum depth is attained.
    Assigns a 1 everywhere else.
    """
    def __init__(self, min_depth, max_depth):
        self.min_depth = min_depth
        self.max_depth = max_depth

    def __call__(self, sample):
        rawdepth = sample["rawdepth"]
        boolmask = (rawdepth >= self.max_depth) | (rawdepth <= self.min_depth)
        sample["mask"] = 1. - boolmask.astype(np.float32) # Logical NOT
        return sample


class AddSIDDepth():
    """Creates a copy of the depth image where the depth value has been replaced
    by the SID-discretized index

    Discretizes depth into |sid_bins| number of bins, where the edges of the bins are
    given by

    t_i = exp(log(alpha) + i/K*log(beta/alpha))

    for i in {0,...,K}.

    Works in numpy.

    offset is a shift term that we subtract from alpha
    """
    def __init__(self, sid_bins, max_val, min_val, offset, key):
        """
        :param sid_obj: The SID object to use to convert between indices and depth values and vice versa
        :param key: The key (in sample) of the depth map to use.
        """
        self.key = key # Key of the depth image to convert to SID form.
        self.sid_obj = SID(sid_bins, max_val, min_val, offset)
        

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
             Note: The most 1's possible is n, where n is the number of bins:
             [1 1 1 1 1 1 1]
             The fewest is 0.
        """
        depth = sample[self.key]
        sample[self.key + "_sid_index"] = self.sid_obj.get_sid_index_from_value(depth)
        K = np.zeros((self.sid_obj.sid_bins,) + depth.shape)
        for i in range(self.sid_obj.sid_bins): # i = {0, ..., self.sid_bins - 1}
            K[i, :, :] = K[i, :, :] + i * np.ones(depth.shape)
        sample[self.key + "_sid"] = (K < sample["depth_sid_index"][np.newaxis, :, :]).astype(np.int32)
        return sample


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
