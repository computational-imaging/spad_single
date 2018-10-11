###########
# Dataset #
###########
from PIL import Image
import torch
from torch.utils.data import Dataset
import csv, numpy as np
import os
from collections import defaultdict
from torchvision import transforms

class DepthDataset(Dataset):
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
        self.dataDir = dataDir
        self.transform = transform
        self.data = []
        with open(splitfile, "r") as f:
            for line in f.readlines():
                self.data.append(line.strip().split(","))
#         print(self.data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        depthFile, rgbFile = self.data[idx]
        # Extract depth file:
        depthImg = Image.open(os.path.join(self.dataDir, depthFile))
        # Extract rgb file:
        rgbImg = Image.open(os.path.join(self.dataDir, rgbFile))
        sample = {"depth": depthImg, "rgb": rgbImg}
        if self.transform:
            sample = self.transform(sample)
        # Normalize rgb image
        normalize = transforms.Normalize((0, 0, 0), (1, 1, 1))
        sample["rgb"] = normalize(sample["rgb"])
        return sample
        
##############
# Transforms #
##############
# for data augmentation
class RandomCrop(object):
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

class CenterCrop(object):
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
        left = w - new_w//2
        right = w + new_w//2 + (1 if new_w % 2 else 0)
        
        return {"depth": depth[top:bottom, left:right],
                "rgb": rgb[top:bottom, left:right, :]}
    
class Crop_8(object):
    """Crop to a size where both dimensions are divisible by 8"""
    
    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']
        new_h, new_w = (depth.shape[0]//8)*8, (depth.shape[1]//8)*8
        return {"depth": depth[:new_h, :new_w],
                "rgb": rgb[:new_h, :new_w, :]}
        
class Crop_small(object):
    def __call__(self, sample):
        depth, rgb = sample['depth'], sample['rgb']
        h, w = depth.shape[:2]
        x = 16
        return {"depth": depth[h//2-x:h//2+x, w//2-x:w//2+x],
                "rgb": rgb[h//2-x:h//2+x, w//2-x:w//2+x, :]}

class ToFloat(object):
    """Also parses the depth info for sunrgbd."""
    def __call__(self, sample):
        depth = sample['depth']
        x = np.asarray(depth, dtype=np.uint16)
        y = (x >> 3) | (x << 16-3)
        z = y.astype(np.float32)/1000
        z[z>8.] = 8. # Clip to maximum depth of 8m.
        return {"depth": z,
                "rgb": np.asarray(sample['rgb']).astype(np.float32)}
    
class ToTensor(object):
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
    
class AddDepthHist(object):
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

