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
    
    def get_global_stats(self, outFile=None, writeFile=False):
        """Calculate mean and variance of each rgb channel.
        
        Optionally caches the result of this calculation in outfile so it doesn't need to be done each
        time the dataset is loaded.
        """
        S = np.zeros(3)
        S_sq = np.zeros(3)
        npixels = 0.
        for depthFile, rgbFile in self.data:
            rgbImg = Image.open(os.path.join(self.dataDir, rgbFile))
            rgbImg = np.asarray(rgbImg, dtype=np.uint16)
#             print(rgbImg[0:10, 0:10, :])
            
            npixels += rgbImg.shape[0]*rgbImg.shape[1]
            for channel in range(rgbImg.shape[2]):
                S[channel] += np.sum(rgbImg[:,:,channel])
                S_sq[channel] += np.sum((rgbImg[:,:,channel])**2)
        mean = S/npixels
        var = S_sq/npixels - mean**2
        
        # Load full dataset (memory-intensive)
#         full = []
#         for depthFile, rgbFile in self.data:
#             rgbImg = Image.open(os.path.join(self.dataDir, rgbFile))
#             rgbImg = np.asarray(rgbImg, dtype=np.uint16)
#             full.append(rgbImg)
            
#         a = np.array(full)
#         mean_true = np.mean(a, axis=(0, 1, 2))
#         var_true = np.var(a, axis=(0, 1, 2))
#         print("actual mean and variance: {} {}".format(mean_true, var_true))
#         print(a.shape)
        return mean, var
                
        
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        depthFile, rgbFile = self.data[idx]
        depthImg = Image.open(os.path.join(self.dataDir, depthFile))
        rgbImg = Image.open(os.path.join(self.dataDir, rgbFile))
        sample = {"depth": depthImg, "rgb": rgbImg}
        if self.transform:
            sample = self.transform(sample)
        return sample

#############
# Load data #
#############

def load_data(trainFile, trainDir, valFile, valDir):
    """Generates training and validation datasets from
    text files and directories. Sets up datasets with transforms."""
    train = DepthDataset(trainFile, trainDir)
    mean, var = train.get_global_stats()
    train.transform = transforms.Compose([ToFloat(),
#                                        RandomCrop((400, 320)),
                                         CenterCrop((320, 400)),
                                         NormalizeRGB(mean, var),
                                         ToTensor()
                                         ])
    print("Loaded training dataset from {} with size {}.".format(trainFile, len(train)))
    val = None
    if valFile is not None:
        val = DepthDataset(valFile, valDir, 
                           transform=transforms.Compose([ToFloat(),
                                                         CenterCrop((320, 400)),
                                                         NormalizeRGB(mean, var),
                                                         ToTensor(),
                                                        ])
                          )

        print("Loaded val dataset from {} with size {}.".format(valFile, len(val)))
    return train, val
    
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
        left = w//2 - new_w//2
        right = w//2 + new_w//2 + (1 if new_w % 2 else 0)
        
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

class NormalizeRGB(object):
    def __init__(self, mean, var):
        """
        mean - np.array of size 3 - the means of the three color channels over the whole (training) dataset
        var - np.array of size 3 - the variances of the three color channels over the whole (training) dataset
        """
        self.mean = mean
        self.var = var
    def __call__(self, sample):
        sample["rgb"] -= self.mean
        sample["rgb"] /= np.sqrt(self.var)
#         print(sample["rgb"][0:10, 0:10, 0])
        return sample

