#! /usr/bin/env python3

import os
from os.path import basename, normpath
from argparse import ArgumentParser
from PIL import Image

parser = ArgumentParser(description="Generate train-val-test splits.")
parser.add_argument("--split-only", action="store_true", help="Only generate the splits for the chosen directory.")
opt = parser.parse_args()

if not opt.split_only:
    outputdir = os.path.join("data", "sunrgbd_all")
    depthdir = "depth_bfx"
    rgbdir = "image"
    depthext = ".png"
    rgbext = ".jpg"
    counter = 0
    for dirName, subDirList, fileName in os.walk(os.path.join("data", "SUNRGBD")):
        if "image" in subDirList:
            # This is an image directory
            image_id = basename(dirName)
            print(image_id)
            depthfile = None
            for depthDirName, _, depthFileName in os.walk(os.path.join(dirName, depthdir)):
                depthfile = os.path.join(depthDirName, depthFileName[0])
                break
            depthImg = Image.open(depthfile)
            rgbfile = None
            for rgbDirName, _, rgbFileName in os.walk(os.path.join(dirName, rgbdir)):
                rgbfile = os.path.join(rgbDirName, rgbFileName[0])
                break
            rgbImg = Image.open(rgbfile)
            
            depthImg.save(os.path.join(outputdir, "{}_depth.png".format(counter)))
            rgbImg.save(os.path.join(outputdir, "{}_rgb.png".format(counter)))
            counter += 1
    with open("info.txt", "w") as f:
        f.write(str(counter)+"\n")
                
    print("Wrote {} image(s) to {}.".format(counter, outputdir))
            
# Do the split

import random
import os.path

# dataDir = os.path.join("data", "sunrgbd_nyu")
dataDir = os.path.join("data", "sunrgbd_all")

totalImg = None
with open("info.txt", "r") as f:
    totalImg = int(f.readline())

filenames = [("{}_depth.png".format(i), "{}_rgb.png".format(i)) for i in range(0, counter)]
random.seed(2018)
random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

# 90-5-5 split
split_1 = int(0.9 * len(filenames))
split_2 = int(0.95 * len(filenames))
train_filenames = filenames[:split_1]
dev_filenames = filenames[split_1:split_2]
test_filenames = filenames[split_2:]

# Save to files
for name, split in zip(["train", "val", "test"], [train_filenames, dev_filenames, test_filenames] ):
    with open(os.path.join(dataDir, "{}.txt".format(name)), "w") as f:
        for depth, rgb in split:
            f.write(depth + "," + rgb + "\n")
                                            
    
