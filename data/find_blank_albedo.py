#! /usr/bin/env python3

from PIL import Image
import numpy as np
try:
    with open("blacklist.txt", "r") as f:
        blacklist = [line.strip() for line in f.readlines()]
except:
    blacklist = []

bad_imgs = []
for i in range(10335):
    print(i)
    if str(i) not in blacklist:
        img = np.asarray(Image.open("{}_albedo.png".format(i)))
        if np.sum(img[:, :, 1]) == 0:
            bad_imgs.append(i)
            print("bad")
    else:
        print("skipped")
    # if i == 10: # Debugging
    #     bad_imgs = [1234]
    #     break

with open("blank_albedo.txt", "w") as f:
    for imname in bad_imgs:
        f.write(str(imname) + "\n")
