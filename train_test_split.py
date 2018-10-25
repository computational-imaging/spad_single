#! /usr/bin/env python3

import os
from os.path import basename, normpath
from argparse import ArgumentParser
from PIL import Image

def main(opt):
    if not opt.split_only:
        outputdir = os.path.join("data", "sunrgbd_all")
        depthdir = "depth_bfx"
        rgbdir = "image"
        # depthext = ".png"
        # rgbext = ".jpg"
        counter = 0
        for dir_name, sub_dir_list, _ in os.walk(os.path.join("data", "SUNRGBD")):
            if "image" in sub_dir_list:
                # This is an image directory
                image_id = basename(dir_name)
                print(image_id)
                depthfile = None
                for depth_dir_name, _, depth_file_name in os.walk(os.path.join(dir_name, depthdir)):
                    depthfile = os.path.join(depth_dir_name, depth_file_name[0])
                    break
                depth_img = Image.open(depthfile)
                rgbfile = None
                for rgb_dir_name, _, rgb_file_name in os.walk(os.path.join(dir_name, rgbdir)):
                    rgbfile = os.path.join(rgb_dir_name, rgb_file_name[0])
                    break
                rgb_img = Image.open(rgbfile)

                depth_img.save(os.path.join(outputdir, "{}_depth.png".format(counter)))
                rgb_img.save(os.path.join(outputdir, "{}_rgb.png".format(counter)))
                counter += 1
        with open("info.txt", "w") as f:
            f.write(str(counter)+"\n")

        print("Wrote {} image(s) to {}.".format(counter, outputdir))

    # Do the split

    import random
    import os.path

    # dataDir = os.path.join("data", "sunrgbd_nyu")
    data_dir = os.path.join("data", "sunrgbd_all")

    total_img = None
    with open("info.txt", "r") as f:
        total_img = int(f.readline())
    print("total_img: {}".format(total_img))
    filenames = [("{}_depth.png".format(i), "{}_rgb.png".format(i)) for i in range(0, total_img)]
    random.seed(2018)
    random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

    if opt.small:
        small_filenames = filenames[:5]
        with open(os.path.join(data_dir, "small.txt"), "w") as f:
            for depth, rgb in small_filenames:
                f.write(depth + "," + rgb + "\n")
        return


    # 90-5-5 split
    split_1 = int(0.9 * len(filenames))
    split_2 = int(0.95 * len(filenames))
    train_filenames = filenames[:split_1]
    dev_filenames = filenames[split_1:split_2]
    test_filenames = filenames[split_2:]

    # Save to files
    for name, split in zip(["train", "val", "test"], [train_filenames, dev_filenames, test_filenames]):
        with open(os.path.join(data_dir, "{}.txt".format(name)), "w") as f:
            for depth, rgb in split:
                f.write(depth + "," + rgb + "\n")

if __name__ == '__main__':
    parser = ArgumentParser(description="Generate train-val-test splits.")
    parser.add_argument("--split-only", action="store_true", help="Only generate the splits for the chosen directory.")
    parser.add_argument("--small", action="store_true", help="Generate a small dataset for overfitting.")
    opt = parser.parse_args()
    main(opt)
