#! /usr/bin/env python3

import os
from argparse import ArgumentParser
from PIL import Image
import random
import json

def main(opt):
    parsedata = {}
    if not opt.split_only:
        rawdepthdir = "depth"
        depthdir = "depth_bfx"
        rgbdir = "image"
        # depthext = ".png"
        # rgbext = ".jpg"
        counter = 0
        for dir_name, sub_dir_list, _ in os.walk(opt.data_dir):
            if "image" in sub_dir_list:
                # This is an image directory
                print(os.path.basename(dir_name))
                # dir_path = os.path.dirname(dir_name)
                # print(os.path.basename(dir_path))
                # print(os.path.basename(os.path.dirname(dir_path)))
                # print(os.path.sep)
                # print(dir_name.split(os.path.sep))

                rawdepthfile = None
                for rawdepth_dir_name, _, rawdepth_file_names in os.walk(os.path.join(dir_name, rawdepthdir)):
                    rawdepthfile = os.path.join(rawdepth_dir_name, rawdepth_file_names[0])
                    break
                rawdepth_img = Image.open(rawdepthfile)

                depthfile = None
                for depth_dir_name, _, depth_file_names in os.walk(os.path.join(dir_name, depthdir)):
                    depthfile = os.path.join(depth_dir_name, depth_file_names[0])
                    break
                depth_img = Image.open(depthfile)

                rgbfile = None
                for rgb_dir_name, _, rgb_file_names in os.walk(os.path.join(dir_name, rgbdir)):
                    rgbfile = os.path.join(rgb_dir_name, rgb_file_names[0])
                    break
                rgb_img = Image.open(rgbfile)


                rawdepth_img.save(os.path.join(opt.output_dir, "{}_rawdepth.png".format(counter)))
                depth_img.save(os.path.join(opt.output_dir, "{}_depth.png".format(counter)))
                rgb_img.save(os.path.join(opt.output_dir, "{}_rgb.png".format(counter)))

                parsedata[counter] = {"keywords": dir_name.split(os.path.sep)}

                counter += 1
        # with open(os.path.join(opt.output_dir, "info.txt"), "w") as f:
            # f.write(str(counter)+"\n")
        parsedata["num_images"] = counter
        with open(os.path.join(opt.output_dir, "info.json"), "w") as f:
            json.dump(parsedata, f)

        print("Wrote {} image(s) to {}.".format(counter, opt.output_dir))

    # Do the split

    # dataDir = os.path.join("data", "sunrgbd_nyu")

    total_img = None
    # with open(os.path.join(opt.output_dir, "info.txt"), "r") as f:
    #     total_img = int(f.readline())
    with open(os.path.join(opt.output_dir, "info.json"), "r") as f:
        parsedata = json.load(f)
    num_images = parsedata["num_images"]
    print("total_img: {}".format(total_img))
    filenames = [(i, "{}_rawdepth.png".format(i), "{}_depth.png".format(i), "{}_rgb.png".format(i)) 
                 for i in range(0, num_images)]
    random.seed(2018)
    random.shuffle(filenames) # shuffles the ordering of filenames (deterministic given the chosen seed)

    if opt.small:
        small_filenames = filenames[:5]
        with open(os.path.join(opt.output_dir, "small.txt"), "w") as f:
            for counter, depthraw, depth, rgb in small_filenames:
                f.write(str(counter) + "," + depthraw + "," + depth + "," + rgb + "\n")
        return


    # 90-5-5 split
    split_1 = int(0.9 * len(filenames))
    split_2 = int(0.95 * len(filenames))
    train_filenames = filenames[:split_1]
    dev_filenames = filenames[split_1:split_2]
    test_filenames = filenames[split_2:]

    # Save to files
    for name, split in zip(["train", "val", "test"], [train_filenames, dev_filenames, test_filenames]):
        with open(os.path.join(opt.output_dir, "{}.txt".format(name)), "w") as f:
            for counter, depthraw, depth, rgb in split:
                f.write(str(counter) + "," + depthraw + "," + depth + "," + rgb + "\n")

if __name__ == '__main__':
    parser = ArgumentParser(description="Generate train-val-test splits.")
    parser.add_argument("--split-only", action="store_true", help="Only generate the splits for the chosen directory.")
    parser.add_argument("--small", action="store_true", help="Generate a small dataset for overfitting.")
    parser.add_argument("--data-dir", default=os.path.join("data", "SUNRGBD"))
    parser.add_argument("--output-dir", default=os.path.join("data", "sunrgbd_all"))
    opt = parser.parse_args()
    main(opt)
