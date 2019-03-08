import os
import json

from argparse import ArgumentParser
from split_utils import random_split, split_dict_on_keywords, build_index

from PIL import Image
import numpy as np

def find_and_write_blank_albedo(index, rootdir, blacklist_file="blacklist.txt"):
    print("Finding blank albedo files...")
    try:
        with open(blacklist_file, "r") as f:
            blacklist = [line.strip() for line in f.readlines()]
    except IOError:
        # Create new blacklist
        blacklist = []

    for entry in index:
        # print(entry)
        if entry not in blacklist:
            albedo_file = os.path.join(rootdir, "{}_albedo.png".format(entry))
            try:
                img = np.asarray(Image.open(albedo_file))
                if np.sum(img[:, :, 1]) <= 0.:
                    blacklist.append(entry)
                    print(entry)
            except FileNotFoundError:
                # No albedo file found
                blacklist.append(entry)
                print(entry)
        else:
            print("\tskipped")

    blacklist = sorted(blacklist)
    with open(blacklist_file, "w") as f:
        for imname in blacklist:
            f.write(str(imname) + "\n")
    print("blacklist size: {}".format(len(blacklist)))


def read_nyu_splitfile(filename):
    out = []
    with open(filename, 'r') as f:
        for line in f.readlines():
            out.append(line.strip())
    return out


def main(opt):
    print("Building index from root directory {}".format(opt.data_dir))
    file_types = ["rgb", "depth", "rawdepth", "albedo"]
    index = build_index(opt.data_dir, file_types)
    print("Built index of size {}.".format(len(index)))

    if not opt.no_blank_albedo:
        find_and_write_blank_albedo(index, opt.data_dir,
                                    blacklist_file=os.path.join(opt.data_dir, opt.blacklist_file))

    train_split = read_nyu_splitfile(opt.train_splitfile)
    test_split = read_nyu_splitfile(opt.test_splitfile)

    # NYU Depth v2 doesn't have an official validation set.
    # Take a representative sample from train_split instead.
    train_val = split_dict_on_keywords(index, train_split)
    train, val = random_split(list(train_val.items()), [80, 20])
    train = dict(train)
    val = dict(val)
    print("Training set: {} points.".format(len(train)))
    print("Val set: {} points.".format(len(val)))
    test = split_dict_on_keywords(index, test_split)
    print("Testing set: {} points.".format(len(test)))

    # check overlap
    # overlap = set(train.keys()) & set(test.keys())
    # print("overlap: {}".format(overlap))
    missing = set(index.keys()) - (set(train.keys()) | set(val.keys()) | set(test.keys()))
    print("Missing: {}".format(len(missing)))
    print(list(missing)[:3])

    with open(os.path.join(opt.data_dir, opt.output_train_file), "w") as f:
        json.dump(train, f)
        print("Wrote train split to {}".format(os.path.join(opt.data_dir, opt.output_train_file)))
    with open(os.path.join(opt.data_dir, opt.output_val_file), "w") as f:
        json.dump(val, f)
        print("Wrote val split to {}".format(os.path.join(opt.data_dir, opt.output_val_file)))
    with open(os.path.join(opt.data_dir, opt.output_test_file), "w") as f:
        json.dump(test, f)
        print("Wrote test split to {}".format(os.path.join(opt.data_dir, opt.output_test_file)))

    info = {}
    for entry in index:
        info[entry] = {"keywords": entry.split(os.path.sep)}
    with open(os.path.join(opt.data_dir, opt.output_info_file), "w") as f:
        json.dump(info, f)
    print("Wrote info file to {}".format(os.path.join(opt.data_dir, opt.output_info_file)))

if __name__ == '__main__':
    parser = ArgumentParser(description="Generate train-val-test splits.")
    parser.add_argument("--split-only", action="store_true", help="Only generate the splits for the chosen directory.")
    parser.add_argument("--small", action="store_true", help="Generate a small dataset for overfitting.")
    parser.add_argument("--no-blank-albedo", action="store_true", help="Find and blacklist the blank albedo files.")
    parser.add_argument("--blacklist-file", default="blacklist.txt")
    parser.add_argument("--data-dir", default="nyu_depth_v2_scaled16")
    parser.add_argument("--train-splitfile", default=os.path.join("code_nyu", "nyu_train.txt"))
    parser.add_argument("--test-splitfile", default=os.path.join("code_nyu", "nyu_test.txt"))
    parser.add_argument("--output-train-file", default="train.json")
    parser.add_argument("--output-val-file", default="val.json")
    parser.add_argument("--output-test-file", default="test.json")
    parser.add_argument("--output-info-file", default="info.json")
    opt = parser.parse_args()
    main(opt)
