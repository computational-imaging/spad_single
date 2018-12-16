#! /usr/bin/env python3

import os
import json
import random
import re

from collections import defaultdict
from argparse import ArgumentParser
from PIL import Image


def random_split(index, seed=2018):
    """Takes a dictionary where each entry is a datapoint and outputs
    a 90-5-5 train-val-test split of dictionaries.
    """
    random.seed(seed)
    list_index = list(index.items())
    # shuffles the ordering of filenames (deterministic given the chosen seed)
    random.shuffle(list_index)

    split_1 = int(0.9 * len(list_index))
    split_2 = int(0.95 * len(list_index))
    train_filenames = list_index[:split_1]
    val_filenames = list_index[split_1:split_2]
    test_filenames = list_index[split_2:]
    return dict(train_filenames), dict(val_filenames), dict(test_filenames)

def split_on_keywords(index, keywords):
    """Finds all entries in index whose key contains some entry in split
    as a substring."""
    split = {}
    for keyword in keywords:
        keyword_start = re.compile("^{}".format(keyword))
        split.update({key: index[key] for key in index if keyword_start.match(key)})
    return split

def build_index(rootdir, file_types=["rgb", "depth", "rawdepth", "albedo"]):
    patterns = []
    for file_type in file_types:
        # e.g. append the 2-tuple ("rgb", re.compile("(.+)_rgb.png")) to the list of patterns
        patterns.append((file_type, re.compile("^(.+)_{}.png".format(file_type))))

    index = defaultdict(dict)
    for dir_name, sub_dir_list, file_list in os.walk(rootdir):
        for file in file_list:
            relpath = os.path.relpath(dir_name, rootdir)
            for file_type, pattern in patterns:
                match = pattern.match(file)
                if match:
                    global_id = os.path.join(relpath, match.group(1))
                    index[global_id][file_type] = os.path.join(relpath, file)
    return index

if __name__ == '__main__':
    index = build_index("./data/nyu_depth_v2_processed")
    # print(index.keys())
    # print(len(index))
    # train, val, test = random_split(index)
    # print(train.keys())
    # print(len(train))
    bedroom_split = split_on_keywords(index, ["bedroom", "bathroom"])
    print(len(bedroom_split))
