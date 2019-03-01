#! /usr/bin/env python3

import os
import json
import random
import re

from collections import defaultdict
from argparse import ArgumentParser
from typing import List, Any

from PIL import Image


def random_split(index, proportions, seed=2018):
    """Split an index dictionary according to the proportions in |proportions|
    :param index: A sequence of things to split.
    :param delimiters: A list of numbers corresponding to the ratio of sizes of splits
            e.g. [90, 5, 5], [50, 50], etc.
    :param seed: The random seed.
    :return:

    """
    if len(proportions) == 1:
        return index # No split required.
    random.seed(seed)

    # shuffles the ordering of filenames (deterministic given the chosen seed)
    random.shuffle(index)

    delimiters = [sum(proportions[:i+1])/sum(proportions) for i in range(len(proportions) - 1)]
    splits = []
    prev_index = 0
    for i in range(len(delimiters)):
        next_index = int(delimiters[i]*len(index))
        splits.append(index[prev_index:next_index])
        prev_index = next_index
    splits.append(index[prev_index:])
    return splits


def split_dict_on_keywords(index: dict, keywords: List[str]):
    """Finds all entries in index whose key contains some entry in |keywords|
    as a substring."""
    split = {}
    for keyword in keywords:
        keyword_start = re.compile("^{}".format(keyword))
        split.update({key: index[key] for key in index if keyword_start.match(key)})
    return split


def build_index(rootdir: str, file_types: List[str], ext="png"):
    """
    :param rootdir - string - the name of the root of the dataset directory.
    :param file_types - list of string - the types of files to search for in the dataset
    :param ext - the file extension of the file types to search for.

    Given a rootdir of a dataset directory, create an index by looking for files of the form:

    (.+)_|file_type|.|ext|

    Then, return a dictionary
    """
    patterns = []
    for file_type in file_types:
        # e.g. If file_type="rgb" and ext="png", append the 2-tuple
        # ("rgb", re.compile("(.+)_rgb.png"))
        # to the list of patterns.
        patterns.append((file_type, re.compile("^(.+)_{}.{}".format(file_type, ext))))

    index = defaultdict(dict)
    for dir_name, sub_dir_list, file_list in os.walk(rootdir):
        for file in file_list:
            relpath = os.path.relpath(dir_name, rootdir)
            for file_type, pattern in patterns:
                match = pattern.match(file)
                if match:
                    global_id = os.path.join(relpath, match.group(1)) # match for (.+)
                    index[global_id][file_type] = os.path.join(relpath, file)
    return index


if __name__ == '__main__':
    file_types = ["rgb", "depth", "rawdepth", "albedo"]
    index = build_index("./data/nyu_depth_v2_processed", file_types)
    # print(index.keys())
    # print(len(index))
    s1, s2, s3, s4 = random_split(index, [1, 1,1, 1])
    print(len(index))
    print(len(s1))
    print(len(s2))
    print(len(s3))
    print(len(s4))
    # print(train.keys())
    # print(len(train))
    # bedroom_split = split_on_keywords(index, ["bedroom", "bathroom"])
    # print(len(bedroom_split))
