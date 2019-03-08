#! /usr/bin/env python3

import os
import json
import random
import re

from collections import defaultdict
from argparse import ArgumentParser
from typing import List, Any

from PIL import Image


def get_files_by_type(rootdir, file_types, ext="png"):
    """
    Generator function for traversing a directory structure in search of files of the form
    (.+)_{file_type}.{ext}
    where the match for the initial (.+) is used as the global id for this file.
    :param rootdir: The root directory to begin the search.
    :param file_types: List[str] of file types to extract. e.g. ["rgb", "depth"]
    :param ext: File extension (e.g. "png")
    :yield: The next file in the directory structure.
    """
    patterns = []
    for file_type in file_types:
        # e.g. If file_type="rgb" and ext="png", append the 2-tuple
        # ("rgb", re.compile("(.+)_rgb.png"))
        # to the list of patterns.
        patterns.append((file_type, re.compile("^(.+)_{}.{}".format(file_type, ext))))
    for dir_name, sub_dir_list, file_list in os.walk(rootdir):
        for file in file_list:
            relpath = os.path.relpath(dir_name, rootdir)
            for file_type, pattern in patterns:
                match = pattern.match(file)
                if match:
                    global_id = os.path.join(relpath, match.group(1)) # match for (.+)
                    file_path = os.path.join(relpath, file)
                    yield global_id, file_type, file_path


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
    # print(delimiters)
    # print(len(index))
    splits = []
    prev_index = 0
    for frac in delimiters:
        next_index = int(frac*len(index))
        # print(prev_index, next_index)
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

    :return: A dictionary mapping global_ids to dictionaries mapping file_types to file paths
    relative to rootdir.
    """
    index = defaultdict(dict)
    for global_id, file_type, file_path in get_files_by_type(rootdir, file_types, "png"):
        index[global_id][file_type] = file_path
    return index


if __name__ == '__main__':
    file_types = ["rgb", "depth", "rawdepth", "albedo"]
    index = build_index("./nyu_depth_v2_processed", file_types)
    # print(index.keys())
    print(len(index))
    s1, s2, s3, s4 = random_split(list(index.items()), [1, 1, 1, 1])
    print(len(index))
    print(len(s1))
    print(len(s2))
    print(len(s3))
    print(len(s4))
    print(dict(s1))
    # print(train.keys())
    # print(len(train))
    # bedroom_split = split_on_keywords(index, ["bedroom", "bathroom"])
    # print(len(bedroom_split))
