#! /usr/bin/env python3
import h5py
import scipy.io as sio
import numpy as np
import json
import argparse
import os

# EIGEN_CROP = np.array([44, 471, 40, 601]) # Based on crop_image.m and get_projection_mask.m in nyuv2 toolbox
DEFAULT_CROP = np.array([20, 460, 24, 616]) # Based on https://github.com/ialhashim/DenseDepth

def crop_nyu(img, crop):
    """
    (test crop)
    :param img:
    :return:
    """
    return img[crop[0]:crop[1], crop[2]:crop[3], ...]

def json_save(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f)

def get_list_from_h5(h5file, list_key):
    """
    Given an hdf5.File object and a key corresponding to a list of strings
    stored in that object, returns a python list of those strings.
    :param h5file: The original hdf5 file e.g. labeled
    :param list_key: The key into the file corresponding to the list e.g. scenes
    :return: list of strings
    """
    refs = h5file[list_key][0]
    return ["".join(chr(i) for i in h5file[ref][:]) for ref in refs]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", help="root dir containing the *.mat files",
                        default="/media/data1/markn1")
    parser.add_argument("--labeled-path", help="location of nyu_depth_v2_labeled.mat",
                       default="nyu_depth_v2_labeled.mat")
    parser.add_argument("--split-path", help="location of splits.mat",
                        default="splits.mat")
    # parser.add_argument("--crop-type", choices={"wonka", "eigen"}, help="type of cropping to do",
    #                     default="wonka")
    parser.add_argument("--output-dir", help="where to save all the output data",
                        default=".")
    args = parser.parse_args()
    # crop = WONKA_CROP if args.crop_type == "wonka" else EIGEN_CROP
    crop = DEFAULT_CROP
    # Load data from .mat files
    labeled_path = os.path.join(args.root_dir, args.labeled_path)
    print("Loading data from {}...".format(args.labeled_path))

    data = {}
    labeled = h5py.File(labeled_path, 'r')
    data["images"] = np.array(labeled.get("images")).T        # H x W x C x N
    data["depths"] = np.array(labeled.get("depths")).T        # H x W x N
    data["rawDepths"] = np.array(labeled.get("rawDepths")).T  # H x W x N

    scenes = get_list_from_h5(labeled, "scenes")
    # print("Cropping according to {}...".format(args.crop_type))
    print("crop: ", crop)
    cropped = {}
    for key, arr in data.items():
        cropped[key  + "_cropped"] = crop_nyu(arr, crop)
    data.update(cropped)

    # Load split (subtract 1 since MATLAB is 1-indexed)
    split_path = os.path.join(args.root_dir, args.split_path)
    print("Loading split from {}...".format(split_path))
    split = sio.loadmat(split_path)
    trainNdxs = split["trainNdxs"].squeeze() - 1
    testNdxs = split["testNdxs"].squeeze() - 1

    # Split into train and test and save
    print("Saving to {}...".format(args.output_dir))
    for key, arr in data.items():
        np.save(os.path.join(args.output_dir, "train_{}.npy".format(key)),
                arr[..., trainNdxs])
        np.save(os.path.join(args.output_dir, "test_{}.npy".format(key)),
                arr[..., testNdxs])
    trainScenes = [scenes[i] for i in trainNdxs]
    testScenes = [scenes[i] for i in testNdxs]
    json_save(trainScenes, os.path.join(args.output_dir, "train_scenes.json"))
    json_save(testScenes, os.path.join(args.output_dir, "test_scenes.json"))
    np.save(os.path.join(args.output_dir, "crop.npy"), crop)


