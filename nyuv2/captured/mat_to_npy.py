#! /usr/bin/env python3

import scipy.io as sio
import numpy as np

# Extract root
def mat_to_npy(spad_file):
    root = spad_file.rpartition(".")[0]
    npy_file = root + ".npy"
    spad_data = sio.loadmat(spad_file)
    np.save(npy_file, spad_data)
    print("{} -> {}".format(spad_file, npy_file))

if __name__ == '__main__':
    # Get the file path somehow
    spad_files = [
        "spad1.mat"
    ]
    for file in spad_files:
        mat_to_npy(file)
