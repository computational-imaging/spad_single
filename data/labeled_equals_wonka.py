#! /usr/bin/env python
import numpy as np

wonka = np.load("nyu_depth_v2_wonka/eigen_test_rgb.npy")
labeled = np.load("nyu_depth_v2_labeled_numpy/test_images_.npy").transpose(3, 0, 1, 2) # N x H x W x C

img = labeled[0,:,:,:]
missing_matches = 0
wonka_unmatched = list(range(wonka.shape[0]))
for j in range(labeled.shape[0]):
    found_match = False
    for i in wonka_unmatched:
        if (wonka[i, :, :, :] == labeled[j, :, :, :]).all():
            print("Found a match for labeled[{}] at wonka[{}].".format(j, i))
            found_match = True
            wonka_unmatched.remove(i)
            break
    if not found_match:
        print("Failed to find match for labeled[{}].".format(j))
        missing_matches += 1

print("Wonka unmatched: ", wonka_unmatched)



