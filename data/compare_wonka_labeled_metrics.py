#! /usr/bin/env python3
import numpy as np
import json

home = "/home/markn1"

labeled_output = home + "/spad_single/results/nyu_depth_v2_labeled/test_0/DenseDepth/metrics.json"
wonka_output = home + "/DenseDepth/depth_scores.npy"
labeled_to_wonka_file = home + "/spad_single/data/labeled_to_wonka.json"
wonka_to_labeled_file = home + "/spad_single/data/wonka_to_labeled.json"

# Read mappings files
with open(labeled_to_wonka_file, "r") as f:
    labeled_to_wonka = json.load(f)
with open(wonka_to_labeled_file, "r") as f:
    wonka_to_labeled = json.load(f)

# Read metrics files
with open(labeled_output, "r") as f:
    labeled_metrics_dict = json.load(f)

# Turn labeled_metrics_dict into numpy array
print(len(labeled_metrics_dict))
labeled_metrics = np.zeros((6, len(labeled_metrics_dict)))
metric_list = ["delta1", "delta2", "delta3", "rel_abs_diff", "rmse", "log10"]
for i, metric in enumerate(metric_list):
    for j in labeled_metrics_dict:
        entry = str(j)
        val = labeled_metrics_dict[entry][metric]
        labeled_metrics[i, labeled_to_wonka[entry]] = val

wonka_metrics = np.load(wonka_output)

