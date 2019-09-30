#!/usr/bin/env python3
import os
import torch
from loss import get_depth_metrics

import numpy as np
import pandas as pd

# Model
from MiDaSModel import get_midas, midas_gt_predict_masked

def main(model_path,
         crop,
         output_dir,
         device):
    # Load the data
    rgb_data = np.load("data/nyu_depth_v2_labeled_numpy/test_images.npy")
    print("rgb", rgb_data.shape)
    rawDepth_data = np.load("data/nyu_depth_v2_labeled_numpy/test_rawDepths.npy")
    print("rawDepth", rawDepth_data.shape)
    data_len = rgb_data.shape[3]

    # Load the model
    model = get_midas(model_path, device)

    metric_list = ["delta1", "delta2", "delta3", "rel_abs_diff", "rmse", "mse", "log10", "weight"]
    metrics = np.zeros((data_len, len(metric_list)))
    entry_list = []
    outputs = []
    for i in range(data_len):
        entry_list.append(i)
        rgb = rgb_data[..., i]
        rawDepth = rawDepth_data[crop[0]:crop[1], crop[2]:crop[3], i]
        mask = ((rawDepth > 0.) & (rawDepth < 10.)).astype('float')
        print("Evaluating {}".format(i))

        # Predict depth
        pred = midas_gt_predict_masked(model, rgb, rawDepth, mask, crop, device)

        # Compute metrics
        pred_metrics = get_depth_metrics(pred, rawDepth, mask)
        print(pred_metrics)
        for j, metric_name in enumerate(metric_list[:-1]):
            metrics[i, j] = pred_metrics[metric_name]

        metrics[i, -1] = np.sum(mask)   # Weight this prediction with the number of valid pixels
        # Option to save outputs:
        outputs.append(pred)

    np.save(os.path.join(output_dir, "midas_test_outputs.npy"), np.array(outputs))

    # Save metrics using pandas
    metrics_df = pd.DataFrame(data=metrics, index=entry_list, columns=metric_list)
    metrics_df.to_pickle(path=os.path.join(output_dir, "midas_test_metrics.pkl"))
    # Compute weighted averages:
    average_metrics = np.average(metrics_df.ix[:, :-1], weights=metrics_df.weight, axis=0)
    average_df = pd.Series(data=average_metrics, index=metric_list[:-1])
    average_df.to_csv(os.path.join(output_dir, "midas_test_avg_metrics.csv"), header=True)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(average_metrics[0],
                                                                            average_metrics[1],
                                                                            average_metrics[2],
                                                                            average_metrics[3],
                                                                            average_metrics[4],
                                                                            average_metrics[6]))
    print("wrote results to {}".format(output_dir))

if __name__ == '__main__':
    model_path = os.path.join("MiDaS", "model.pt")
    # crop = (20, 460, 24, 616)   # Standard crop
    crop = (0, 480, 0, 640)       # No crop

    output_dir = "results"
    cuda_device = "3"  # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))

    main(model_path, crop, output_dir, device)