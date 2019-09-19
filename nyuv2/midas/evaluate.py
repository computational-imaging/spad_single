#!/usr/bin/env python3
import os
import torch
# from torch.utils.data import DataLoader
# from utils.train_utils import init_randomness
# from utils.eval_utils import evaluate_model_on_dataset
# from models.core.checkpoint import load_checkpoint, safe_makedir
from loss import get_depth_metrics
from sacred import Experiment
from sacred.observers import FileStorageObserver

import numpy as np
import pandas as pd

# Model
from MiDaSModel import get_midas, midas_gt_predict_masked
# Dataset
# from nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data

ex = Experiment('eval_midas_nyuv2_labeled')

@ex.config
def cfg():
    dataset_type = "test"
    entry = None
    save_outputs = True
    seed = 95290421
    small_run = 0

    model_path = os.path.join("MiDaS", "model.pt")
    crop = (20, 460, 24, 616)

    output_dir = "results"
    cuda_device = "0"                       # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))


@ex.automain
def main(model_path,
         crop,
         dataset_type,
         entry,
         save_outputs,
         output_dir,
         seed,
         small_run,
         device):

    # Load the data
    # dataset = load_data(channels_first=False, dataset_type=dataset_type)

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
        rgb = rgb_data[..., i]
        rawDepth = rawDepth_data[crop[0]:crop[1], crop[2]:crop[3], i]
        mask = ((rawDepth > 0.) & (rawDepth < 10.)).astype('float')
        print("Evaluating {}".format(i))
        # pred, pred_metrics = model.evaluate(data, device)
        pred = midas_gt_predict_masked(model, rgb, rawDepth, mask, crop, device)

        pred_metrics = get_depth_metrics(pred, rawDepth, mask)
        print(pred_metrics)
        for j, metric_name in enumerate(metric_list[:-1]):
            metrics[i, j] = pred_metrics[metric_name]

        metrics[i, -1] = np.sum(mask)
        # Option to save outputs:
        if save_outputs:
            outputs.append(pred)

    if save_outputs:
        np.save(os.path.join(output_dir, "midas_{}_outputs.npy".format(dataset_type)), np.concatenate(outputs, axis=0))

    # Save metrics using pandas
    metrics_df = pd.DataFrame(data=metrics, index=entry_list, columns=metric_list)
    metrics_df.to_pickle(path=os.path.join(output_dir, "midas_{}_metrics.pkl".format(dataset_type)))
    # Compute weighted averages:
    average_metrics = np.average(metrics_df.ix[:, :-1], weights=metrics_df.weight, axis=0)
    average_df = pd.Series(data=average_metrics, index=metric_list[:-1])
    average_df.to_csv(os.path.join(output_dir, "midas_{}_avg_metrics.csv".format(dataset_type)), header=True)
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(average_metrics[0],
                                                                            average_metrics[1],
                                                                            average_metrics[2],
                                                                            average_metrics[3],
                                                                            average_metrics[4],
                                                                            average_metrics[6]))
    print("wrote results to {}".format(output_dir))
