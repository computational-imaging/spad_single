#! /usr/bin/env python3

import os
import torch
import pandas as pd
import numpy as np
from sacred import Experiment
from nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data

from models.loss import get_depth_metrics

ex = Experiment("densedepth_median_rescaling", ingredients=[nyuv2_labeled_ingredient])


@ex.config
def cfg(data_config):
    dataset_type = "test"
    entry = None
    save_outputs = True
    seed = 95290421
    small_run = 0
    input_file = os.path.join("data", "densedepth_{}_outputs.npy".format(dataset_type))

    output_dir = "results"



@ex.automain
def run(dataset_type,
        entry,
        save_outputs,
        small_run,
        input_file,
        output_dir):
    dataset = load_data(channels_first=False, dataset_type=dataset_type)
    cnn_data = np.load(input_file)

    if entry is None:
        metric_list = ["delta1", "delta2", "delta3", "rel_abs_diff", "rmse", "mse", "log10", "weight"]
        metrics = np.zeros((len(dataset) if not small_run else small_run, len(metric_list)))
        entry_list = []
        outputs = []
        for i in range(len(dataset)):
            if small_run and i == small_run:
                break
            print("Running {}[{}]".format(dataset_type, i))
            entry_list.append(i)
            init = cnn_data[i, ...]
            gt = dataset[i]["depth_cropped"]
            pred = init * (torch.median(gt).item()/np.median(init))
            pred_metrics = get_depth_metrics(torch.from_numpy(pred).float(),
                                             gt,
                                             torch.ones_like(gt))
            for j, metric_name in enumerate(metric_list[:-1]):
                metrics[i, j] = pred_metrics[metric_name]

            metrics[i, -1] = torch.numel(gt)
            # Option to save outputs:
            if save_outputs:
                outputs.append(pred)

        if save_outputs:
            np.save(os.path.join(output_dir, "densedepth_median_{}_outputs.npy".format(dataset_type)),
                    np.concatenate(outputs, axis=0))

        # Save metrics using pandas
        metrics_df = pd.DataFrame(data=metrics, index=entry_list, columns=metric_list)
        metrics_df.to_pickle(path=os.path.join(output_dir, "densedepth_median_{}_metrics.pkl".format(dataset_type)))
        # Compute weighted averages:
        average_metrics = np.average(metrics_df.ix[:, :-1], weights=metrics_df.weight, axis=0)
        average_df = pd.Series(data=average_metrics, index=metric_list[:-1])
        average_df.to_csv(os.path.join(output_dir, "densedepth_median_{}_avg_metrics.csv".format(dataset_type)), header=True)
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
        print(
            "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(average_metrics[0],
                                                                                average_metrics[1],
                                                                                average_metrics[2],
                                                                                average_metrics[3],
                                                                                average_metrics[4],
                                                                                average_metrics[6]))
    else:
        raise NotImplementedError
