#!/usr/bin/env python3
import os
import torch
from torch.utils.data import DataLoader
from utils.train_utils import init_randomness
from utils.eval_utils import evaluate_model_on_dataset
from models.core.checkpoint import load_checkpoint, safe_makedir
from models import make_model
from sacred import Experiment
from sacred.observers import FileStorageObserver

import numpy as np
import pandas as pd

# Model
from DenseDepthModel import DenseDepth
# Dataset
from nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data

ex = Experiment('eval_densedepth_nyuv2_labeled', ingredients=[nyuv2_labeled_ingredient])

@ex.config
def cfg(data_config):
    dataset_type = "test"
    entry = None
    save_outputs = True
    seed = 95290421
    small_run = 0

    output_dir = "results"
    safe_makedir(output_dir)
    cuda_device = "0"                       # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))


@ex.automain
def main(dataset_type,
         entry,
         save_outputs,
         output_dir,
         seed,
         small_run,
         device):

    # Load the data
    dataset = load_data(dataset_type=dataset_type)

    # Load the model
    model = DenseDepth()

    init_randomness(seed)

    if entry is None:
        dataloader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=0,  # needs to be 0 to not crash autograd profiler.
                                pin_memory=True)
        # if eval_config["save_outputs"]:

        with torch.no_grad():
            metric_list = ["delta1", "delta2", "delta3", "rel_abs_diff", "rmse", "mse", "log10", "weight"]
            metrics = np.zeros((len(dataset) if not small_run else small_run, len(metric_list)))
            entry_list = []
            outputs = []
            for i, data in enumerate(dataloader):
                # TESTING
                if small_run and i == small_run:
                    break
                entry = data["entry"][0]
                entry = entry if isinstance(entry, str) else entry.item()
                entry_list.append(entry)
                print("Evaluating {}".format(data["entry"][0]))
                # pred, pred_metrics = model.evaluate(data, device)
                pred, pred_metrics, pred_weight = model.evaluate(data["rgb"].to(device),
                                                                 data["depth_cropped"].to(device),
                                                                 torch.ones_like(data["depth_cropped"]).to(device))
                for j, metric_name in enumerate(metric_list[:-1]):
                    metrics[i, j] = pred_metrics[metric_name]

                metrics[i, -1] = pred_weight
                # Option to save outputs:
                if save_outputs:
                    outputs.append(pred.cpu().numpy())

            if save_outputs:
                np.save(os.path.join(output_dir, "densedepth_{}_outputs.npy".format(dataset_type)), np.concatenate(outputs, axis=0))

            # Save metrics using pandas
            metrics_df = pd.DataFrame(data=metrics, index=entry_list, columns=metric_list)
            metrics_df.to_pickle(path=os.path.join(output_dir, "densedepth_{}_metrics.pkl".format(dataset_type)))
            # Compute weighted averages:
            average_metrics = np.average(metrics_df.ix[:, :-1], weights=metrics_df.weight, axis=0)
            average_df = pd.Series(data=average_metrics, index=metric_list[:-1])
            average_df.to_csv(os.path.join(output_dir, "densedepth_{}_avg_metrics.csv".format(dataset_type)), header=True)
            print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
            print(
                "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(average_metrics[0],
                                                                                    average_metrics[1],
                                                                                    average_metrics[2],
                                                                                    average_metrics[3],
                                                                                    average_metrics[4],
                                                                                    average_metrics[6]))
        print("wrote results to {}".format(output_dir))

    else:
        input_unbatched = dataset.get_item_by_id(entry)
        # for key in ["rgb", "albedo", "rawdepth", "spad", "mask", "rawdepth_orig", "mask_orig", "albedo_orig"]:
        #     input_[key] = input_[key].unsqueeze(0)
        from torch.utils.data._utils.collate import default_collate
        data = default_collate([input_unbatched])

        # Checks
        entry = data["entry"][0]
        entry = entry if isinstance(entry, str) else entry.item()
        print("Entry: {}".format(entry))
        # print("remove_dc: ", model.remove_dc)
        # print("use_intensity: ", model.use_intensity)
        # print("use_squared_falloff: ", model.use_squared_falloff)
        pred, pred_metrics, pred_weight = model.evaluate(data["rgb"].to(device),
                                                         data["depth_cropped"].to(device),
                                                         torch.ones_like(data["depth_cropped"]).to(device))
        if save_outputs:
            np.save(os.path.join(output_dir, "{}_{}_out.npy".format(dataset_type, entry)))
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
        print(
            "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(pred_metrics["delta1"],
                                                                                pred_metrics["delta2"],
                                                                                pred_metrics["delta3"],
                                                                                pred_metrics["rel_abs_diff"],
                                                                                pred_metrics["rms"],
                                                                                pred_metrics["log10"]))