#!/usr/bin/env python3
import os
import torch
import numpy as np
from utils.train_utils import init_randomness
from utils.eval_utils import evaluate_model_on_dataset
from collections import defaultdict
import json
from models.core.checkpoint import load_checkpoint, safe_makedir
from models.data.utils.transforms import AddDepthMask
from models import make_model
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Dataset
from models.data.nyuv2_test_split_dataset import nyuv2_test_split_ingredient, load_data

ex = Experiment('eval_dorn_nohints_test_split', ingredients=[nyuv2_test_split_ingredient])


# Tensorboardx
# writer = SummaryWriter()

@ex.config
def cfg(data_config):
    model_config = {  # Load pretrained model for testing
        "model_name": "DORN_nyu_nohints",
        "model_params": {
            "in_channels": 3,
            "in_height": 257,
            "in_width": 353,
            "frozen": True,
            "pretrained": True,
            "state_dict_file": os.path.join("models", "torch_params_nyuv2_BGR.pth.tar"),
        },
        "model_state_dict_fn": None
    }
    ckpt_file = None  # Keep as None
    save_outputs = True
    seed = 95290421
    small_run = 0

    # hyperparams = ["sgd_iters", "sinkhorn_iters", "sigma", "lam", "kde_eps", "sinkhorn_eps"]
    pdict = model_config["model_params"]
    del pdict

    # print(data_config.keys())
    output_dir = os.path.join("results",
                              data_config["data_name"],  # e.g. nyu_depth_v2
                              "{}_{}".format("test", small_run),
                              model_config["model_name"])  # e.g. DORN_nyu_nohints

    safe_makedir(output_dir)
    ex.observers.append(FileStorageObserver.create(os.path.join(output_dir, "runs")))

    cuda_device = "0"  # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    # print("after: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))
    if ckpt_file is not None:
        model_update, _, _ = load_checkpoint(ckpt_file)
        model_config.update(model_update)
        del model_update, _  # So sacred doesn't collect them.

@ex.automain
def main(model_config,
         save_outputs,
         output_dir,
         data_config,
         seed,
         small_run,
         device):
    # Load the model
    model = make_model(**model_config)
    model.eval()
    model.to(device)
    # model.sid_obj.to(device)

    # Load the data
    dataset = load_data(dorn_mode=True)

    init_randomness(seed)

    print("Evaluating the model on {}".format(data_config["data_name"]))
    all_metrics = defaultdict(dict)
    avg_metrics = defaultdict(float)
    total_pixels = 0.
    for i in range(len(dataset)):
        if small_run and i == small_run:
            break
        print("Evaluating {}".format(i))
        input_ = dataset[i]
        input_modified = {}
        input_modified["rgb"] = input_["rgb_cropped"].unsqueeze(0)  # The input (cropped to dorn input)
        input_modified["rgb_orig"] = input_["rgb_cropped_orig"].unsqueeze(0)  # The original rgb image (cropped to output size)
        input_modified["rawdepth_orig"] = input_["depth_cropped_orig"].unsqueeze(0)  # The output
        input_modified["mask_orig"] = input_["mask_orig"].unsqueeze(0)
        # Get valid pixels
        pred, metrics = model.evaluate(input_modified, device)
        all_metrics[i] = metrics

        total_pixels += torch.sum(input_modified["mask_orig"]).item()
        for metric_name in metrics:
            avg_metrics[metric_name] += (torch.sum(input_modified["mask_orig"]) * metrics[metric_name]).item()
        ## Save stuff ##
        if save_outputs:
            torch.save(pred, os.path.join(output_dir, "{:04d}.pt".format(i)), )
        ## Done saving stuff ##
        # Print running average:
        print({metric_name: avg_metrics[metric_name]/total_pixels for metric_name in avg_metrics})

    for metric_name in avg_metrics:
        avg_metrics[metric_name] /= total_pixels
    with open(os.path.join(output_dir, "avg_metrics.json"), "w") as f:
        json.dump(avg_metrics, f)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f)

    print(avg_metrics)
    print("wrote results to {}".format(output_dir))




