#!/usr/bin/env python3
import os

import socket
import json
from datetime import datetime
from warnings import warn

from pprint import PrettyPrinter

import torch
from tensorboardX import SummaryWriter

from depthnet.model import make_model
from depthnet.data import data_ingredient, load_depth_data
from depthnet.train_utils import evaluate
from depthnet.checkpoint import load_checkpoint, safe_makedir
from depthnet.model.loss import berhu, delta, rmse, rel_abs_diff, rel_sqr_diff

from sacred import Experiment

pp = PrettyPrinter(indent=4)
pprint = pp.pprint

ex = Experiment('test', ingredients=[data_ingredient])


# Tensorboardx
# writer = SummaryWriter()

@ex.config
def cfg():
    model_config = {
        "model_name": "UNet",                   # {DepthNet, DepthNetWithHints, UNet, UNetWithHints}
        "model_params": {
            "input_nc": 3,                      # Number of input channels
            "output_nc": 1,                     # Number of output channels
            "hist_len": 800//3,                 # Length of the histogram (hints only)
            "num_hints_layers": 4,              # Number of 1x1 conv layers for hints (hints only)
        },
        "model_state_dict_fn": None,            # Function for getting the state dict
    }

    test_config = {
        "dataset": "val",                       # {train, val, test}
        "mode": "run_tests",                    # {run_tests, check_nan}
        "output_dir": "results.json"
    }

    comment = ""

    ckpt_config = {
        "ckpt_file": None,
        "ckpt_dir": "checkpoints",
        "run_id": datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + comment,
        "log_dir": "runs",
    }

    cuda_device = "0"                       # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    # print("after: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))
    if ckpt_config["ckpt_file"] is not None:
        model_update, _, ckpt_update = load_checkpoint(ckpt_config["ckpt_file"], device)
        model_config.update(model_update)
        ckpt_config.update(ckpt_update)

        del model_update, _, ckpt_update # So sacred doesn't collect them.

    # seed = 2018
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)


@ex.named_config
def current():
    # model_config = {"model_name": "UNet"}
    model_config = {"model_name": "UNetWithHints"}
    # ckpt_config = {"ckpt_file": "checkpoints/Nov02_17-50-11_ares_hints/checkpoint_epoch_79.pth.tar"}
    # ckpt_config = {"ckpt_file": "checkpoints/Nov25_00-20-06_ares_hints/checkpoint_epoch_35.pth.tar"}
    ckpt_config = {"ckpt_file": "checkpoints/Dec03_17-20-39_ares_hints/checkpoint_epoch_79.pth.tar"}

def test_avg(model,
             dataset,
             device,
             loss):
    losses = torch.zeros(len(dataset))
    for i in range(len(dataset)):
        with torch.set_grad_enabled(False):
            data = dataset[i]
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].unsqueeze(0) # Batch size 1
            test_loss = evaluate(loss, model, data, data["depth"], data["mask"], device=device)
            losses[i] = test_loss.item()
    return torch.mean(losses)




# To see the full configuration, run $ python train.py print_config
@ex.automain
def main(model_config,
         test_config,
         ckpt_config,
         device):
    """Run stuff"""
    if ckpt_config["ckpt_file"] is None:
        warn("checkpoint file not specified! Will run on untrained model.")

    # Load data
    _, val_dataset, test_dataset = load_depth_data()
    if test_config["dataset"] == "val":
        dataset = val_dataset
    else:
        dataset = test_dataset

    model = make_model(**model_config)
    model.to(device)
    # loss = get_loss(train_config["loss_fn"])
    if test_config["mode"] == "run_tests":
        print("Running tests...")
        test = lambda fn: test_avg(model, dataset, device, fn)
        results = {}
        results["berhu"] = test(berhu)
        results["rmse"] = test(rmse)
        results["delta1"] = test(lambda p, t, m: delta(p, t, m, threshold=1.25))
        results["delta2"] = test(lambda p, t, m: delta(p, t, m, threshold=1.25**2))
        results["delta3"] = test(lambda p, t, m: delta(p, t, m, threshold=1.25**3))
        results["rel_abs_diff"] = test(rel_abs_diff)
        results["rel_sqr_diff"] = test(rel_sqr_diff)
        print(results)

        # Save as a json
        with open(test_config["output_dir"], "w") as f:
            json.dump(results, f)
    elif test_config["mode"] == "check_nan":
        print("Checking for NaNs...")
        for _, param in model.named_parameters():
            print(param)
            if torch.isnan(param).any():

                print("found nan")
