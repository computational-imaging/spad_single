#!/usr/bin/env python3
import os

import socket
import json
from datetime import datetime
from warnings import warn

from pprint import PrettyPrinter

import torch

from depthnet.model import make_network
from depthnet.data import data_ingredient, load_depth_data
from depthnet.checkpoint import load_checkpoint
from depthnet.model.loss import berhu, delta, rmse, rel_abs_diff, rel_sqr_diff
from depthnet.wrappers import make_wrapper

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
            "hist_len": 1000//3,                # Length of the histogram (hints only)
            "num_hints_layers": 4,              # Number of 1x1 conv layers for hints (hints only)
        },
        "model_state_dict_fn": None,            # Function for getting the state dict
    }

    test_config = {
        "dataset": "val",                       # {train, val, test}
        "mode": "run_tests",                    # {run_tests, check_nan}
        "output_file": "results.json",          # Output file for aggregate numerical results
        "losses_file": "losses.tar"             # Output file for per-image numerical results
    }

    comment = ""
    stop_early = False

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
    ckpt_config = {"ckpt_file": "checkpoints/Nov02_17-50-11_ares_hints/checkpoint_epoch_79.pth.tar"}
    # ckpt_config = {"ckpt_file": "checkpoints/Nov25_00-20-06_ares_hints/checkpoint_epoch_35.pth.tar"}
    # ckpt_config = {"ckpt_file": "checkpoints/Dec03_17-20-39_ares_hints/checkpoint_epoch_79.pth.tar"}

@ex.named_config
def test_rawhints():
    ckpt_config = {"ckpt_file": "checkpoints/Dec27_01-39-23_ares_rawhist/checkpoint_epoch_79.pth.tar"}
    data_config = {"hist_use_albedo": False, "hist_use_squared_falloff": False}
    test_config = {"output_file": "results_rawhints.json", "losses_file": "losses_rawhints.tar"}

@ex.named_config
def test_hints():
    ckpt_config = {"ckpt_file": "checkpoints/Dec27_01-39-23_ares_hints/checkpoint_epoch_79.pth.tar"}
    test_config = {"output_file": "results_hints.json", "losses_file": "losses_hints.tar"}

@ex.named_config
def test_nohints():
    ckpt_config = {"ckpt_file": "checkpoints/Dec27_01-39-23_ares_nohints/checkpoint_epoch_79.pth.tar"}
    test_config = {"output_file": "results_nohints.json", "losses_file": "losses_nohints.tar"}


def test_avg(model,
             dataset,
             device,
             loss_fns, # list of ("name", loss function) pairs.
             stop_early): # For testing
    results = {}
    losses = torch.zeros(len(loss_fns), len(dataset)) # Holds the average loss value for each image.
    # npixels = torch.zeros(len(loss_fns), len(dataset)) # Holds the number of valid pixels per image (determined by the mask)
    for i in range(len(dataset)):
        with torch.set_grad_enabled(False):
            data = dataset[i]
            for key in data:
                if isinstance(data[key], torch.Tensor):
                    data[key] = data[key].unsqueeze(0).to(device) # Batch size 1
            target = data["depth"]
            print(target)
            mask = data["mask"]
            # save_images(target, output_dir=".", filename="target_{}".format(i))
            target = target.to(device)
            mask = mask.to(device)
            output = model(data) # Includes postprocessing
            print(output)
            for j, (_, loss_fn) in enumerate(loss_fns):
                losses[j, i] = loss_fn(output, target, mask)
            if stop_early and i == 5:
                break
            # npixels[i, :] = torch.sum(data["mask"])
    # loss_values = torch.sum(losses*npixels, axis=1)/torch.sum(npixels)
    loss_values = torch.mean(losses, dim=1)
    for (loss_name, _), value in zip(loss_fns, loss_values):
        results[loss_name] = value.item()
    return results, losses




# To see the full configuration, run $ python train.py print_config
@ex.automain
def main(network_config,
         test_config,
         ckpt_config,
         data_config,
         device,
         stop_early):
    """Run stuff"""
    if ckpt_config["ckpt_file"] is None:
        warn("checkpoint file not specified! Will run on untrained model.")

    # Load data
    train_set, val_set, test_set = load_depth_data()
    if test_config["dataset"] == "val":
        dataset = val_set
    elif test_config["dataset"] == "test":
        dataset = test_set
    else:
        print("unknown dataset: {} - using val dataset instead".format(test_config["dataset"]))

    network = make_network(**network_config)
    network.to(device)

    stats_and_params = {}
    stats_and_params.update(network_config["network_params"])
    stats_and_params.update(data_config)
    stats_and_params.update(train_set.get_global_stats())

    model = make_wrapper(network=network, network_config=network_config,
                         pre_active=True, post_active=True, device=device,
                         **stats_and_params)

    # loss = get_loss(train_config["loss_fn"])
    if test_config["mode"] == "run_tests":
        print("Running tests...")
        loss_fns = []
        loss_fns.append(("berhu", berhu))
        loss_fns.append(("rmse", rmse))
        loss_fns.append(("delta1", lambda p, t, m: delta(p, t, m, threshold=1.25)))
        loss_fns.append(("delta2", lambda p, t, m: delta(p, t, m, threshold=1.25**2)))
        loss_fns.append(("delta3", lambda p, t, m: delta(p, t, m, threshold=1.25**3)))
        loss_fns.append(("rel_abs_diff", rel_abs_diff))
        loss_fns.append(("rel_sqr_diff", rel_sqr_diff))
        results, losses = test_avg(model, dataset, device, loss_fns, stop_early)

        # Save as a json
        with open(test_config["output_file"], "w") as f:
            json.dump(results, f)
        torch.save(losses, test_config["losses_file"])
        # print(losses)
    elif test_config["mode"] == "check_nan":
        print("Checking for NaNs...")
        for _, param in model.network.named_parameters():
            print(param)
            if torch.isnan(param).any():
                print("found nan")
