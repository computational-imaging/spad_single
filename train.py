#!/usr/bin/env python3
import os

import random
import socket
from datetime import datetime

from pprint import PrettyPrinter

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from tensorboardX import SummaryWriter

import numpy as np

### Project-specific loaders ###
from depthnet.data import data_ingredient, load_depth_data as load_data, worker_init
from depthnet.train_utils import train, make_training
from depthnet.wrappers import make_wrapper
from depthnet.checkpoint import load_checkpoint, safe_makedir
### end ###

from sacred import Experiment

pp = PrettyPrinter(indent=4)
pprint = pp.pprint

ex = Experiment('train', ingredients=[data_ingredient])
ex.add_source_file(os.path.join("depthnet", "model", "unet_model.py"))
ex.add_source_file(os.path.join("depthnet", "model", "unet_parts.py"))


@ex.config
def cfg():
    network_config = {
        "network_name": "UNet",                 # Class of model to use (see model/utils.py)
        "network_params": {
            "wrapper_name": "DepthNetWrapper",  # {DepthNetWrapper, DORNWrapper}
            "input_nc": 3,                      # Number of input channels
            "output_nc": 1,                     # Number of output channels
            "hist_len": 1000//3,                # Length of the histogram (hints only)
            "num_hints_layers": 4,              # Number of 1x1 conv layers for hints (hints only)
            "len_hints_layers": 512,            # Number of units in the hints conv layers.
            "upsampling_mode": "bilinear",           # {bilinear, nearest}
        },
        "network_state_dict_fn": None,            # Function for getting the state dict
    }

    train_config = {
        "loss_fn": "berhu",                     # Loss function to use to train the network
        "target_key": "depth",                  # Key (index into data dict) for the object to compare the output of the network against
        "ground_truth_key": "depth",            # Key (index into data dict) for the ground truth depth image
        "batch_size": 20,                       # Batch size to use for a single train step
        "batch_size_val": 40,                   # Batch size for a single validation step
        "optim_name": "Adam",
        "optim_params": {
            "lr": 1e-2,                         # Learning rate (initial)
            "weight_decay": 1e-8,               # Strength of L2 regularization (weights only)
        },
        "optim_state_dict_fn": None,            # Function for getting the state dict
        "scheduler_name": "MultiStepLR",
        "scheduler_params": {
            "milestones": [10, 20],             # Learning rate milestone epochs
            "gamma": 0.1,                       # Gamma of MultistepLR decay
        },
        "last_epoch": -1,
        "global_it": 0,
        "num_epochs": 30,
    }
    comment = ""

    ckpt_config = {
        "ckpt_file": None,
        "ckpt_dir": "checkpoints",
        "run_id": datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + comment,
        "log_dir": "runs",
    }

    seed = 95290421
    cuda_device = "0"                       # The gpu index to run on. Should be a string
    test_run = False                        # Whether or not to truncate epochs for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))

    if ckpt_config["ckpt_file"] is not None:
        network_update, train_update, ckpt_update = load_checkpoint(ckpt_config["ckpt_file"], device)
        network_config.update(network_update)
        train_config.update(train_update)
        ckpt_config.update(ckpt_update)

        del network_update, train_update, ckpt_update

@ex.named_config
def unet_vincent():
    comment = "_unet_vincent"
    network_config = {
        "network_name": "Unet",
        "network_params": {
            "in_channels": 3,
            "out_channels": 1,
            "nf0": 128,
            "num_down": 4,
            "max_channels": 1024,
            "use_dropout": False,
            "outermost_linear": True
        }
    }
    train_config = {
        "batch_size": 5,
        "batch_size_val": 20,
        "target_key": "depth",
        "ground_truth_key": "depth",
        "num_epochs": 80,
        "optim_params": {"lr": 1e-4},
        "scheduler_params": {
            "milestones": [40, 60]
        }
    }

# @ex.named_config
# def restart():


@ex.named_config
def dorn():
    comment = "_dorn"
    network_config = {
        "network_name": "DORN",
        "network_params": {

        }
    }


@ex.named_config
def unet_dorn():
    comment = "_unet_dorn"
    network_config = {
        "network_name": "UNetDORN",
        "network_params": {
            "sid_bins": 80,
            "wrapper_name": "DORNWrapper",
        }
    }
    train_config = {
        "loss_fn": "ord_reg_loss",
        "target_key": "depth_sid",
        "ground_truth_key": "depth",
        "num_epochs": 10,
        "optim_params": {"lr": 1e-4},
        "scheduler_params": {
            "milestones": [4, 6]
        }
    }


@ex.named_config
def unet_dorn_hints():
    comment = "_unet_dorn_hints"
    network_config = {
        "network_name": "UNetDORNWithHints",
        "network_params": {
            "sid_bins": 80,
            "wrapper_name": "DORNWrapper",
        }
    }
    train_config = {
        "loss_fn": "ord_reg_loss",
        "target_key": "depth_sid",
        "ground_truth_key": "depth",
        "num_epochs": 10,
        "optim_params": {"lr": 1e-4},
        "scheduler_params": {
            "milestones": [4, 6]
        }
    }


@ex.named_config
def no_batchnorm_up():
    comment += "_no_batchnorm_up"
    network_config = {
        "network_params": {
            "upnorm": None
        }
    }
        

@ex.named_config
def no_hints_80():
    comment = "_nohints"
    network_config = {"network_name": "UNet"}
    train_config = {
        "num_epochs": 80,
        "optim_params": {"lr": 1e-3},
        "scheduler_params": {
            "milestones": [40]
        }
    }


@ex.named_config
def hints_80():
    comment = "_hints"
    network_config = {"network_name": "UNetWithHints"}
    train_config = {
        "num_epochs": 80,
        "optim_params": {"lr": 1e-3},
        "scheduler_params": {
            "milestones": [40]
        }
    }


@ex.named_config
def multi_hints_80():
    comment = "_multihints"
    network_config = {"network_name": "UNetMultiScaleHints"}
    train_config = {
        "num_epochs": 80,
        "scheduler_params": {
            "milestones": [40]
        }
    }


def init_randomness(seed):
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# To see the full configuration, run $ python train.py print_config
@ex.automain
def main(network_config,
         train_config,
         ckpt_config,
         data_config,
         device,
         test_run,
         seed):
    """Run stuff"""
    init_randomness(seed)   # Initialize randomness for repeatability

    # Load network, scheduler, loss
    network, scheduler, loss = make_training(network_config,
                                             train_config,
                                             device)
    print(network)
    # Load data
    train_set, val_set, _ = load_data()

    # Perform additional configuration on data transforms and model wrapper.
    stats_and_params = {}
    stats_and_params.update(network_config["network_params"])
    stats_and_params.update(data_config)
    stats_and_params.update(train_set.get_global_stats())

    # Configure data transform
    train_set.configure_transform(**stats_and_params)
    val_set.configure_transform(**stats_and_params)
    # print(train_set.transform)
    # print(val_set.transform)
    # Make and configure wrapper
    model = make_wrapper(network=network, network_config=network_config,
                         pre_active=False, post_active=False, device=device,
                         **stats_and_params)

    # Configure Data Loader using model wrapper, data
    train_loader = DataLoader(train_set,
                              batch_size=train_config["batch_size"],
                              shuffle=True,
                              num_workers=4,
                              pin_memory=False,
                              worker_init_fn=worker_init)

    val_loader = DataLoader(val_set,
                            batch_size=train_config["batch_size_val"],
                            shuffle=False,
                            num_workers=1,
                            pin_memory=False,
                            worker_init_fn=worker_init
                           )

    config = {
        "network_config": network_config,
        "train_config": train_config,
        "ckpt_config": ckpt_config,
    }
    writer = None

    # Create checkpoint directory
    safe_makedir(os.path.join(ckpt_config["ckpt_dir"],
                              ckpt_config["run_id"]))
    # Initialize tensorboardX writer
    writer = SummaryWriter(log_dir=os.path.join(ckpt_config["log_dir"],
                                                ckpt_config["run_id"]))
    # Log some stuff before the run begins
    if "write_globals" in dir(model):
        model.write_globals(writer)
    # Run Training
    train(model,
          scheduler,
          loss,
          train_loader,
          val_loader,
          config,
          device,
          writer,
          test_run)

    writer.close()
