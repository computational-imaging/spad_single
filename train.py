#!/usr/bin/env python3
import os

import random
import socket
from datetime import datetime

from pprint import PrettyPrinter

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from tensorboardX import SummaryWriter

import numpy as np

### Project-specific loaders ###
from depthnet.model import (make_model, split_params_weight_bias, get_loss,
                            delta, rmse, rel_abs_diff, rel_sqr_diff)
from depthnet.data import data_ingredient, get_depth_loaders
from depthnet.train_utils import make_training, train
from depthnet.checkpoint import load_checkpoint, safe_makedir
### end ###

from sacred import Experiment

pp = PrettyPrinter(indent=4)
pprint = pp.pprint

ex = Experiment('train', ingredients=[data_ingredient])


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
            "upsampling": "bilinear",
        },
        "model_state_dict_fn": None,            # Function for getting the state dict
    }

    train_config = {
        "loss_fn": "berhu",
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
        "num_epochs": 10,
    }
    comment = ""

    ckpt_config = {
        "ckpt_file": None,
        "ckpt_dir": "checkpoints",
        "run_id": datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + comment,
        "log_dir": "runs",
    }

    seed = 2018
    cuda_device = "0"                       # The gpu index to run on. Should be a string
    test_run = False                        # Whether or not to truncate epochs for testing
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    # print("after: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))
    if ckpt_config["ckpt_file"] is not None:
        model_update, train_update, ckpt_update = load_checkpoint(ckpt_config["ckpt_file"], device)
        model_config.update(model_update)
        train_config.update(train_update)
        ckpt_config.update(ckpt_update)

        del model_update, train_update, ckpt_update
    # seed = 2018
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

@ex.named_config
def no_hints_80():
    comment = "_nohints"
    model_config = {"model_name": "UNet"}
    train_config = {
        "num_epochs": 80,
        "scheduler_params": {
            "milestones": [40]
        }
    }

@ex.named_config
def hints_80():
    comment = "_hints"
    model_config = {"model_name": "UNetWithHints"}
    train_config = {
        "num_epochs": 80,
        "scheduler_params": {
            "milestones": [40]
        }
    }

@ex.named_config
def multi_hints_80():
    comment = "_multihints"
    model_config = {"model_name": "UNetMultiScaleHints"}
    train_config = {
        "num_epochs": 80,
        "scheduler_params": {
            "milestones": [40]
        }
    }

@ex.named_config
def overfit_small():
    train_config = {
        "num_epochs": 100,
        "scheduler_params": {
            "milestones": [50]
        }
    }
    data_config = {
        "train_file": "data/sunrgbd_all/small.txt",
        "val_file": "data/sunrgbd_all/small.txt"
    }

def init_randomness(seed):
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

# To see the full configuration, run $ python train.py print_config
@ex.automain
def main(model_config,
         train_config,
         ckpt_config,
         device,
         test_run,
         seed):
    """Run stuff"""
    init_randomness(seed)
    # Load data
    train_loader, val_loader, _ = get_depth_loaders()
    model, scheduler, loss = make_training(model_config,
                                           train_config,
                                           device)
    config = {
        "model_config": model_config,
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
