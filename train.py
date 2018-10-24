#!/usr/bin/env python3
import os
import socket
from datetime import datetime

from pprint import PrettyPrinter

import torch
from tensorboardX import SummaryWriter

# Load training options from file
import depthnet.utils as u
import depthnet.data as d
import depthnet.train_utils as tu

from sacred import Experiment

pp = PrettyPrinter(indent=4)
pprint = pp.pprint

ex = Experiment()

# Tensorboardx
# writer = SummaryWriter()

@ex.config
def cfg():
    # Model specification
    train_setup = {
        "model_name": "UNet",  # {depth, depth_hints, unet} The model to train
        "optim_name": "Adam",  # The optimizer to use
        "scheduler_name": "MultiStepLR", # The scheduler to use.
        "loss_fn": "berhu",    # {berhu, l2, l1} loss function to use
    }
    model_params = {
        "input_nc": 3,         # Number of input channels
        "output_nc": 1,        # Number of output channels
        "hist_len": 800//3,    # Length of the histogram (hints only)
        "num_hints_layers": 4, # Number of 1x1 conv layers for hints (hints only)
    }
    train_setup["model_params"] = model_params

    optim_params = {
        "lr": 1e-5,                 # Learning rate (initial)
        "weight_decay": 1e-8,       # Strength of L2 regularization (weights only)
    }
    train_setup["optim_params"] = optim_params

    scheduler_params = {
        "milestones": [25],         # Learning rate milestone epochs
        "gamma": 0.1,               # Gamma of MultistepLR decay
    }
    train_setup["scheduler_params"] = scheduler_params

    # Training parameters
    train_hyperparams = {
        "num_epochs": 50,           # Number of epochs to train the model
    }
    train_setup.update(train_hyperparams)

    comment = ""                    # log_dir for tensorboard writer.
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
    train_setup["log_dir"] = log_dir
    # Data
    data_setup = {
        "train_file": os.path.join("data", "sunrgbd_all", "train.txt"),
        "train_dir": os.path.join("data", "sunrgbd_all"),
        "val_file": os.path.join("data", "sunrgbd_all", "val.txt"),
        "val_dir": os.path.join("data", "sunrgbd_all"),
        "batch_size": 10,           # Number of training/val images per batch
    }
    # Other
    checkpointfile = None     # The checkpoint to load.
    checkpoint_dir = "checkpoint" # The directory to save further checkpoints.
    cuda_device = "0"         # The gpu index to run on. Should be a string.
    test_run = False          # Whether or not to truncate epochs for testing

@ex.automain
def main(train_setup,
         data_setup,
         checkpointfile,
         cuda_device,
         test_run):
    """Run stuff"""
    # Set up cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Set cuda device
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Make the device order as in nvidia-smi
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        print(os.environ["CUDA_DEVICE_ORDER"])
    print("using device: {} (CUDA VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))

    # Load data
    train_loader, val_loader = d.get_loaders(**data_setup)

    if checkpointfile is not None:
        setup, metadata = u.load_checkpoint(checkpointfile)
    else:
        setup, metadata = u.make_training(train_setup)

    setup["model"].to(device)
    # Print summary of setup
    print("Setup:")
    pprint(setup)
    pprint(setup["scheduler"].state_dict())
    pprint(setup["scheduler"].optimizer.state_dict())
    print("Metadata:")
    pprint(metadata)
    # print("device: {}".format(next(setup["model"].parameters()).device))
    # return # Testing
    writer = SummaryWriter(log_dir=metadata["log_dir"])
    # Run Training
    tu.train(setup,
             metadata,
             train_loader,
             val_loader,
             device,
             test_run,
             writer)

    writer.close()
