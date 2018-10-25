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
    ### Start configuration ###
    model_name = "UNet"                 # {depth, depth_hints, unet} The model to train
    optim_name = "Adam"                 # The optimizer to use
    scheduler_name = "MultiStepLR"      # The scheduler to use.
    loss_fn = "berhu"                   # {berhu, l2, l1} loss function to use

    input_nc = 3                        # Number of input channels
    output_nc = 1                       # Number of output channels
    hist_len = 800//3                   # Length of the histogram (hints only)
    num_hints_layers = 4                # Number of 1x1 conv layers for hints (hints only)

    lr = 1e-3                           # Learning rate (initial)
    weight_decay = 1e-8                 # Strength of L2 regularization (weights only)

    milestones = [1, 2]                 # Learning rate milestone epochs
    gamma = 0.1                         # Gamma of MultistepLR decay

    num_epochs = 10                     # Number of epochs to train the model

    batch_size = 10                     # Number of training examples to use every iteration

    comment = ""                        # Comment for tensorboardX

    checkpointfile = None               # The checkpoint to load.
    checkpoint_dir = "checkpoint"       # The directory to save further checkpoints
    cuda_device = "0"                   # The gpu index to run on. Should be a string
    test_run = False                    # Whether or not to truncate epochs for testing

    train_file = os.path.join("data", "sunrgbd_all", "train.txt")
    train_dir = os.path.join("data", "sunrgbd_all")
    val_file = os.path.join("data", "sunrgbd_all", "val.txt")
    val_dir = os.path.join("data", "sunrgbd_all")
    ### End configuration ###

    ### Package everything nicely (Don't edit) ###

    train_setup = {
        "model_name": model_name,
        "optim_name": optim_name,
        "scheduler_name": scheduler_name,
        "loss_fn": loss_fn,
    }
    model_params = {
        "input_nc": input_nc,
        "output_nc": output_nc,
        "hist_len": hist_len,
        "num_hints_layers": num_hints_layers,
    }
    train_setup["model_params"] = model_params

    optim_params = {
        "lr": lr,
        "weight_decay": weight_decay,
    }
    train_setup["optim_params"] = optim_params

    scheduler_params = {
        "milestones": milestones,
        "gamma": gamma,
    }
    train_setup["scheduler_params"] = scheduler_params

    # Training parameters
    train_hyperparams = {
        "num_epochs": num_epochs,
    }
    train_setup.update(train_hyperparams)

    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    log_dir = os.path.join('runs', current_time + '_' + socket.gethostname() + comment)
    train_setup["log_dir"] = log_dir

    # Data
    data_setup = {
        "train_file": train_file,
        "train_dir": train_dir,
        "val_file": val_file,
        "val_dir": val_dir,
        "batch_size": batch_size,           # Number of training/val images per batch
    }

    checkpoint_info = (checkpointfile, checkpoint_dir)

@ex.named_config
def hints_config():
    model_name = "UNetWithHints"
    checkpoint_dir = "checkpoints_hints"


@ex.named_config
def overfit_small():
    num_epochs=100
    train_file="data/sunrgbd_all/small.txt"
    val_file="data/sunrgbd_all/small.txt"
    milestones=[50]

@ex.automain
def main(train_setup,
         data_setup,
         checkpoint_info,
         cuda_device,
         test_run):
    """Run stuff"""
    # Set up cuda
    # print(os.environ["CUDA_DEVICE_ORDER"])
    # print("before: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    # print("after: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))

    # Load data
    train_loader, val_loader = d.get_loaders(**data_setup)
    checkpointfile, checkpoint_dir = checkpoint_info
    if checkpointfile is not None:
        setup, metadata = u.load_checkpoint(checkpointfile, device)
    else:
        setup, metadata = u.make_training(train_setup, device)
    setup["num_epochs"] = train_setup["num_epochs"]
    setup["checkpoint_dir"] = checkpoint_dir
    # Print summary of setup
    print("Setup:")
    pprint(setup)
    pprint(setup["scheduler"].state_dict())
    pprint(setup["scheduler"].optimizer.state_dict())
    print("Metadata:")
    pprint(metadata)

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
