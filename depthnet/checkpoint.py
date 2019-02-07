import os
import sys
from datetime import datetime
import socket

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import depthnet.model as m

def safe_makedir(path):
    """Makes a directory, or returns if the directory
    already exists.

    Taken from:
    https://stackoverflow.com/questions/273192/how-can-i-safely-create-a-nested-directory-in-python
    """
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def load_checkpoint(ckpt_file, device):
    """Loads a checkpoint from a checkpointfile.
    Checkpoint is a dict consisting of:

    network_ckpt
    ----------
    network_name
    network_params
    network_state_dict

    train_ckpt
    ----------
    loss_fn (string)

    optim_name
    optim_params
    optim_state_dict

    scheduler_name
    scheduler_params
    last_epoch
    global_it

    ckpt_ckpt
    --------
    run_id
    log_dir
    ckpt_dir

    --
    Can derive network_name and network_state_dict from model
    Can derive scheduler_name from scheduler
       Can derive optim_name and optim_state_dict from scheduler.optimizer

    """
    if torch.cuda.is_available():
        checkpoint = torch.load(ckpt_file,
                                map_location="cuda")
    else:
        # Load GPU network on CPU
        checkpoint = torch.load(ckpt_file,
                                map_location=lambda storage,
                                                    loc: storage)
    network_update = {
        "network_name": checkpoint["network_name"],
        "network_params": checkpoint["network_params"],
        # Because of sacred - don't want to explode the config
        "network_state_dict_fn": lambda: checkpoint["network_state_dict"],
    }
    train_update = {
        "loss_fn": checkpoint["loss_fn"],
        "optim_name": checkpoint["optim_name"],
        # Because of sacred - don't want to explode the config
        "optim_state_dict_fn": lambda: checkpoint["optim_state_dict"],
        "scheduler_name": checkpoint["scheduler_name"],
        "scheduler_params": checkpoint["scheduler_params"],
        "last_epoch": checkpoint["last_epoch"],
        "global_it": checkpoint["global_it"],
    }
    ckpt_update = {
        "run_id": checkpoint["run_id"],
        "log_dir": checkpoint["log_dir"],
        "ckpt_dir": checkpoint["ckpt_dir"],
    }
    return network_update, train_update, ckpt_update


def save_checkpoint(network,
                    scheduler,
                    config,
                    state,
                    filename):
    """Save checkpoint using
    network - the current network
    scheduler - the scheduler (and optimizer)
    config - configuration of the network
    state - the current state of the training process
    """
    # Unpack
    network_config = config["network_config"]
    train_config = config["train_config"]
    ckpt_config = config["ckpt_config"]

    checkpoint = {
        "network_name": network.__class__.__name__,
        "network_params": network_config["network_params"],
        "network_state_dict": network.state_dict(),

        "loss_fn": train_config["loss_fn"],
        "optim_name": scheduler.optimizer.__class__.__name__,
        "optim_params": train_config["optim_params"],
        "optim_state_dict": scheduler.optimizer.state_dict(),
        "scheduler_name": scheduler.__class__.__name__,
        "scheduler_params": train_config["scheduler_params"],
        "last_epoch": state["last_epoch"],
        "global_it": state["global_it"],

        "run_id": ckpt_config["run_id"],
        "log_dir": ckpt_config["log_dir"],
        "ckpt_dir": ckpt_config["ckpt_dir"],
    }
    print("=> Saving checkpoint to: {}".format(filename))
    torch.save(checkpoint, filename)  # save checkpoint
