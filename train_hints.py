#!/usr/bin/env python3
import os
import socket
import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from utils.train_utils import train, make_training, init_randomness, worker_init_randomness
from models.core.checkpoint import load_checkpoint, safe_makedir
from sacred import Experiment
from datetime import datetime

# Dataset
from models.data.nyuv2_official_hints_sid_dataset import nyuv2_hints_sid_ingredient, load_data

ex = Experiment('train_hints', ingredients=[nyuv2_hints_sid_ingredient])

@ex.config
def cfg(data_config):
    model_config = {
        "model_name": "DORN_nyu_hints",
        "model_params": {
            "in_channels": 3,
            "in_height": 257,
            "in_width": 353,
            "sid_bins": 68,
            "offset": data_config["offset"],
            "min_depth": data_config["min_depth"],
            "max_depth": data_config["max_depth"],
            "alpha": data_config["alpha"],
            "beta": data_config["beta"],
            "frozen": True,
            "pretrained": True,
            "state_dict_file": os.path.join("models", "torch_params_nyuv2_BGR.pth.tar"),

            # New for hints
            "hints_len": 68,
            "spad_weight": 1.,
        },
        "model_state_dict_fn": None
    }
    train_config = {
        "batch_size": 5,                       # Batch size to use for a single train step
        "batch_size_val": 5,                   # Batch size for a single validation step
        "optim_name": "Adam",
        "optim_params": {
            "lr": 1e-3,                         # Learning rate (initial)
            # "momentum": 0.9,
            "weight_decay": 1e-8,               # Strength of L2 regularization (weights only)
        },
        "optim_state_dict_fn": None,            # Function for getting the state dict
        "scheduler_name": "ReduceLROnPlateau",
        "scheduler_params": {
            "factor": 0.1,
            "patience": 2,
        },
        "global_it": 0,
        "num_epochs": 10,
        "last_epoch": -1,
    }
    comment = "hints"

    ckpt_config = {
        "ckpt_file": "checkpoints/Mar15/04-10-54_DORN_nyu_hints_nyu_depth_v2",
        "ckpt_dir": "checkpoints",
        "run_id": os.path.join(datetime.now().strftime('%b%d'),
                               datetime.now().strftime('%H-%M-%S_')) +
                               model_config["model_name"] + "_" +
                               data_config["data_name"] + "_" +
                               comment,
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
        model_update, train_update, ckpt_update = load_checkpoint(ckpt_config["ckpt_file"])
        model_config.update(model_update)
        train_config.update(train_update)
        ckpt_config.update(ckpt_update)

        del model_update, train_update, ckpt_update


# To see the full configuration, run $ python train_cifar10.py print_config
@ex.automain
def main(model_config,
         train_config,
         ckpt_config,
         device,
         test_run,
         seed):
    """Run stuff"""

    # Load network, scheduler, loss
    model, scheduler = make_training(model_config,
                                     train_config,
                                     device)
    model.to(device)
    # Apply custom learning rates
    optimizer = scheduler.optimizer

    # model.sid_obj.to(device)
    # print(network)
    # Load data
    train_set, val_set, _ = load_data()

    init_randomness(seed)   # Initialize randomness for repeatability
    # Configure Data Loader using model wrapper, data
    train_loader = DataLoader(train_set,
                              batch_size=train_config["batch_size"],
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True,
                              worker_init_fn=worker_init_randomness)

    val_loader = DataLoader(val_set,
                            batch_size=train_config["batch_size_val"],
                            shuffle=False,
                            num_workers=1,
                            pin_memory=True,
                            worker_init_fn=worker_init_randomness)

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
    # Log some stuff before the run begins
    writer.add_scalar("learning_rate", train_config["optim_params"]["lr"], 0)

    # Run Training
    train(model,
          scheduler,
          train_loader,
          val_loader,
          config,
          device,
          writer,
          test_run)

    writer.close()
