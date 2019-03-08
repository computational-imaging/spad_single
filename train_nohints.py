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
from models.data.nyuv2_nohints import nyuv2_nohints_ingredient, load_data

ex = Experiment('train_nohints', ingredients=[noisy_cifar10_ingredient])

@ex.config
def cfg(data_config):
    model_config = {
        "model_name": "DenoisingUnetModel",
        "model_params": {
            "img_sidelength": 32
        },
        "model_state_dict_fn": None
    }
    train_config = {
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
        "num_epochs": 2,
    }
    comment = "denoising_unet"

    ckpt_config = {
        "ckpt_file": None,
        "ckpt_dir": "checkpoints",
        "run_id": os.path.join(datetime.now().strftime('%b%d'),
                               datetime.now().strftime('%H-%M-%S_')) +
                               model_config["model_name"] + "_" +
                               data_config["data_name"],
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
    # print(network)
    # Load data
    train_set, val_set, _ = load_data()

    init_randomness(seed)   # Initialize randomness for repeatability
    # Configure Data Loader using model wrapper, data
    train_loader = DataLoader(train_set,
                              batch_size=train_config["batch_size"],
                              shuffle=True,
                              num_workers=4,
                              pin_memory=False,
                              worker_init_fn=worker_init_randomness)

    val_loader = DataLoader(val_set,
                            batch_size=train_config["batch_size_val"],
                            shuffle=False,
                            num_workers=1,
                            pin_memory=False,
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
