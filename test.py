#!/usr/bin/env python3
import os

import socket
from datetime import datetime

from pprint import PrettyPrinter

import torch
from tensorboardX import SummaryWriter

from depthnet.data import data_ingredient, load_depth_data
from depthnet.train_utils import evaluate
from depthnet.checkpoint import load_checkpoint, safe_makedir

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
        "loss_fn": "rmse"
    }

    comment = ""

    ckpt_config = {
        "ckpt_file": None,
        "ckpt_dir": "checkpoints",
        "run_id": datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname() + comment,
        "log_dir": "runs",
    }

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
        ckpt_config.update(ckpt_update)

        del model_update, train_update, ckpt_update
    # seed = 2018
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)

def test_avg(model,
             dataset,
             device,
             loss):
    total = 0.
    for i in range(len(dataset)):
        with torch.set_grad_enabled(False):
            data = dataset[i]
            for key in data:
                data[key] = data[key].unsqueeze(0) # Batch size 1
            test_loss = evaluate(loss, model, data, device=device, tag="test")
            total += test_loss.item()
    return total/len(dataset)




# To see the full configuration, run $ python train.py print_config
@ex.automain
def main(model_config,
         test_config,
         ckpt_config,
         device,
         test_run):
    """Run stuff"""
    # Load data
    _, _, test_dataset = load_depth_data()
    model = make_model(**model_config)
    model.to(device)
    loss = get_loss(train_config["loss_fn"])


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
