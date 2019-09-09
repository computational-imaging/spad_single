#! /usr/bin/env python3

import os
import torch
from torch.utils.data import DataLoader
from OrdNet import OrdNet
from tensorboardX import SummaryWriter
from models.core.checkpoint import safe_makedir
import datetime
from sacred import Experiment
from sacred.observers import FileStorageObserver
from nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data

ex = Experiment("train_ordnet", ingredients=["nyuv2_labeled_ingredient"])
safe_makedir(os.path.join("runs", "train"))
file_observer = FileStorageObserver.create(os.path.join("runs", "train"))
ex.observers.append(file_observer)

def load_model(checkpoint=None):
    if checkpoint is None:
        return OrdNet()
    net = OrdNet()
    net.load_state_dict(torch.load(checkpoint))
    return net

def load_optimizer(param_groups, lr, checkpoint=None):
    if checkpoint is None:
        return torch.optim.Adam(param_groups, lr)
    optimizer = torch.optim.Adam(param_groups, lr)
    optimizer.load_state_dict(torch.load(checkpoint))
    return optimizer

@ex.config
def cfg(data_config):
    model_name = "OrdNet"
    checkpoint = None
    batch_size = 5      # Batch size
    lr = 1e-5           # Initial Learning Rate
    num_epochs = 5      # Number of epochs to train
    lr_decay = 1e-1     # Amount to decay learning rate every epoch

    ckpt_config = {
        "ckpt_file": None,
        "ckpt_dir": "model",
        "run_id": os.path.join(datetime.now().strftime('%b%d'),
                               datetime.now().strftime('%H-%M-%S_')) +
                               model_name,
                               data_config["data_name"],
        "log_dir": "runs",
    }

    dataset_type = "train"
    seed = 95290421
    output_dir = "results"
    cuda_device = "0"  # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))

@ex.automain
def train(checkpoint):
    net = load_model(checkpoint)
    dataset = load_data(channels_first=True)
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=0,  # needs to be 0 to not crash autograd profiler.
                            pin_memory=True)

    writer = SummaryWriter(log_dir=file_observer.dir)
    # Log some stuff before the run begins
    writer.add_scalar("learning_rate", train_config["optim_params"]["lr"], 0)

    ckpt_dir = os.path.join(file_observer.dir, "checkpoints")
    safe_makedir(ckpt_dir)

