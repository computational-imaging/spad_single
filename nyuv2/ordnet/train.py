#! /usr/bin/env python3

import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
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
        return optim.Adam(param_groups, lr)
    optimizer = optim.Adam(param_groups, lr)
    optimizer.load_state_dict(torch.load(checkpoint))
    return optimizer

def scale_invariant_error(pred, gt, lam = 1., eps=1e-5):
    """
    Taken from Eigen et. al. Multi Scale Network paper
    :param pred: Predicted Depth, N x C x H x W
    :param gt: Ground Truth Depth, N x C x H x W
    :param lam: tradeoff parameter that controls strength of ordinal term.
    :param eps: Amount to add to pred and gt to avoid numerical issues
    :return: The scale invariant error
    """
    d = torch.log(pred + eps) - torch.log(gt + eps)
    loss = torch.mean(d**2) - (lam/torch.numel(d)**2) * torch.sum(d)**2
    return loss

@ex.config
def cfg(data_config):
    model_name = "OrdNet"
    checkpoint = None
    batch_size = 5      # Batch size
    lr = 1e-5           # Initial Learning Rate
    num_epochs = 10     # Number of epochs to train
    milestones = [5, 7] # When to decay the learning rate
    lr_decay = 1e-1     # Amount to decay learning rate at milestone epochs

    ckpt_config = {
        "ckpt_file": None,
        "ckpt_dir": "model",
        "run_id": os.path.join(datetime.now().strftime('%b%d'),
                               datetime.now().strftime('%H-%M-%S_') +
                               model_name,
                               data_config["data_name"]),
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


def train_epoch(model, scheduler, train_loader):
    for i, input_ in enumerate(train_loader):
        output = model.predict(input_["rgb"])
        output_cropped = d
        loss = scale_invariant_error(output, input_["gt_cropped"])


@ex.automain
def train(batch_size,
          lr,
          num_epochs,
          milestones,
          lr_decay,
          checkpoint):
    net = load_model(checkpoint)
    train_dataset = load_data(channels_first=True, dataset_type="train")
    val_dataset = load_data(channels_first=True, dataset_type="val")


    writer = SummaryWriter(log_dir=file_observer.dir)
    # Log some stuff before the run begins
    writer.add_scalar("learning_rate", lr, 0)

    ckpt_dir = os.path.join(file_observer.dir, "checkpoints")
    safe_makedir(ckpt_dir)

    model = OrdNet()

    optimizer = load_optimizer(model.parameters(), lr, checkpoint)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones, lr_decay)

    for epoch in range(len(num_epochs)):
        train_loader = DataLoader(train,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=0,  # needs to be 0 to not crash autograd profiler.
                            pin_memory=True)
        val_loader = DataLoader(val_dataset,
                                batch_size=10,
                                shuffle=True,
                                num_workers=0,  # needs to be 0 to not crash autograd profiler.
                                pin_memory=True)
        model = train_epoch(model, scheduler, train_loader)

        evaluate(model, val_loader)

