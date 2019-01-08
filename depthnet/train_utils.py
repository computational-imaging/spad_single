import os

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from depthnet.model import (make_model, split_params_weight_bias, get_loss,
                            delta, rmse, rel_abs_diff, rel_sqr_diff)
from depthnet.model.utils import ModelWrapper
from depthnet.checkpoint import save_checkpoint
from depthnet.utils import log_depth_data
from depthnet.model.unet_model import UNet, UNetWithHints


def make_optimizer(model, optim_name, optim_params, optim_state_dict_fn):
    opt_class = getattr(optim, optim_name)
    split_params = split_params_weight_bias(model)
    optimizer = opt_class(params=split_params, **optim_params)
    if optim_state_dict_fn is not None:
        optimizer.load_state_dict(optim_state_dict_fn())
    return optimizer

def make_scheduler(optimizer, scheduler_name, scheduler_params, last_epoch):
    scheduler_class = getattr(optim.lr_scheduler, scheduler_name)
    scheduler = scheduler_class(optimizer,
                                last_epoch=last_epoch,
                                **scheduler_params)
    return scheduler

def make_training(model_config,
                  train_config,
                  device):
    """
    Create a training instance, consisting of a (setup, metadata) tuple.

    In order to create a training instance, specify the following:

    model_config
    ------------
    model_name - string - the name of the model
    model_params - dict - keyword args for initializing the model
    model_state_dict_fn - function - State dict if initializing from checkpoint,
                                 None otherwise

    train_config
    ------------
    loss_fn - string - the name of the loss function to use.
    optim_name - The name of the class of the optimizer.
    optim_params - dict - keyword args for initializing the optimizer,
    optim_state_dict_fn - function - State dict if initializing from checkpoint,
                          None otherwise
    scheduler_name - string - the name of the class of the scheduler.
    scheduler_params - Keyword args for initializing the scheduler.
    last_epoch - int - The last epoch trained, or -1 if this is a new training instance.
    num_epochs - int - the number of (further) epochs to train the model.
    global_it - int - the global iteration. 0 for a new training instance.
    trainlosses - list - list of training losses up to now.
    vallosses - list - list of validation losses up to now.
    """
    # model
    model = make_model(**model_config)
    model.to(device)
    loss = get_loss(train_config["loss_fn"])

    # optimizer
    optimizer = make_optimizer(model,
                               train_config["optim_name"],
                               train_config["optim_params"],
                               train_config["optim_state_dict_fn"])
    scheduler = make_scheduler(optimizer,
                               train_config["scheduler_name"],
                               train_config["scheduler_params"],
                               train_config["last_epoch"])

    return model, scheduler, loss

##############
# Validation #
##############
def evaluate(loss, model, input_, target, mask, device="cpu", log_fn=None, log_kwargs=None):
    """Computes the error of the model on the data set.
    Returns an ordinary number (i.e. not a tensor)
    loss - callable - loss(prediction, target) should give the loss on the particular image or
    batch of images.

    target should be the tensor such that loss(output, data[target_key])
    is the correct thing to do.
    """
    for key in input_:
        if isinstance(input_[key], torch.Tensor):
            # print(key)
            input_[key] = input_[key].to(device)
    # print(input_["eps"].shape)
    target = target.to(device)
    mask = mask.to(device)
    output = model(input_)
    loss_value = loss(output, target, mask)
    if log_fn is not None:
        output = model.post(output)
        log_fn(loss, model, input_, output, target, mask, device, **log_kwargs)
    return loss_value


####################
# Run the training #
####################
# Customize this function depending on your train needs
def train(model,
          scheduler,
          loss,
          train_loader,
          val_loader,
          config,
          device,
          writer=None,
          test_run=False):
    # unpack
    train_config = config["train_config"]
    ckpt_config = config["ckpt_config"]

    start_epoch = train_config["last_epoch"] + 1
    num_epochs = train_config["num_epochs"]
    global_it = train_config["global_it"]

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print("epoch: {}".format(epoch))
        data = None
        for it, data in enumerate(train_loader):
            trainloss = evaluate(loss, model, data, data["depth"], data["mask"],
                                 device,
                                 log_fn=log_depth_data, log_kwargs={"writer": writer,
                                                                    "tag": "train",
                                                                    "it": global_it,
                                                                    "write_images": False})
            scheduler.optimizer.zero_grad()
            trainloss.backward()
            scheduler.optimizer.step()
            global_it += 1

            if not it % 10:
                print("\titeration: {}\ttrain loss: {}".format(it, trainloss.item()))

            if test_run: # Stop after 5 batches
                if it == 5:
                    break
        # Checkpointing
        if val_loader is not None:
            # One sample from validation set
            for it, data in enumerate(val_loader):
                if it == (global_it % len(val_loader)):
                    with torch.set_grad_enabled(False):
                        valloss = evaluate(loss, model, data, data["depth"], data["mask"],
                                           device,
                                           log_fn=log_depth_data, log_kwargs={"writer": writer,
                                                                              "tag": "val",
                                                                              "it": epoch,
                                                                              "write_images": True})
                        break
            print("End epoch {}\tval loss: {}".format(epoch, valloss))

        # Save the last batch output of every epoch
        if writer is not None:
            for name, param in model.network.named_parameters():
                if torch.isnan(param).any():
                    print("NaN detected.")
                else:
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), global_it)

        # Save checkpoint
        state = {
            "last_epoch": epoch,
            "global_it": global_it,
        }
        save_checkpoint(model.network,
                        scheduler,
                        config,
                        state,
                        filename=os.path.join(ckpt_config["ckpt_dir"],
                                              ckpt_config["run_id"],
                                              "checkpoint_epoch_{}.pth.tar".format(epoch)))
