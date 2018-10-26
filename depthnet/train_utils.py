import os

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from depthnet.model import make_model, split_params_weight_bias, get_loss
from depthnet.checkpoint import load_checkpoint, save_checkpoint


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
                  ckpt_config,
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
def validate(loss, model, val_loader, epoch, writer=None):
    """Computes the validation error of the model on the validation set.
    val_loader should be a DataLoader.
    Returns an ordinary number (i.e. not a tensor)
    """
    data = next(iter(val_loader))
    input_ = {}
    for key in data:
        input_[key] = data[key].float()
        if torch.cuda.is_available():
            input_[key] = input_[key].cuda()

    depth = input_["depth"]
    output = model(input_)
    valloss = loss(output, depth).item()
    if writer is not None:
        mseloss = nn.MSELoss()
        writer.add_scalar("data/val_mseloss", mseloss(output, depth).item(), epoch)
        writer.add_scalar("data/valloss", valloss, epoch)
        rgb_input = vutils.make_grid(data["rgb"], nrow=2, normalize=True, scale_each=True)
        writer.add_image('image/rgb_input', rgb_input, epoch)

        depth_truth = vutils.make_grid(data["depth"], nrow=2, normalize=True, scale_each=True)
        writer.add_image('image/depth_truth', depth_truth, epoch)

        depth_output = vutils.make_grid(output, nrow=2, normalize=True, scale_each=True)
        writer.add_image('image/depth_output', depth_output, epoch)
    return loss(output, depth).item()


####################
# Run the training #
####################
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
    model_config = config["model_config"]
    train_config = config["train_config"]
    ckpt_config = config["ckpt_config"]

    start_epoch = train_config["last_epoch"] + 1
    num_epochs = train_config["num_epochs"]
    global_it = train_config["global_it"]

    # Other scalars
    mseloss = nn.MSELoss()

    for epoch in range(start_epoch, start_epoch + num_epochs):
        print("epoch: {}".format(epoch))
        data = None
        output = None
        for it, data in enumerate(train_loader):
            input_ = {}
            for key in data:
                input_[key] = data[key].float()
#                if torch.cuda.is_available():
                input_[key] = input_[key].to(device)

            depth = input_["depth"]
            # New batch
            scheduler.optimizer.zero_grad()
            output = model(input_)
            trainloss = loss(output, depth)
            trainloss.backward()
            scheduler.optimizer.step()
            global_it += 1

            if not it % 10:
                print("\titeration: {}\ttrain loss: {}".format(it, trainloss.item()))
            if writer is not None:
                writer.add_scalar("data/train_mseloss", mseloss(output, depth).item(), global_it)
                writer.add_scalar("data/trainloss", trainloss.item(), global_it)

            if test_run: # Stop after 5 batches
                if not (it + 1) % 5:
                    break
        # Checkpointing
        if val_loader is not None:
            valloss = validate(loss, model, val_loader, epoch, writer)
            print("End epoch {}\tval loss: {}".format(epoch, valloss))

        # Save the last batch output of every epoch
        if writer is not None:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), global_it)

        # Save checkpoint
        state = {
            "last_epoch": epoch,
            "global_it": global_it,
        }
        save_checkpoint(model,
                        scheduler,
                        config,
                        state,
                        filename=os.path.join(ckpt_config["ckpt_dir"],
                                              ckpt_config["run_id"],
                                              "checkpoint_epoch_{}.pth.tar".format(epoch)))
