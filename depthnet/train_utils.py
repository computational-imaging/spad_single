import os

import torch
import torch.optim as optim

import depthnet.model.loss as loss_fns
from depthnet.model import (make_network, split_params_weight_bias,
                            delta, rmse, rel_abs_diff, rel_sqr_diff)

import depthnet.wrappers as wrappers
from depthnet.checkpoint import save_checkpoint

from itertools import cycle

def make_optimizer(network, optim_name, optim_params, optim_state_dict_fn):
    opt_class = getattr(optim, optim_name)
    split_params = split_params_weight_bias(network)
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


def make_training(network_config,
                  train_config,
                  device):
    """
    Create a training instance, consisting of a (setup, metadata) tuple.

    In order to create a training instance, specify the following:

    network_config
    ------------
    network_name - string - the name of the network
    network_params - dict - keyword args for initializing the network
    network_state_dict_fn - function - State dict if initializing from checkpoint,
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
    num_epochs - int - the number of (further) epochs to train the network.
    global_it - int - the global iteration. 0 for a new training instance.
    """
    # network
    network = make_network(**network_config)
    network.to(device)

    # loss
    loss = getattr(loss_fns, train_config["loss_fn"])

    # optimizer
    optimizer = make_optimizer(network,
                               train_config["optim_name"],
                               train_config["optim_params"],
                               train_config["optim_state_dict_fn"])
    scheduler = make_scheduler(optimizer,
                               train_config["scheduler_name"],
                               train_config["scheduler_params"],
                               train_config["last_epoch"])

    return network, scheduler, loss


##############
# Validation #
##############
def evaluate(loss, model, input_, target, ground_truth, mask, device="cpu", log_kwargs=None):
    """Computes the error of the network on the data set.
    Returns an ordinary number (i.e. not a tensor)
    loss - callable - loss(prediction, target) should give the loss on the particular image or
    batch of images.

    target should be the tensor such that loss(output, data[target_key])
    is the correct thing to do.
    """
    for key in input_:
        if isinstance(input_[key], torch.Tensor):
            # print(key, input_[key].size())
            input_[key] = input_[key].to(device)
    # print(input_["eps"].shape)
    target = target.to(device)
    mask = mask.to(device)
    output = model(input_)
    loss_value = loss(output, target, mask)
    if "write_updates" in dir(model):
        with torch.no_grad():
            # Perform post-processing on the model output before passing to the log function.
            prediction = model.post(output)
            ground_truth = ground_truth.to(device)
            model.write_updates(loss, input_, output, target, prediction, ground_truth, mask,
                                device, **log_kwargs)
    if torch.isnan(loss_value).any():
        print("loss is nan")
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

    # Keys for indexing into data
    target_key = train_config["target_key"]
    ground_truth_key = train_config["ground_truth_key"]

    val_loader_iter = iter(val_loader)
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.network.train()
        print("epoch: {}".format(epoch))
        for it, data in enumerate(train_loader):
            trainloss = evaluate(loss, model, data, data[target_key], data[ground_truth_key], data["mask"],
                                 device,
                                 log_kwargs={"writer": writer,
                                             "tag": "train",
                                             "it": global_it,
                                             "write_images": not global_it % 100})
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
            ### TESTING
            # import gc
            # for obj in gc.get_objects():
            #     if torch.is_tensor(obj):
            #         print(type(obj), obj.size())
            ### TESTING
            # One sample from validation set
            try:
                data = next(val_loader_iter)
            except StopIteration:
                val_loader_iter = iter(val_loader)
                data = next(val_loader_iter) # Restart iterator
            with torch.no_grad():
                model.network.eval()
                valloss = evaluate(loss, model, data, data[target_key], data[ground_truth_key], data["mask"],
                                   device,
                                   log_kwargs={"writer": writer,
                                               "tag": "val",
                                               "it": epoch,
                                               "write_images": True})
            print("End epoch {}\tval loss: {}".format(epoch, valloss))
            del data, valloss # Clean up
        # # Save the last batch output of every epoch
        # if writer is not None:
        #     for name, param in model.network.named_parameters():
        #         if torch.isnan(param).any():
        #             print("NaN detected.")
        #         else:
        #             writer.add_histogram(name, param.clone().cpu().data.numpy(), global_it)

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
        if device.type == 'cuda':
            # print(torch.cuda.get_device_name(0))
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
            torch.cuda.empty_cache()
