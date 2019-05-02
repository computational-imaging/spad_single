import os

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import random
import numpy as np

from models import make_model
from models.core.checkpoint import save_checkpoint


def init_randomness(seed):
    """
    Set the random seed across all libraries for replicability.
    (well, as much as possible at least...)
    :param seed: The random seed.
    :return: None
    """
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def worker_init_randomness(worker_id):
    """
    Initialize dataloader workers the same way for replicability
    :param worker_id: Passed by the dataloader initializer upon startup
    :return: None
    """
    cudnn.deterministic = True
    random.seed(1 + worker_id)
    np.random.seed(1 + worker_id)
    torch.manual_seed(1 + worker_id)
    torch.cuda.manual_seed(1 + worker_id)


def make_optimizer(model, optim_name, optim_params, optim_state_dict_fn):
    opt_class = getattr(optim, optim_name)
    param_groups = model.get_param_groups()
    optimizer = opt_class(params=param_groups, **optim_params)
    if optim_state_dict_fn is not None:
        optimizer.load_state_dict(optim_state_dict_fn())
    return optimizer


def make_scheduler(optimizer, scheduler_name, scheduler_params, last_epoch):
    scheduler_class = getattr(optim.lr_scheduler, scheduler_name)
    if scheduler_name is "ReduceLROnPlateau":
        scheduler = scheduler_class(optimizer,
                                    **scheduler_params)
    else:
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
    mdoel_name - string - the name of the network
    model_params - dict - keyword args for initializing the network
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
    num_epochs - int - the number of (further) epochs to train the network.
    global_it - int - the global iteration. 0 for a new training instance.
    """
    # network
    model = make_model(**model_config)
    model.to(device)

    # optimizer
    optimizer = make_optimizer(model,
                               train_config["optim_name"],
                               train_config["optim_params"],
                               train_config["optim_state_dict_fn"])
    scheduler = make_scheduler(optimizer,
                               train_config["scheduler_name"],
                               train_config["scheduler_params"],
                               train_config["last_epoch"])

    return model, scheduler


def train(model,
          scheduler,
          train_loader,
          val_loader,
          config,
          device,
          writer=None,
          test_run=False):
    train_config = config["train_config"]
    ckpt_config = config["ckpt_config"]

    start_epoch = train_config["last_epoch"] + 1
    num_epochs = train_config["num_epochs"]
    global_it = train_config["global_it"]

    its_per_epoch = len(train_loader.dataset)//train_loader.batch_size + 1

    val_loader_iter = iter(val_loader)
    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        # print("epoch: {}".format(epoch))
        for it, input_ in enumerate(train_loader):
            trainloss, output = model.get_loss(input_, device)
            with torch.no_grad():
                if not it % 10:
                    model.write_updates(writer, input_, output, trainloss, global_it, "train")
            scheduler.optimizer.zero_grad()
            trainloss.backward()
            scheduler.optimizer.step()

            if not it % 10:
                print("Global_Iter {:07d}   Iter {:04d}/{}   Epoch {:03d}   train_loss {:0.4f}".format(
                    global_it, it, its_per_epoch,  epoch, trainloss)
                )
            global_it += 1

            if test_run: # Stop after 5 batches
                if it == 5:
                    break

        # Validation
        if val_loader is not None:
            try:
                input_ = next(val_loader_iter)
            except StopIteration:
                val_loader_iter = iter(val_loader)
                input_ = next(val_loader_iter) # Restart iterator
            with torch.no_grad():
                model.eval()
                valloss, output = model.get_loss(input_, device)
                model.write_updates(writer, input_, output, valloss, epoch, "val")
            print("End epoch {:03d}   val_loss: {:0.4f}".format(epoch, valloss))
            if type(scheduler).__name__ == "ReduceLROnPlateau":
                scheduler.step(valloss)
            else:
                scheduler.step()
            del input_, valloss  # Clean up
        elif type(scheduler).__name__ == "ReduceLROnPlateau":
            raise RuntimeWarning("ReduceLROnPlateau scheduler used with no validation - using last training loss.")
            scheduler.step(trainloss)
        else:
            scheduler.step()


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
        if device.type == 'cuda':
            print('Memory Usage:')
            print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
            print('Cached:   ', round(torch.cuda.memory_cached(0) / 1024 ** 3, 1), 'GB')
            torch.cuda.empty_cache()
