import os.path
import sys

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import depthnet.model as m

#################
# Checkpointing #
#################
def save_checkpoint(model,
                    scheduler,
                    metadata,
                    state,
                    filename='/output/checkpoint.pth.tar'):
    """Save checkpoint using
    model - the current model.
    scheduler - the scheduler (and optimizer),
    metadata - extra info about the setup of the problem,
    state - the current state of the training process.
    """
    checkpoint = {
        "model_name": model.__class__.__name__,
        "model_params": metadata["model_params"],
        "model_state_dict": model.state_dict(),
        "loss_fn": metadata["loss_fn"],
        "optim_name": scheduler.optimizer.__class__.__name__,
        "optim_params": metadata["optim_params"],
        "optim_state_dict": scheduler.optimizer.state_dict(),
        "scheduler_name": scheduler.__class__.__name__,
        "scheduler_params": metadata["scheduler_params"],
        "epoch": state["epoch"],
        "global_it": state["global_it"],
        "trainlosses": state["trainlosses"],
        "vallosses": state["vallosses"],
        "log_dir": metadata["log_dir"]
    }
    print("=> Saving checkpoint to: {}".format(filename))
    torch.save(checkpoint, filename)  # save checkpoint


def load_checkpoint(checkpointfile, device):
    """Loads a checkpoint from a checkpointfile.
    Checkpoint is a dict consisting of:

    model_name
    model_params
    model_state_dict

    loss_fn (string)

    optim_name
    optim_params
    optim_state_dict

    scheduler_name
    scheduler_params

    last_epoch
    global_it

    trainlosses
    vallosses

    --
    Can derive model_name and model_state_dict from model
    Can derive scheduler_name from scheduler
       Can derive optim_name and optim_state_dict from scheduler.optimizer

    """
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpointfile)
    else:
        # Load GPU model on CPU
        checkpoint = torch.load(checkpointfile,
                                map_location=lambda storage,
                                                    loc: storage)
    return make_training(checkpoint, device)

def make_training(train_dict, device):
    """
    Create a training instance, consisting of a (setup, metadata) tuple.

    In order to create a training instance, specify the following:
    model_name - string - the name of the model
    model_params - dict - keyword args for initializing the model
    model_state_dict - dict - State dict if initializing from checkpoint,
                              None otherwise
    loss_fn - string - the name of the loss function to use.
    optim_name - The name of the class of the optimizer.
    optim_params - dict - keyword args for initializing the optimizer,
    optim_state_dict - State dict if initializing from checkpoint,
                       None otherwise
    scheduler_name - string - the name of the class of the scheduler.
    scheduler_params - Keyword args for initializing the scheduler.
    last_epoch - int - The last epoch trained, or -1 if this is a new training instance.
    num_epochs - int - the number of (further) epochs to train the model.
    global_it - int - the global iteration. 0 for a new training instance.
    trainlosses - list - list of training losses up to now.
    vallosses - list - list of validation losses up to now.
    """
    # unpack
    trd = train_dict
    # model
    model_class = getattr(m, trd["model_name"])
    model = model_class(**trd["model_params"])
    if "model_state_dict" in trd:
        model.load_state_dict(trd["model_state_dict"])
    else: # New model - apply initialization
        # m.initialize(model)
        pass # Use default pytorch initialization
    model.to(device)
    # loss_fn (string)
    loss = m.get_loss(trd["loss_fn"])

    # optimizer
    opt_class = getattr(optim, trd["optim_name"])
    split_params = m.split_params_weight_bias(model)
    optimizer = opt_class(params=split_params, **trd["optim_params"])
    if "optim_state_dict" in trd:
        optimizer.load_state_dict(trd["optim_state_dict"])

    # start_epoch
    if "last_epoch" in trd:
        start_epoch = trd["last_epoch"] + 1
    else:
        start_epoch = 0

    # scheduler
    scheduler_class = getattr(optim.lr_scheduler, trd["scheduler_name"])
    scheduler = scheduler_class(optimizer,
                                last_epoch=start_epoch - 1,
                                **trd["scheduler_params"])

    if "global_it" in trd:
        global_it = trd["global_it"]
    else:
        global_it = 0
    if "trainlosses" in trd:
        trainlosses = trd["trainlosses"]
    else:
        trainlosses = []
    if "vallosses" in trd:
        vallosses = trd["vallosses"]
    else:
        vallosses = []
    setup = {
        "model": model,
        "loss": loss,
        "scheduler": scheduler,
        "start_epoch": start_epoch,
        "global_it": global_it,
        "trainlosses": trainlosses,
        "vallosses": vallosses
    }
    metadata = {
        "model_params": trd["model_params"],
        "loss_fn": trd["loss_fn"],
        "optim_params": trd["optim_params"],
        "scheduler_params": trd["scheduler_params"],
        "log_dir": trd["log_dir"]
    }
    return setup, metadata

##############
# Validation #
##############
def validate(loss, model, val_loader):
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
    return loss(output, depth).item()

##################
# Viewing Images #
##################
def save_images(*batches, output_dir, filename):
    """
    Given a list of tensors of size (B, C, H, W) (batch, channels, height, width) torch.Tensor
    Saves each entry of the batch as an rgb or grayscale image, depending on how many channels
    the image has.
    """
    I = None
    trans = transforms.ToPILImage()
    for batchnum, batch in enumerate(batches):
        if batch.shape[1] == 3:
            pass
        elif batch.shape[1] == 1:
            batch /= torch.max(batch) # normalize to lie in [0, 1]
        else:
            raise ValueError("Unsupported number of channels: {}".format(batch.shape[1]))
        batch = batch.type(torch.float32)
        for img in range(batch.shape[0]):
            I = trans(batch[img, :, :, :].cpu().detach())
            I.save(os.path.join(output_dir, filename + "_{}_{}.png".format(batchnum, img)))


############
# Plotting #
############
def save_train_val_loss_plots(trainlosses, vallosses, epoch):
    # Train loss
    fig = plt.figure()
    plt.plot(trainlosses)
    plt.title("Train loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.savefig("trainloss_epoch{}.png".format(epoch))
    # Train loss
    fig = plt.figure()
    plt.plot(trainlosses)
    plt.title("Val loss")
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.savefig("Val loss{}.png".format(epoch))
