import os

import torch
import torch.optim as optim
import torch.nn as nn
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

from depthnet.model import DepthNet, DepthNetWithHints, UNet
import depthnet.utils as u

####################
# Run the training #
####################
def train(setup,
          metadata,
          train_loader,
          val_loader,
          device,
          test_run=False,
          writer=None):
    """
    setup = {
        "model": model,
        "loss": loss,
        "scheduler": scheduler,
        "start_epoch": start_epoch,
        "num_epochs": trd["num_epochs"],
        "global_it": trd["global_it"],
        "trainlosses": trd["trainlosses"],
        "vallosses": trd["vallosses"]
    }
    metadata = {
        "model_params": trd["model_params"],
        "loss_fn": trd["loss_fn"],
        "optim_params": trd["optim_params"],
        "scheduler_params": trd["scheduler_params"]
    }
    """
    # unpack
    model = setup["model"]
    loss = setup["loss"]
    scheduler = setup["scheduler"]
    start_epoch = setup["start_epoch"]
    num_epochs = setup["num_epochs"]
    global_it = setup["global_it"]
    trainlosses = setup["trainlosses"]
    vallosses = setup["vallosses"]
    checkpoint_dir = setup["checkpoint_dir"]

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
            trainlosses.append(trainloss.item())
            if writer is not None:
                writer.add_scalar("data/trainloss", trainloss.item(), global_it)
            if test_run: # Stop after 5 batches
                if not (it + 1) % 5:
                    break
        # Checkpointing
        if val_loader is not None:
            valloss = u.validate(loss, model, val_loader)
            print("End epoch {}\tval loss: {}".format(epoch, valloss))
            vallosses.append(valloss)
            if writer is not None:
                writer.add_scalar("data/valloss", valloss, epoch)

        # Save the last batch output of every epoch
        if writer is not None:
            rgb_input = vutils.make_grid(data["rgb"], nrow=2, normalize=True, scale_each=True)
            writer.add_image('image/rgb_input', rgb_input, epoch)

            depth_truth = vutils.make_grid(data["depth"], nrow=2, normalize=True, scale_each=True)
            writer.add_image('image/depth_truth', depth_truth, epoch)

            depth_output = vutils.make_grid(output, nrow=2, normalize=True, scale_each=True)
            writer.add_image('image/depth_output', depth_output, epoch)

            for name, param in model.named_parameters():
                writer.add_histogram(name, param.clone().cpu().data.numpy(), global_it)

        # Save checkpoint
        state = {
            "epoch": epoch,
            "global_it": global_it,
            "trainlosses": trainlosses,
            "vallosses": vallosses,
        }
        u.save_checkpoint(model,
                          scheduler,
                          metadata,
                          state,
                          filename=os.path.join(checkpoint_dir,
                                                "checkpoint_epoch_{}.pth.tar".format(epoch)))
