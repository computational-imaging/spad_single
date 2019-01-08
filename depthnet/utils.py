import os.path
import sys
from datetime import datetime
import socket

import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torchvision.transforms as transforms

from depthnet.model import delta, rmse, rel_abs_diff, rel_sqr_diff

NYU_MIN = 1e-3
NYU_MAX = 8.0

def clip_min_max(depth, min_depth, max_depth):
    depth[depth < min_depth] = min_depth
    depth[depth > max_depth] = max_depth
    return depth

###########
# Logging #
###########

def log_depth_data(loss, model, input_, output, target, mask, device,
                   writer, tag, it, write_images=False, save_output=False):
    writer.add_scalar("data/{}_d1".format(tag), delta(output, target, mask, 1.25).item(), it)
    writer.add_scalar("data/{}_d2".format(tag), delta(output, target, mask, 1.25**2).item(), it)
    writer.add_scalar("data/{}_d3".format(tag), delta(output, target, mask, 1.25**3).item(), it)
    writer.add_scalar("data/{}_rmse".format(tag), rmse(output, target, mask).item(), it)
    # print(rmse(output, depth).item())
    log_output = torch.log(output)
    log_target = torch.log(target)
    # print("log output nans: {}".format(torch.isnan(log_output).any()))
    # print("log output infs: {}".format(torch.sum(log_output == float('-inf'))))
    # print("log target nans: {}".format(torch.isnan(log_target).any()))
    # log_target[torch.isnan(log_target)] = 0
    writer.add_scalar("data/{}_logrmse".format(tag), rmse(log_output, log_target, mask), it)
    writer.add_scalar("data/{}_rel_abs_diff".format(tag), rel_abs_diff(output, target, mask), it)
    writer.add_scalar("data/{}_rel_sqr_diff".format(tag), rel_sqr_diff(output, target, mask), it)
    writer.add_scalar("data/{}_loss".format(tag), loss(output, target, mask).item(), it)
    # writer.add_scalar("data/{}_depth_min".format(tag), torch.min(output).item(), it)
    # writer.add_scalar("data/{}_depth_max".format(tag), torch.max(output).item(), it)
    if write_images:
        rgb_input = vutils.make_grid(input_["rgb"], nrow=4)
        writer.add_image('image/rgb_input', rgb_input, it)

        depth_truth = vutils.make_grid(target, nrow=4,
                                       normalize=True, range=(model.min_depth, model.max_depth))
        writer.add_image('image/depth_truth', depth_truth, it)

        depth_output = vutils.make_grid(output, nrow=4,
                                        normalize=True, range=(model.min_depth, model.max_depth))
        writer.add_image('image/depth_output', depth_output, it)

        # depth_mask = vutils.make_grid(input_["mask"], nrow=2, normalize=False, scale_each=True)
        # writer.add_image('image/depth_mask', depth_mask, it)
    if save_output:
        vutils.save_image(output, "output.png")

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
