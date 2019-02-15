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

def log_depth_data(loss, model, input_, output, target, prediction, ground_truth, mask, device,
                   writer, tag, it, write_images=False, save_output=False):
    """
    Logging depth data using the tensorboardX writer.
    :param loss: The loss being used to train the model. Takes (output, target).
    :param model: The (wrapped) network being trained.
    :param input_: The minibatch input from the dataloader.
    :param output: The output of the network (not post-processed).
    :param target: The target output for the network that the loss
    :param prediction: The depth prediction of the network (post-processed).
    :param ground_truth: The actual depth from the dataset.
    :param mask: An array of 1.0 and 0.0 showing which pixels should be used in calculating the metrics.
    :param device: The device to run the computation on.
    :param writer: A tensorboardX SummaryWriter object to do the writing.
    :param tag: A tag (usually either "train" or "val") for bookkeeping.
    :param it: The current iteration (either the training iteration, in the case of training, or the current epoch).
    :param write_images: Whether or not to write images to the tensorboard.
    :param save_output: Whether or not to save the output of the network as an image.
    :return: Nothing.
    """
    if writer is None:
        return
    writer.add_scalar("data/{}_d1".format(tag), delta(prediction, ground_truth, mask, 1.25).item(), it)
    writer.add_scalar("data/{}_d2".format(tag), delta(prediction, ground_truth, mask, 1.25**2).item(), it)
    writer.add_scalar("data/{}_d3".format(tag), delta(prediction, ground_truth, mask, 1.25**3).item(), it)
    writer.add_scalar("data/{}_rmse".format(tag), rmse(prediction, ground_truth, mask).item(), it)
    # print(rmse(prediction, ground_truth).item())
    log_prediction = torch.log(prediction)
    log_ground_truth = torch.log(ground_truth)
    # print("log prediction nans: {}".format(torch.isnan(log_prediction).any()))
    # print("log prediction infs: {}".format(torch.sum(log_prediction == float('-inf'))))
    # print("log target nans: {}".format(torch.isnan(log_target).any()))
    # log_target[torch.isnan(log_target)] = 0
    writer.add_scalar("data/{}_logrmse".format(tag), rmse(log_prediction, log_ground_truth, mask), it)
    writer.add_scalar("data/{}_rel_abs_diff".format(tag), rel_abs_diff(prediction, ground_truth, mask), it)
    writer.add_scalar("data/{}_rel_sqr_diff".format(tag), rel_sqr_diff(prediction, ground_truth, mask), it)
    writer.add_scalar("data/{}_loss".format(tag), loss(output, target, mask).item(), it)
    # writer.add_scalar("data/{}_ground_truth_min".format(tag), torch.min(output).item(), it)
    # writer.add_scalar("data/{}_ground_truth_max".format(tag), torch.max(output).item(), it)
    if write_images:
        if "rgb_orig" in input_:
            # print(input_["rgb_orig"].size())
            rgb_orig = vutils.make_grid(input_["rgb_orig"]/255, nrow=4)
        else:
            rgb_orig = vutils.make_grid(input_["rgb"]/255, nrow=4)
        writer.add_image('image/{}_rgb_orig'.format(tag), rgb_orig, it)

        depth_truth = vutils.make_grid(ground_truth, nrow=4,
                                       normalize=True, range=(model.min_depth, model.max_depth))
        writer.add_image('image/{}_depth_truth'.format(tag), depth_truth, it)

        depth_output = vutils.make_grid(prediction, nrow=4,
                                        normalize=True, range=(model.min_depth, model.max_depth))
        writer.add_image('image/{}_depth_output'.format(tag), depth_output, it)

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
