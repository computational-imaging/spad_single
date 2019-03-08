import torch
import torch.nn as nn
import torch.nn.functional as F

import skimage.measure
import numpy as np

import torchvision
import os

from models.pytorch_prototyping.pytorch_prototyping import Unet
from models.core.model_core import Model


def num_divisible_by_2(number):
    return np.floor(np.log2(number)).astype(int)


def get_num_net_params(net):
    """Counts number of trainable parameters in pytorch module
    """
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params


class DenoisingUnetModel(Model):
    """A simple unet-based denoiser. This class is overly complicated for what it's accomplishing, because it
    serves as an example model class for more complicated models.

    Assumes images are scaled from -1 to 1.
    """


    def __init__(self, img_sidelength):
        super(DenoisingUnetModel, self).__init__()

        self.norm = nn.InstanceNorm2d
        self.img_sidelength = img_sidelength

        num_downs_unet = num_divisible_by_2(img_sidelength)

        self.nf0 = 64  # Number of features to use in the outermost layer of U-Net

        self.denoising_net = nn.Sequential(
            Unet(in_channels=3,
                 out_channels=3,
                 use_dropout=False,
                 nf0=self.nf0,
                 max_channels=8 * self.nf0,
                 norm=self.norm,
                 num_down=num_downs_unet,
                 outermost_linear=True),
            nn.Tanh()
        )

        # Losses
        self.loss = nn.MSELoss()

        # List of logs
        self.counter = 0  # A counter to enable logging every nth iteration
        self.logs = list()

        self.cuda()

        print("*" * 100)
        print(self)  # Prints the model
        print("*" * 100)
        print("Number of parameters: %d" % get_num_net_params(self))
        print("*" * 100)

    def get_loss(self, data, device):
        input_, ground_truth = data
        input_ = input_.to(device)
        ground_truth = ground_truth.to(device)
        output = self.forward(input_)
        loss = self.get_distortion_loss(output, ground_truth)
        return loss, output

    def get_distortion_loss(self, prediction, ground_truth):
        trgt_imgs = ground_truth.cuda()
        return self.loss(prediction, trgt_imgs)


    def write_updates(self, writer, input_, output, loss, it, tag):
        """Writes out tensorboard summaries as logged in self.logs.
        """
        rgb, ground_truth = input_
        predictions = output.cpu().detach()
        batch_size = predictions.size(0)

        if not self.logs:
            return

        for dtype, name, content in self.logs:
            if dtype == 'image':
                writer.add_image(tag + "/" + name, content.detach().cpu().numpy(), it)
                writer.add_scalar(tag + "/" + name + '_min', content.min(), it)
                writer.add_scalar(tag + "/" + name + '_max', content.max(), it)
            elif dtype == 'figure':
                writer.add_figure(tag + "/" + name, content, it, close=True)

        # Cifar10 images are tiny - to see them better in tensorboard, upsample to 256x256
        output_input_gt = torch.cat((rgb, predictions, ground_truth), dim=0)
        output_input_gt = F.interpolate(output_input_gt, scale_factor=256 / self.img_sidelength)
        grid = torchvision.utils.make_grid(output_input_gt,
                                           scale_each=True,
                                           nrow=batch_size,
                                           normalize=True).cpu().detach().numpy()
        writer.add_image(tag + "/" + "Output_vs_gt", grid, it)

        writer.add_scalar(tag + "/" + "train_loss", loss.item(), it)
        writer.add_scalar(tag + "/" + "psnr", self.get_psnr(predictions.cpu().numpy(),
                                                            ground_truth.cpu().numpy()), it)

        writer.add_scalar(tag + "/" + "out_min", predictions.min(), it)
        writer.add_scalar(tag + "/" + "out_max", predictions.max(), it)

        writer.add_scalar(tag + "/" + "trgt_min", ground_truth.min(), it)
        writer.add_scalar(tag + "/" + "trgt_max", ground_truth.max(), it)

    def write_eval(self, data, path, device):
        """At test time, this saves examples to disk in a format that allows easy inspection
        """
        input_, ground_truth = data
        input_ = input_.to(device)
        prediction = self.forward(input_)
        pred = prediction.detach().cpu().numpy()
        gt = ground_truth.detach().cpu().numpy()

        output = np.concatenate((pred, gt), axis=1).squeeze(0)
        output /= 2.
        output += 0.5
        # print(output.shape)
        np.save(path, output)

    def evaluate_dir(self, output_dir, device):
        """Calculate individual PSNRs and also the average PSNR.
        """
        print(output_dir)
        psnrs = {}
        for filename in os.listdir(output_dir):
            if not filename.endswith(".npy"):
                continue
            print(filename)
            output = np.load(os.path.join(output_dir, filename))
            # print(output)
            pred = output[:3, :, :]
            gt = output[3:, :, :]
            psnrs[filename.split(".")[0]] = self.get_psnr(pred, gt)
        avg_psnr = np.mean([psnrs[k] for k in psnrs])
        print("Average psnr: {}".format(avg_psnr))
        psnrs.update({"avg_psnr": avg_psnr})
        return psnrs

    def get_psnr(self, predictions, ground_truth):
        """Calculates the PSNR of the model's prediction."""
        return skimage.measure.compare_psnr(ground_truth, predictions, data_range=2)

    def forward(self, input_):
        noisy_img = input_
        self.logs = list()  # Resets the logs

        batch_size, _, _, _ = noisy_img.shape

        # We implement a resnet (good reasoning see https://arxiv.org/abs/1608.03981)
        pred_noise = self.denoising_net(noisy_img)
        output = noisy_img - pred_noise

        if not self.counter % 50:
            # Cifar10 images are tiny - to see them better in tensorboard, upsample to 256x256
            pred_noise = F.interpolate(pred_noise,
                                       scale_factor=256 / self.img_sidelength)
            grid = torchvision.utils.make_grid(pred_noise,
                                               scale_each=True,
                                               normalize=True,
                                               nrow=batch_size)
            self.logs.append(('image', 'pred_noise', grid))

        self.counter += 1

        return output
