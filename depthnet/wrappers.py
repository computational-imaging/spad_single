import torch
import torchvision.utils as vutils
import torch.nn as nn
import numpy as np
from PIL import Image

from depthnet.model.utils import ModelWrapper
from depthnet.model.loss import delta, rmse, rel_abs_diff, rel_sqr_diff

################
# Make wrapper #
################
def make_wrapper(network, network_config, pre_active, post_active, device, **wrapper_kwargs):
    """
    All wrapper classes should expect to be loaded via this function.
    :param network: The network to wrap.
    :param network_config: The configuration for creating the network.
                           Only needed to retrieve the wrapper name.
    :param pre_active: Boolean - whether or not to activate preprocessing.
    :param post_active: Boolean - whether or not to activate postprocessing.
    :param device: The device on which the network and model should run.
    :param wrapper_kwargs: Extra keyword arguments for initializing the model
    :return: An initialized wrapper wrapping the input network.
    """
    wrapper_class = globals()[network_config["network_params"]["wrapper_name"]]
    wrapper = wrapper_class(network, pre_active=pre_active, post_active=post_active, device=device,
                            **wrapper_kwargs)
    return wrapper

class DepthNetWrapper(ModelWrapper):
    """Wrapper specific for depth estimation networks.
    """
    def __init__(self, network, pre_active, post_active, 
                 rgb_mean, rgb_var,
                 min_depth, max_depth, device, **kwargs):
        super(DepthNetWrapper, self).__init__(network, pre_active, post_active)
        """
        rgb_mean - iterable - length should be the same as nchannels of rgb
        rgb_var - iterable - length should be the same as nchannels of rgb
        min_depth - minimum depth to clip to
        max_depth - maximum depth to clip to
        """
        # Preprocessing params
        self.rgb_mean = rgb_mean
        self.rgb_var = rgb_var
        self.device = device

        # Postprocessing params
        self.min_depth = min_depth
        self.max_depth = max_depth

    def pre(self, input_):
        """Depth preprocessing
        - Make RGB zero mean unit variance.
        """

        rgb = input_["rgb"]
        mean_tensor = torch.tensor(self.rgb_mean, dtype=rgb.dtype).unsqueeze(-1).unsqueeze(-1).to(self.device)
        var_tensor = torch.tensor(self.rgb_var, dtype=rgb.dtype).unsqueeze(-1).unsqueeze(-1).to(self.device)
        rgb = rgb.to(self.device)
        rgb = rgb - mean_tensor
        rgb = rgb / var_tensor
        input_["rgb"] = rgb
        return input_

    def post(self, output):
        """Depth postprocessing
        - Clip depth to min and max values
        """
        depth = output
        if self.min_depth is not None:
            depth[depth < self.min_depth] = self.min_depth
        if self.max_depth is not None:
            depth[depth > self.max_depth] = self.max_depth
        return depth

    def write_updates(self, loss, input_, output, target, prediction, ground_truth, mask, device,
                      writer, tag, it, write_images=False, save_output=False):
        """
        Logging depth data using the tensorboardX writer.
        :param loss: The loss being used to train the model. Takes (output, target).
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
        writer.add_scalar("data/{}_d2".format(tag), delta(prediction, ground_truth, mask, 1.25 ** 2).item(), it)
        writer.add_scalar("data/{}_d3".format(tag), delta(prediction, ground_truth, mask, 1.25 ** 3).item(), it)
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
                rgb_orig = vutils.make_grid(input_["rgb_orig"] / 255, nrow=4)
            else:
                rgb_orig = vutils.make_grid(input_["rgb"] / 255, nrow=4)
            writer.add_image('image/{}_rgb_orig'.format(tag), rgb_orig, it)

            depth_truth = vutils.make_grid(ground_truth, nrow=4,
                                           normalize=True, range=(self.min_depth, self.max_depth))
            writer.add_image('image/{}_depth_truth'.format(tag), depth_truth, it)

            depth_output = vutils.make_grid(prediction, nrow=4,
                                            normalize=True, range=(self.min_depth, self.max_depth))
            writer.add_image('image/{}_depth_output'.format(tag), depth_output, it)

            depth_mask = vutils.make_grid(input_["mask"], nrow=4, normalize=False)
            writer.add_image('image/depth_mask', depth_mask, it)
        if save_output:
            vutils.save_image(output, "output.png")

class DORNWrapper(DepthNetWrapper):
    """Wrapper for depth networks using the Ordinal Regression Loss as in
    H. Fu et al., “Deep Ordinal Regression Network for Monocular Depth Estimation.”

    Wrapped model should output two lists:
    (1) a per-pixel list of log-probabilities log P_0,...,log P_k-1
    where
    P_i = P(L > i)
    where L is the bin index corresponding to the estimated depth of this pixel.
    and
    (2) a list of the log complementary probabilities, log (1 - P_i).
    """

    def __init__(self, network, pre_active, post_active,
                 rgb_mean, rgb_var,
                 min_depth, max_depth,
                 sid_bins, device,
                 **kwargs):
        """
        :param depth_bins - array of same length as the number of output probs of the network.
                            Maps the bin number to the real-world depth value.
        """
        super(DORNWrapper, self).__init__(network, pre_active, post_active,
                                          rgb_mean, rgb_var, min_depth, max_depth, device)
        self.sid_offset = 1.0 - self.min_depth
        self.sid_bins = sid_bins
        # Compute sid_depths
        # If bin i is from t_k to t_k+1, then the depth associated with that bin is
        # d = (t_k + t_k+1)/2 - offset
        # Implicitly, start = 1.0
        end = self.max_depth + self.sid_offset
        self.sid_bin_edges = np.array([np.power(end, i/self.sid_bins)
                                       for i in range(self.sid_bins+1)])
        self.sid_depths = (self.sid_bin_edges[:-1] + self.sid_bin_edges[1:])/2 - self.sid_offset
        self.sid_depths = torch.from_numpy(self.sid_depths).float().to(device)

    def pre(self, input_):
        rgb = input_["rgb"]
        mean_tensor = torch.tensor(self.rgb_mean, dtype=rgb.dtype).unsqueeze(-1).unsqueeze(-1).to(self.device)
        var_tensor = torch.tensor(self.rgb_var, dtype=rgb.dtype).unsqueeze(-1).unsqueeze(-1).to(self.device)
        rgb = rgb.to(self.device)
        rgb = rgb - mean_tensor
        rgb = rgb / var_tensor
        input_["rgb"] = rgb
        return input_

    def post(self, output):
        """Post-processing for the log-probabilities to convert them into an actual depth image.
        :param output - a tuple of (log_probs, _) where log_probs is an N x K x H x W tensor
        where each pixel location is a length K vector containing log-probabilities log P(l > 0),..., log P(l > K-1).

        The ignored input is the same, but contains the log-probabilities log (1 - P(l > 0)),..., log (1 - P(l > K-1))
        instead.
        """
        log_probs, _ = output
        MAX_BIN = log_probs.size(1) - 1
        depth_index = torch.sum((log_probs >= np.log(0.5)), dim=1).long().unsqueeze(1)
        # Clip to maximum bin index in case all the probabilities are 1.
        depth_index[depth_index > MAX_BIN] = MAX_BIN
        depth_vals = torch.take(self.sid_depths, depth_index)
        return depth_vals
    # pred = pred[0,0,:,:] - 1.0
    # pred = pred/25.0 - 0.36
    # pred = np.exp(pred)

    def write_globals(self, writer):
        """
        Outputs a (5*(nbins))x 20 pixel image corresponding to the depth values for each bin.
        Shows the discretization spacing.
        :return: Nothing.
        """
        if writer is None:
            return
        # sid_depths_np = self.sid_depths.numpy()
        bin_width = 20
        img_arr = np.zeros((bin_width*len(self.sid_depths), bin_width*len(self.sid_depths)))
        for i in range(len(self.sid_depths)):
            img_arr[:, bin_width*i:bin_width*(i+1)] = self.sid_depths[i].item()
        img_arr_norm = (img_arr - self.min_depth) / (self.max_depth - self.min_depth)
        writer.add_image("image/depth_scale", np.uint8(img_arr_norm*255), 0)

    def write_updates(self, loss, input_, output, target, prediction, ground_truth, mask, device,
                      writer, tag, it, write_images=False, save_output=False):
        """
        Logging depth data using the tensorboardX writer.
        :param loss: The loss being used to train the model. Takes (output, target).
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
        writer.add_scalar("data/{}_d2".format(tag), delta(prediction, ground_truth, mask, 1.25 ** 2).item(), it)
        writer.add_scalar("data/{}_d3".format(tag), delta(prediction, ground_truth, mask, 1.25 ** 3).item(), it)
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
                rgb_orig = vutils.make_grid(input_["rgb_orig"] / 255, nrow=4)
            else:
                rgb_orig = vutils.make_grid(input_["rgb"] / 255, nrow=4)
            writer.add_image('image/{}_rgb_orig'.format(tag), rgb_orig, it)

            if "depth_sid" in input_:
                # print(input_["depth_sid_index"].size())
                # print(self.sid_depths)
                # print(torch.max(input_["depth_sid_index"]))
                # print(torch.min(input_["depth_sid_index"]))
                depth_sid_truth = torch.take(self.sid_depths, input_["depth_sid_index"])
                depth_sid_truth = vutils.make_grid(depth_sid_truth, nrow=4,
                                             normalize=True, range=(self.min_depth, self.max_depth))
                writer.add_image('image/{}_depth_sid_truth'.format(tag), depth_sid_truth, it)


            depth_truth = vutils.make_grid(ground_truth, nrow=4,
                                           normalize=True, range=(self.min_depth, self.max_depth))
            writer.add_image('image/{}_depth_truth'.format(tag), depth_truth, it)

            depth_output = vutils.make_grid(prediction, nrow=4,
                                            normalize=True, range=(self.min_depth, self.max_depth))
            writer.add_image('image/{}_depth_output'.format(tag), depth_output, it)

            depth_mask = vutils.make_grid(input_["mask"], nrow=4, normalize=False)
            writer.add_image('image/depth_mask', depth_mask, it)
        if save_output:
            vutils.save_image(output, "output.png")

if __name__ == '__main__':
    a = {"rgb": torch.tensor([[4, 6, 8], [8, 6, 4], [4, 6, 8]], dtype=torch.float32)}
    mean = torch.tensor([6, 5, 4], dtype=torch.float32)
    var = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)
    model = lambda d: torch.sum(d["rgb"], dim=0)


    wrapper = DepthNetWrapper(model, True, True, "rgb", mean, var, 0., 8, "cpu")

    out = wrapper(a)
    print(out)
    # print(wrapper.preprocessed)
    # print(wrapper.model_output)
    # print(wrapper.postprocessed)
