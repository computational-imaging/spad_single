import torch
import torch.nn as nn


class Model(nn.Module):
    """
    A class representing a wrapper around a neural net that includes extra features
    for data preprocessing, postprocessing, and data loading for training and testing.
    """
    def __init__(self):
        super(Model, self).__init__()

    def get_loss(self, input_, device):
        """
        :param input_: Data from the dataloader.
        :return: A tuple of (trainloss, output) where |trainloss| is the loss incurred on this data point
        and |output| is the output from the network.
        """
        raise NotImplementedError

    def write_updates(self, writer, input_, output, loss, it, tag):
        """
        Write stuff to tensorboard every iteration.
        :param writer: The writer to use.
        :param it: The current iteration.
        :param input_: The input data to the network at this iteration.
        :param output: The output of the network at this iteration.
        :param loss: The value of the loss (training or validiation)
        :param tag: Extra info about the logging.
        :return: None
        """
        raise NotImplementedError

    def write_eval(self, data, output_file, device):
        """
        Write model outputs to disk for later evaluation.
        :param data: A data entry to evaluate
        :param output_file: The output file for this input.
        :param device: The device to run on
        :return: None
        """
        raise NotImplementedError

    def evaluate_dir(self, output_dir, device):
        """
        Load model outputs from directory and calculate figures of merit
        :param output_dir: The directory containing the model outputs.
        :param device: The device to run things on.
        :return: A dictionary of all the things to save.
        """
        raise NotImplementedError


