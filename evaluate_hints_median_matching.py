#!/usr/bin/env python3
import os
import torch
import json
from torch.utils.data import DataLoader
from utils.train_utils import init_randomness, worker_init_randomness
from utils.eval_utils import evaluate_model_on_dataset
from models.core.checkpoint import load_checkpoint, safe_makedir
from models import make_model
from sacred import Experiment
from time import perf_counter

# Dataset
from models.data.nyuv2_official_hints_sid_dataset import nyuv2_hints_sid_ingredient, load_data
from models.data.utils.spad_utils import spad_ingredient

ex = Experiment('eval_hints_histogram_matching', ingredients=[nyuv2_hints_sid_ingredient, spad_ingredient])


# Tensorboardx
# writer = SummaryWriter()

@ex.config
def cfg(data_config, spad_config):
    model_config = {                            # Load pretrained model for testing
        "model_name": "DORN_nyu_histogram_matching",
        "model_params": {
            "hints_len": 68,
            "in_channels": 3,
            "in_height": 257,
            "in_width": 353,
            "sid_bins": data_config["sid_bins"],
            "offset": data_config["offset"],
            "min_depth": data_config["min_depth"],
            "max_depth": data_config["max_depth"],
            "alpha": data_config["alpha"],
            "beta": data_config["beta"],
            "frozen": True,
            "pretrained": True,
            "state_dict_file": os.path.join("models", "torch_params_nyuv2_BGR.pth.tar"),
        },
        "model_state_dict_fn": None
    }
    # ckpt_file = "checkpoints/Mar15/04-10-54_DORN_nyu_hints_nyu_depth_v2/checkpoint_epoch_9_name_fixed.pth.tar"
    ckpt_file = None # Bayesian hints eval
    dataset_type = "val"
    save_outputs = True
    comment = ""
    # print(data_config.keys())
    fullcomment = comment + spad_config["spad_comment"]
    output_dir = os.path.join("results",
                              data_config["data_name"],    # e.g. nyu_depth_v2
                              model_config["model_name"],  # e.g. DORN_nyu_nohints
                              dataset_type)
    if fullcomment is not "":
        output_dir = os.path.join("results",
                                  data_config["data_name"],    # e.g. nyu_depth_v2
                                  model_config["model_name"],  # e.g. DORN_nyu_nohints
                                  fullcomment,
                                  dataset_type)
    seed = 95290421
    small_run = False

    cuda_device = "0"                       # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    # print("after: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))
    if ckpt_file is not None:
        model_update, _, _ = load_checkpoint(ckpt_file)
        model_config.update(model_update)

        del model_update, _  # So sacred doesn't collect them.

@ex.automain
def main(model_config,
         dataset_type,
         save_outputs,
         output_dir,
         data_config,
         seed,
         small_run,
         device):
    # Load the model
    model = make_model(**model_config)
    model.to(device)
    model.eval()
    # model.sid_obj.to(device)
    print(model)

    # Load the data
    _, val, test = load_data()
    dataset = test if dataset_type == "test" else val

    init_randomness(seed)

    print("Evaluating the model on {} ({})".format(data_config["data_name"],
                                                   dataset_type))
    evaluate_model_on_dataset(model, dataset, small_run, device, save_outputs, output_dir)

