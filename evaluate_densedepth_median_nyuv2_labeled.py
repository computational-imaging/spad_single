#!/usr/bin/env python3
import os
import numpy as np
import torch
from utils.train_utils import init_randomness
from collections import defaultdict
import json
from models.core.checkpoint import load_checkpoint, safe_makedir
from models.data.data_utils.transforms import AddDepthMask
from utils.eval_utils import evaluate_model_on_dataset, evaluate_model_on_data_entry
from models import make_model
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Dataset
from models.data.nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data

ex = Experiment('densedepth_nyuv2_labeled', ingredients=[nyuv2_labeled_ingredient])


# Tensorboardx
# writer = SummaryWriter()

@ex.config
def cfg(data_config):
    model_config = {                            # Load pretrained model for testing
        "model_name": "DenseDepthMedianRescaling",
        "model_params": {
            "existing": os.path.join("models", "nyu.h5"),
        },
        "model_state_dict_fn": None
    }
    ckpt_file = None                            # Keep as None
    save_outputs = True
    seed = 95290421 # changing seed does not impact evaluation
    small_run = 0
    dataset_type = "test"
    entry = None

    # print(data_config.keys())
    output_dir = os.path.join("results",
                              data_config["data_name"],    # e.g. nyu_depth_v2
                              "{}_{}".format("test", small_run),
                              model_config["model_name"])  # e.g. DORN_nyu_nohints

    safe_makedir(output_dir)
    ex.observers.append(FileStorageObserver.create(os.path.join(output_dir, "runs")))

    cuda_device = "0"                       # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    # print("after: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    if ckpt_file is not None:
        model_update, _, _ = load_checkpoint(ckpt_file)
        model_config.update(model_update)

        del model_update, _  # So sacred doesn't collect them.


@ex.automain
def main(model_config,
         save_outputs,
         output_dir,
         data_config,
         seed,
         small_run,
         dataset_type,
         entry):
    # Load the model
    model = make_model(**model_config)
    # model.sid_obj.to(device)

    from tensorboardX import SummaryWriter
    from datetime import datetime
    model.writer = SummaryWriter(log_dir=os.path.join("runs",
                                                      datetime.now().strftime('%b%d'),
                                                      datetime.now().strftime('%H-%M-%S_') + \
                                                      "densedepth_nohints"))

    # Load the data
    train, test = load_data(dorn_mode=False)
    dataset = train if dataset_type == "train" else test
    eval_fn = lambda input_, device: model.evaluate(input_["rgb"],
                                                    input_["crop"][0,:],
                                                    input_["depth_cropped"],
                                                    input_["rawdepth_cropped"],
                                                    input_["mask_cropped"],
                                                    torch.ones_like(input_["depth_cropped"]))

    init_randomness(seed)

    if entry is None:
        print("Evaluating the model on {}.".format(data_config["data_name"]))
        evaluate_model_on_dataset(eval_fn, dataset, small_run, None, save_outputs, output_dir)
    else:
        print("Evaluating {}".format(entry))
        evaluate_model_on_data_entry(eval_fn, dataset, entry, None, save_outputs, output_dir)
