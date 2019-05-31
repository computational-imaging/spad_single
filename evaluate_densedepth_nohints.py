#!/usr/bin/env python3
import os
import numpy as np
from utils.train_utils import init_randomness
from collections import defaultdict
import json
from models.core.checkpoint import load_checkpoint, safe_makedir
from models.data.utils.transforms import AddDepthMask
from models import make_model
from sacred import Experiment
from sacred.observers import FileStorageObserver

# Dataset
from models.data.nyuv2_test_split_dataset import nyuv2_test_split_ingredient, load_data

ex = Experiment('eval_nohints_sid', ingredients=[nyuv2_test_split_ingredient])


# Tensorboardx
# writer = SummaryWriter()

@ex.config
def cfg(data_config):
    model_config = {                            # Load pretrained model for testing
        "model_name": "DenseDepth",
        "model_params": {
            "min_depth": data_config["min_depth"],
            "max_depth": data_config["max_depth"],
            "existing": os.path.join("models", "nyu.h5"),
        },
        "model_state_dict_fn": None
    }
    ckpt_file = None                            # Keep as None
    save_outputs = True
    seed = 95290421
    small_run = 0

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
         small_run):
    # Load the model
    model = make_model(**model_config)
    # model.sid_obj.to(device)

    # Load the data
    dataset = load_data()

    init_randomness(seed)

    print("Evaluating the model on {}.".format(data_config["data_name"]))
    # evaluate_model_on_dataset(model, dataset, small_run, device, save_outputs, output_dir)
    add_mask = AddDepthMask(data_config["min_depth"], data_config["max_depth"], "depth_cropped")
    all_metrics = defaultdict(dict)
    avg_metrics = defaultdict(float)
    total_pixels = 0.
    for i in range(len(dataset)):
        print("Evaluating {}".format(i))
        input_ = dataset[i]
        # Get valid pixels
        input_ = add_mask(input_)

        pred, metrics = model.evaluate(input_)
        all_metrics[i] = metrics

        total_pixels += np.sum(input_["mask"])
        for metric_name in metrics:
            avg_metrics[metric_name] += np.sum(input_["mask"]) * metrics[metric_name]
        ## Save stuff ##
        if save_outputs:
            np.save(os.path.join(output_dir, "{:04d}.npy".format(i)), pred)
        ## Done saving stuff ##
        print({metric_name: avg_metrics[metric_name]/total_pixels for metric_name in avg_metrics})

    for metric_name in avg_metrics:
        avg_metrics[metric_name] /= total_pixels
    with open(os.path.join(output_dir, "avg_metrics.json"), "w") as f:
        json.dump(avg_metrics, f)
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(all_metrics, f)

    print(avg_metrics)
    print("wrote results to {}".format(output_dir))





