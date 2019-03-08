#!/usr/bin/env python3
import os
import torch
import json
from torch.utils.data import DataLoader
from utils.train_utils import init_randomness, worker_init_randomness
from models.core.checkpoint import load_checkpoint, safe_makedir
from models import make_model
from sacred import Experiment

# Dataset
from models.data.noisy_cifar10_dataset import noisy_cifar10_ingredient, load_data

ex = Experiment('eval_cifar10', ingredients=[noisy_cifar10_ingredient])


# Tensorboardx
# writer = SummaryWriter()

@ex.config
def cfg(data_config):
    model_config = {}                           # To be loaded from the checkpoint file.
    ckpt_file = "checkpoints/Mar07/02-18-30_DenoisingUnetModel_cifar10/checkpoint_epoch_0.pth.tar"
    eval_config = {
        "dataset": "val",                       # {val, test}
        "mode": "save_outputs",                 # {save_outputs, evaluate_metrics}
        "output_dir": "cifar10_eval"
    }
    seed = 95290421

    cuda_device = "0"                       # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    # print("after: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))
    if ckpt_file is not None:
        model_update, _, _ = load_checkpoint(ckpt_file)
        model_config.update(model_update)

        del model_update, _ # So sacred doesn't collect them.


@ex.automain
def main(model_config,
         eval_config,
         data_config,
         seed,
         device):
    # Load the model
    model = make_model(**model_config)
    print(model)

    # Load the data
    _, val, test = load_data()
    dataset = test if eval_config["dataset"] == "test" else val

    init_randomness(seed)

    # Make dataloader
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=2,
                            pin_memory=True,
                            worker_init_fn=worker_init_randomness)
    if eval_config["mode"] == "save_outputs":
        # Run the model on everything and save everything to disk.
        safe_makedir(eval_config["output_dir"])
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                print("Evaluating {}".format(i))
                model.write_eval(data,
                                 os.path.join(eval_config["output_dir"],
                                              "{}_out.npy".format(i)),
                                 device)

    elif eval_config["mode"] == "evaluate_metrics":
        # Load things and call the model's evaluate function on them.
        metrics = model.evaluate_dir(eval_config["output_dir"], device)
        with open(os.path.join(eval_config["output_dir"], "metrics.json"), "w") as f:
            json.dump(metrics, f)
    else:
        print("Unrecognized mode: {}".format(eval_config["mode"]))
