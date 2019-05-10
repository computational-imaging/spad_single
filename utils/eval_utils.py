import torch
import os
import json
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader
from utils.train_utils import worker_init_randomness
from models.core.checkpoint import safe_makedir

def evaluate_model_on_dataset(model, dataset, small_run, device,
                              save_outputs, output_dir=None):
    """
    Evaluate a depth estimation model on a dataset.
    :param model: The model to use. If it needs to be in eval() mode, you need to have called that already.
    :param dataset: The dataset to use
    :param small_run: Whether or not to stop early
    :param device: The device to run the model on
    :param save_outputs: Whether or not to save the outputs of the model as torch .pt files
    :param output_dir: The directory to save the results to
    :return: None
    """
    # Make dataloader
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            shuffle=True,
                            num_workers=0, # needs to be 0 to not crash autograd profiler.
                            pin_memory=True,
                            worker_init_fn=worker_init_randomness)
    # if eval_config["save_outputs"]:

    with torch.no_grad():
        # model.eval()
        num_pixels = 0.
        avg_metrics = defaultdict(float)
        metrics = defaultdict(dict)
        for i, data in enumerate(dataloader):
            # TESTING
            if small_run and i == small_run:
                break
            entry = data["entry"][0]
            print("Evaluating {}".format(data["entry"][0]))
            pred, pred_metrics = model.evaluate(data, device)
            metrics[entry] = pred_metrics
            num_valid_pixels = torch.sum(data["mask_orig"]).item()
            num_pixels += num_valid_pixels
            for metric_name in pred_metrics:
                avg_metrics[metric_name] += num_valid_pixels * pred_metrics[metric_name]
            print(pred_metrics)
            # Option to save outputs:
            if save_outputs:
                if output_dir is None:
                    raise ValueError("evaluate_model_on_dataset: output_dir is None")
                save_dict = {
                    "entry": entry,
                    "pred": pred
                }
                path = os.path.join(output_dir, "{}_out.pt".format(entry))
                safe_makedir(os.path.dirname(path))
                torch.save(save_dict, path)


        for metric_name in avg_metrics:
            avg_metrics[metric_name] /= num_pixels
        with open(os.path.join(output_dir, "avg_metrics.json"), "w") as f:
            json.dump(avg_metrics, f)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        print(avg_metrics)
    print("wrote results to {}".format(output_dir))


def collect_results_files(**filepaths):
    """
    Collects statistics
    :param filepaths: Dictionary mapping model names to file paths
    :return: MultiIndex dataframe with major axis [[models],[metrics]]
    """
    model_results = dict()
    for model_name in filepaths:
        with open(filepaths[model_name], 'r') as f:
            model_results[model_name] = pd.DataFrame.from_dict(json.load(f))
    return pd.concat(model_results.values(), keys=model_results.keys())

if __name__ == "__main__":
    filepaths = {
        "DORN_nyu_nohints": "results/nyu_depth_v2/DORN_nyu_nohints/test/metrics.json",
        "DORN_nyu_hints": "results/nyu_depth_v2/DORN_nyu_hints/test/metrics.json",
        "DORN_nyu_histogram_matching/rawhist": "results/nyu_depth_v2/DORN_nyu_histogram_matching/rawhist/test/metrics.json",
    }
    model_results = collect_results_files(**filepaths)
    print(model_results)

