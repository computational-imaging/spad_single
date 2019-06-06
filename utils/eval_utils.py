import torch
import os
import json
import pandas as pd
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from utils.train_utils import worker_init_randomness
from models.core.checkpoint import safe_makedir


class ResultsManager:
    """
    Uses json
    """
    def __init__(self, rootdir=".", entry_list=None, metrics_dict=None, avg_metrics_dict=None):
        self.rootdir = rootdir
        self.entry_list = [] if entry_list is None else entry_list
        self.metrics_dict = {} if metrics_dict is None else metrics_dict
        self.avg_metrics_dict = {} if avg_metrics_dict is None else avg_metrics_dict

    @classmethod
    def load_from_dir(cls, rootdir):
        rootdir = rootdir
        entry_list = cls.load_json_or_none(os.path.join(rootdir, "entries.json"))
        metrics_dict = cls.load_json_or_none(os.path.join(rootdir, "metrics.json"))
        avg_metrics_dict = cls.load_json_or_none(os.path.join(rootdir, "avg_metrics.json"))
        return cls(rootdir, entry_list, metrics_dict, avg_metrics_dict)

    def record_entry(self, entry):
        self.entry_list.append(entry)


    def record_metric(self, entry, metric):
        pass

    def aggregate_metrics(self):
        pass

    def save_to_dir(self, dir):
        pass

    @staticmethod
    def load_json_or_none(filepath):
        """

        :param filepath: Path to a json file
        :return: the data contained in the json, or None if the file does not exist
        """
        try:
             with open(filepath, "r") as f:
                 output = json.load(f)
        except IOError:
            output = None
        return output



def evaluate_model_on_dataset(eval_fn, dataset, small_run, device,
                              save_outputs, output_dir=None):
    """
    Evaluate a depth estimation model on a dataset.
    Aggregate the metrics to get versions of them that are averaged over the entire dataset.
    :param eval_fn: f(input_, device) A function that, when called on an entry from the dataloader,
                    returns the result of running the model on that data entry.
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
        total_num_pixels = 0.
        avg_metrics = defaultdict(float)
        metrics = defaultdict(dict)
        for i, data in enumerate(dataloader):
            # TESTING
            if small_run and i == small_run:
                break
            entry = data["entry"][0]
            print("Evaluating {}".format(data["entry"][0]))
            # pred, pred_metrics = model.evaluate(data, device)
            pred, pred_metrics = eval_fn(data, device)
            metrics[entry] = pred_metrics

            num_valid_pixels = torch.sum(data["mask_orig"]).item()
            alpha = total_num_pixels / (total_num_pixels + num_valid_pixels)
            for metric_name in pred_metrics:
                avg_metrics[metric_name] = alpha * avg_metrics[metric_name] + (1 - alpha) * pred_metrics[metric_name]
            total_num_pixels += num_valid_pixels
            print(avg_metrics)
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

        with open(os.path.join(output_dir, "avg_metrics.json"), "w") as f:
            json.dump(avg_metrics, f)
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        print(avg_metrics)
    print("wrote results to {}".format(output_dir))


def evaluate_model_on_data_entry(eval_fn, dataset, entry, device, save_outputs, output_dir):
    """
    Workaround for when we only to evaluate on a single entry.
    :param model:
    :param dataset:
    :param entry_id:
    :param device:
    :return:
    """
    input_unbatched = dataset.get_item_by_id(entry)
    # for key in ["rgb", "albedo", "rawdepth", "spad", "mask", "rawdepth_orig", "mask_orig", "albedo_orig"]:
    #     input_[key] = input_[key].unsqueeze(0)
    input_ = default_collate([input_unbatched])

    # Checks
    print(input_["entry"])
    # print("remove_dc: ", model.remove_dc)
    # print("use_intensity: ", model.use_intensity)
    # print("use_squared_falloff: ", model.use_squared_falloff)
    pred, pred_metrics = eval_fn(input_, device)
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
    print(pred_metrics)


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

