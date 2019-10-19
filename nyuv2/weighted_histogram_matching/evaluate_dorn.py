import os
import numpy as np
import torch
import pandas as pd
from sacred import Experiment
from weighted_histogram_matching import image_histogram_match, image_histogram_match_variable_bin
from models.data.data_utils.sid_utils import SID
from spad_utils import rescale_bins
from remove_dc_from_spad import remove_dc_from_spad_edge
from nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data

from models.loss import get_depth_metrics

ex = Experiment("dorn_weighted_hist_match", ingredients=[nyuv2_labeled_ingredient])

@ex.config
def cfg(data_config):
    data_dir = "data"
    dataset_type = "test"
    use_intensity = True
    use_squared_falloff = True
    lambertian = True
    dc_count = 1e5
    use_jitter = True
    use_poisson = True
    hyper_string = "{}_int_{}_fall_{}_lamb_{}_dc_{}_jit_{}_poiss_{}".format(
        dataset_type,
        use_intensity,
        use_squared_falloff,
        lambertian,
        dc_count,
        use_jitter,
        use_poisson)
    spad_file = os.path.join(data_dir, "{}_spad.npy".format(hyper_string))
    dorn_depth_file = os.path.join(data_dir, "dorn_{}_outputs.npy".format(dataset_type))

    # SID params
    sid_bins = 68
    bin_edges = np.array(range(sid_bins + 1)).astype(np.float32)
    dorn_decode = np.exp((bin_edges - 1) / 25 - 0.36)
    d0 = dorn_decode[0]
    d1 = dorn_decode[1]
    # Algebra stuff to make the depth bins work out exactly like in the
    # original DORN code.
    alpha = (2 * d0 ** 2) / (d1 + d0)
    beta = alpha * np.exp(sid_bins * np.log(2 * d0 / alpha - 1))
    del bin_edges, dorn_decode, d0, d1
    offset = 0.

    # SPAD Denoising params
    lam = 3e2
    eps_rel = 1e-5

    entry = None
    save_outputs = True
    small_run = 0
    output_dir = "results"


@ex.automain
def run(dataset_type,
        spad_file,
        dorn_depth_file,
        hyper_string,
        sid_bins, alpha, beta, offset, lam, eps_rel,
        entry, save_outputs, small_run, output_dir):
    # Load all the data:
    print("Loading SPAD data from {}".format(spad_file))
    spad_dict = np.load(spad_file, allow_pickle=True).item()
    spad_data = spad_dict["spad"]
    intensity_data = spad_dict["intensity"]
    spad_config = spad_dict["config"]
    print("Loading depth data from {}".format(dorn_depth_file))
    depth_data = np.load(dorn_depth_file)
    dataset = load_data(channels_first=True, dataset_type=dataset_type)

    # Read SPAD config and determine proper course of action
    dc_count = spad_config["dc_count"]
    ambient = spad_config["dc_count"]/spad_config["spad_bins"]
    use_intensity = spad_config["use_intensity"]
    use_squared_falloff = spad_config["use_squared_falloff"]
    lambertian = spad_config["lambertian"]
    min_depth = spad_config["min_depth"]
    max_depth = spad_config["max_depth"]

    print("dc_count: ", dc_count)
    print("use_intensity: ", use_intensity)
    print("use_squared_falloff:", use_squared_falloff)

    print("spad_data.shape", spad_data.shape)
    print("depth_data.shape", depth_data.shape)
    print("intensity_data.shape", intensity_data.shape)

    sid_obj_init = SID(sid_bins, alpha, beta, offset)

    if entry is None:
        metric_list = ["delta1", "delta2", "delta3", "rel_abs_diff", "rmse", "mse", "log10", "weight"]
        metrics = np.zeros((len(dataset) if not small_run else small_run, len(metric_list)))
        entry_list = []
        outputs = []
        for i in range(depth_data.shape[0]):
            if small_run and i == small_run:
                break
            entry_list.append(i)

            print("Evaluating {}[{}]".format(dataset_type, i))
            spad = spad_data[i,...]
            weights = np.ones_like(depth_data[i, 0, ...])
            if use_intensity:
                weights = intensity_data[i, 0, ...]
            if dc_count > 0.:
                spad = remove_dc_from_spad_edge(spad,
                                                ambient=ambient,
                                                grad_th=5*np.sqrt(2*ambient))
            bin_edges = np.linspace(min_depth, max_depth, len(spad) + 1)
            bin_values = (bin_edges[1:] + bin_edges[:-1]) / 2
            if use_squared_falloff:
                if lambertian:
                    spad = spad * bin_values ** 4
                else:
                    spad = spad * bin_values ** 2
            # Scale SID object to maximize bin utilization
            nonzeros = np.nonzero(spad)[0]
            if nonzeros.size > 0:
                min_depth_bin = np.min(nonzeros)
                max_depth_bin = np.max(nonzeros) + 1
                if max_depth_bin > len(bin_edges) - 2:
                    max_depth_bin = len(bin_edges) - 2
            else:
                min_depth_bin = 0
                max_depth_bin = len(bin_edges) - 2
            min_depth_pred = np.clip(bin_edges[min_depth_bin], a_min=1e-2, a_max=None)
            max_depth_pred = np.clip(bin_edges[max_depth_bin+1], a_min=1e-2, a_max=None)
            sid_obj_pred = SID(sid_bins=sid_obj_init.sid_bins,
                               alpha=min_depth_pred,
                               beta=max_depth_pred,
                               offset=0.)
            spad_rescaled = rescale_bins(spad[min_depth_bin:max_depth_bin+1],
                                         min_depth_pred, max_depth_pred, sid_obj_pred)
            pred, t = image_histogram_match_variable_bin(depth_data[i, 0, ...], spad_rescaled, weights,
                                                         sid_obj_init, sid_obj_pred)
            # break
            # break
            # Calculate metrics
            gt = dataset[i]["depth_cropped"].unsqueeze(0)
            # print(gt.dtype)
            # print(pred.shape)

            pred_metrics = get_depth_metrics(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(),
                                             gt,
                                             torch.ones_like(gt))

            for j, metric_name in enumerate(metric_list[:-1]):
                metrics[i, j] = pred_metrics[metric_name]

            metrics[i, -1] = np.size(pred)
            # Option to save outputs:
            if save_outputs:
                outputs.append(pred)
            print("\tAvg RMSE = {}".format(np.mean(metrics[:i + 1, metric_list.index("rmse")])))

        if save_outputs:
            np.save(os.path.join(output_dir, "dorn_{}_outputs.npy".format(hyper_string)), np.array(outputs))

        # Save metrics using pandas
        metrics_df = pd.DataFrame(data=metrics, index=entry_list, columns=metric_list)
        metrics_df.to_pickle(path=os.path.join(output_dir, "dorn_{}_metrics.pkl".format(hyper_string)))
        # Compute weighted averages:
        average_metrics = np.average(metrics_df.ix[:, :-1], weights=metrics_df.weight, axis=0)
        average_df = pd.Series(data=average_metrics, index=metric_list[:-1])
        average_df.to_csv(os.path.join(output_dir, "dorn_{}_avg_metrics.csv".format(hyper_string)), header=True)
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
        print(
            "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(average_metrics[0],
                                                                                average_metrics[1],
                                                                                average_metrics[2],
                                                                                average_metrics[3],
                                                                                average_metrics[4],
                                                                                average_metrics[6]))


        print("wrote results to {} ({})".format(output_dir, hyper_string))

    else:
        input_unbatched = dataset.get_item_by_id(entry)
        # for key in ["rgb", "albedo", "rawdepth", "spad", "mask", "rawdepth_orig", "mask_orig", "albedo_orig"]:
        #     input_[key] = input_[key].unsqueeze(0)
        from torch.utils.data._utils.collate import default_collate

        data = default_collate([input_unbatched])

        # Checks
        entry = data["entry"][0]
        i = int(entry)
        entry = entry if isinstance(entry, str) else entry.item()
        print("Evaluating {}[{}]".format(dataset_type, i))
        # Rescale SPAD
        spad_rescaled = rescale_bins(spad_data[i, ...], min_depth, max_depth, sid_obj)
        weights = np.ones_like(depth_data[i, 0, ...])
        if use_intensity:
            weights = intensity_data[i, 0, ...]
        spad_rescaled = preprocess_spad(spad_rescaled, sid_obj, use_squared_falloff, dc_count > 0.,
                                        lam=lam, eps_rel=eps_rel)

        pred, _ = image_histogram_match(depth_data[i, 0, ...], spad_rescaled, weights, sid_obj)
        # break
        # Calculate metrics
        gt = data["depth_cropped"]
        print(gt.shape)
        print(pred.shape)

        pred_metrics = get_depth_metrics(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0),
                                         gt,
                                         torch.ones_like(gt))
        if save_outputs:
            np.save(os.path.join(output_dir, "dorn_{}[{}]_{}_out.npy".format(dataset_type, entry, hyper_string)))
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
        print(
            "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(pred_metrics["delta1"],
                                                                                pred_metrics["delta2"],
                                                                                pred_metrics["delta3"],
                                                                                pred_metrics["rel_abs_diff"],
                                                                                pred_metrics["rms"],
                                                                                pred_metrics["log10"]))



