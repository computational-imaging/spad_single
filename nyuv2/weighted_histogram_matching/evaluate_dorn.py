import os
import numpy as np
import torch
import pandas as pd
from sacred import Experiment
from weighted_histogram_matching import image_histogram_match
from models.data.data_utils.sid_utils import SID
from simulate_spad import rescale_bins
from nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data

from models.loss import get_depth_metrics

ex = Experiment("dorn_weighted_hist_match", ingredients=[nyuv2_labeled_ingredient])

@ex.config
def cfg(data_config):
    data_dir = "data"
    dataset_type = "test"
    spad_file = os.path.join(data_dir, "{}_int_True_fall_False_dc_0.0_spad.npy".format(dataset_type))
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

    entry = None
    save_outputs = True
    small_run = 0


@ex.automain
def run(dataset_type,
        spad_file,
        dorn_depth_file,
        sid_bins, alpha, beta, offset,
        entry, save_outputs, small_run):
    # Load all the data:
    spad_dict = np.load(spad_file).item()
    spad_data = spad_dict["spad"]
    intensity_data = spad_dict["intensity"]
    spad_config = spad_dict["config"]
    depth_data = np.load(dorn_depth_file)
    dataset = load_data(dataset_type=dataset_type)

    # Read SPAD config and determine proper course of action
    dc_count = spad_config["dc_count"]
    use_intensity = spad_config["use_intensity"]
    use_squared_falloff = spad_config["use_squared_falloff"]
    min_depth = spad_config["min_depth"]
    max_depth = spad_config["max_depth"]

    print("dc_count: ", dc_count)
    print("use_intensity: ", use_intensity)
    print("use_squared_falloff:", use_squared_falloff)

    print("spad_data.shape", spad_data.shape)
    print("depth_data.shape", depth_data.shape)
    print("intensity_data.shape", intensity_data.shape)

    sid_obj = SID(sid_bins, alpha, beta, offset)

    if entry is None:
        metric_list = ["delta1", "delta2", "delta3", "rel_abs_diff", "rmse", "mse", "log10", "weight"]
        metrics = np.zeros((len(dataset) if not small_run else small_run, len(metric_list)))
        entry_list = []
        outputs = []
        for i in range(depth_data.shape[0]):
            print("Evaluating {}[{}]".format(dataset_type, i))
            # Rescale SPAD
            spad_rescaled = rescale_bins(spad_data[i,...], min_depth, max_depth, sid_obj)
            weights = np.ones_like(depth_data[i, 0, ...])
            if use_intensity:
                weights = intensity_data[i, 0, ...]
            if use_squared_falloff:
                spad_rescaled *= sid_obj.sid_bin_values[:-2]
            if dc_count > 0:
                raise NotImplementedError
                # pass  # Solve cvxpy problem

            pred, _ = image_histogram_match(depth_data[i, 0, ...], spad_rescaled, weights, sid_obj)
            # break
            # Calculate metrics
            gt = dataset[i]["depth_cropped"]

            pred_metrics = get_depth_metrics(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0),
                                             gt,
                                             torch.ones_like(gt))


            for j, metric_name in enumerate(metric_list[:-1]):
                metrics[i, j] = pred_metrics[metric_name]

            metrics[i, -1] = pred_weight
            # Option to save outputs:
            if save_outputs:
                outputs.append(pred.cpu().numpy())

        if save_outputs:
            np.save(os.path.join(output_dir, "dorn_{}_outputs.npy".format(dataset_type)), np.concatenate(outputs, axis=0))

            # Save metrics using pandas
        metrics_df = pd.DataFrame(data=metrics, index=entry_list, columns=metric_list)
        metrics_df.to_pickle(path=os.path.join(output_dir, "dorn_{}_metrics.pkl".format(dataset_type)))
        # Compute weighted averages:
        average_metrics = np.average(metrics_df.ix[:, :-1], weights=metrics_df.weight, axis=0)
        average_df = pd.Series(data=average_metrics, index=metric_list[:-1])
        average_df.to_csv(os.path.join(output_dir, "dorn_{}_avg_metrics.csv".format(dataset_type)), header=True)
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
        print(
            "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(average_metrics[0],
                                                                                average_metrics[1],
                                                                                average_metrics[2],
                                                                                average_metrics[3],
                                                                                average_metrics[4],
                                                                                average_metrics[6]))


        print("wrote results to {}".format(output_dir))

else:
    input_unbatched = dataset.get_item_by_id(entry)
    # for key in ["rgb", "albedo", "rawdepth", "spad", "mask", "rawdepth_orig", "mask_orig", "albedo_orig"]:
    #     input_[key] = input_[key].unsqueeze(0)
    from torch.utils.data._utils.collate import default_collate

    data = default_collate([input_unbatched])

    # Checks
    entry = data["entry"][0]
    entry = entry if isinstance(entry, str) else entry.item()
    print("Entry: {}".format(entry))
    # print("remove_dc: ", model.remove_dc)
    # print("use_intensity: ", model.use_intensity)
    # print("use_squared_falloff: ", model.use_squared_falloff)
    pred, pred_metrics, pred_weight = model.evaluate(data["bgr"].to(device),
                                                     data["bgr_orig"].to(device),
                                                     data["depth_cropped"].to(device),
                                                     torch.ones_like(data["depth_cropped"]).to(device))
    if save_outputs:
        np.save(os.path.join(output_dir, "{}_{}_out.npy".format(dataset_type, entry)))
    print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rms', 'log_10'))
    print(
        "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(pred_metrics["delta1"],
                                                                            pred_metrics["delta2"],
                                                                            pred_metrics["delta3"],
                                                                            pred_metrics["rel_abs_diff"],
                                                                            pred_metrics["rms"],
                                                                            pred_metrics["log10"]))



