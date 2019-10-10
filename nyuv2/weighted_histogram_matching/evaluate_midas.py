import os
import numpy as np
import torch
import pandas as pd
from sacred import Experiment
from weighted_histogram_matching import image_histogram_match
from models.data.data_utils.sid_utils import SID
from spad_utils import rescale_bins
from remove_dc_from_spad import remove_dc_from_spad_poisson, remove_dc_from_spad_ambient_estimate, remove_dc_from_spad_edge
from nyuv2_labeled_dataset import nyuv2_labeled_ingredient, load_data

from models.loss import get_depth_metrics

ex = Experiment("midas_weighted_hist_match", ingredients=[nyuv2_labeled_ingredient])

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
    # spad_file = os.path.join(data_dir, "{}_spad{}.npy".format(hyper_string,
    #                                                            "_denoised" if dc_count > 0. and use_poisson else ""))
    spad_file = os.path.join(data_dir, "{}_spad.npy".format(hyper_string))
    midas_depth_file = os.path.join(data_dir, "midas_{}_outputs.npy".format(dataset_type))

    # SID params
    sid_bins = 140
    # alpha = 0.6569154266167957
    # beta = 9.972175646365525
    alpha = 0.1
    beta = 10.
    offset = 0

    # SPAD Denoising params
    lam = 1e1 if use_poisson else 1e-1
    eps_rel = 1e-5
    n_std = 0.5

    entry = None
    save_outputs = True
    small_run = 0
    output_dir = "results"


@ex.automain
def run(dataset_type,
        spad_file,
        midas_depth_file,
        hyper_string,
        sid_bins, alpha, beta, offset,
        lam, eps_rel, n_std,
        entry, save_outputs, small_run, output_dir):
    # Load all the data:
    print("Loading SPAD data from {}".format(spad_file))
    spad_dict = np.load(spad_file, allow_pickle=True).item()
    spad_data = spad_dict["spad"]
    intensity_data = spad_dict["intensity"]
    spad_config = spad_dict["config"]
    print("Loading depth data from {}".format(midas_depth_file))
    depth_data = np.load(midas_depth_file)
    dataset = load_data(channels_first=True, dataset_type=dataset_type)

    # Read SPAD config and determine proper course of action
    dc_count = spad_config["dc_count"]
    ambient = spad_config["dc_count"]/spad_config["spad_bins"]
    use_intensity = spad_config["use_intensity"]
    use_squared_falloff = spad_config["use_squared_falloff"]
    lambertian = spad_config["lambertian"]
    use_poisson = spad_config["use_poisson"]
    min_depth = spad_config["min_depth"]
    max_depth = spad_config["max_depth"]

    print("ambient: ", ambient)
    print("dc_count: ", dc_count)
    print("use_intensity: ", use_intensity)
    print("use_squared_falloff:", use_squared_falloff)
    print("lambertian:", lambertian)
    print("ambient")

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
            if small_run and i == small_run:
                break
            entry_list.append(i)

            print("Evaluating {}[{}]".format(dataset_type, i))
            spad = spad_data[i,...]
            # spad = preprocess_spad_ambient_estimate(spad, min_depth, max_depth,
            #                                             correct_falloff=use_squared_falloff,
            #                                             remove_dc= dc_count > 0.,
            #                                             global_min_depth=np.min(depth_data),
            #                                             n_std=1. if use_poisson else 0.01)
            # Rescale SPAD_data
            # spad_rescaled = rescale_bins(spad, min_depth, max_depth, sid_obj)
            weights = np.ones_like(depth_data[i, 0, ...])
            if use_intensity:
                weights = intensity_data[i, 0, ...]
            # spad_rescaled = preprocess_spad_sid_gmm(spad_rescaled, sid_obj, use_squared_falloff, dc_count > 0.)
            if dc_count > 0.:
                # spad_rescaled = remove_dc_from_spad(spad_rescaled,
                #                            sid_obj.sid_bin_edges,
                #                            sid_obj.sid_bin_values[:-2]**2,
                #                            lam=1e1 if spad_config["use_poisson"] else 1e-1,
                #                            eps_rel=1e-5)
                # spad_rescaled = remove_dc_from_spad_poisson(spad_rescaled,
                #                                        sid_obj.sid_bin_edges,
                #                                        lam=lam)
                spad = remove_dc_from_spad_edge(spad,
                                                ambient=ambient,
                                                grad_th=3 * ambient)
                # print(spad[:10])
                # print(spad)


            if use_squared_falloff:
                # spad_rescaled = spad_rescaled * sid_obj.sid_bin_values[:-2] ** 2
                bin_edges = np.linspace(min_depth, max_depth, len(spad) + 1)
                bin_values = (bin_edges[1:] + bin_edges[:-1])/2
                spad = spad * bin_values ** 4
            spad_rescaled = rescale_bins(spad, min_depth, max_depth, sid_obj)
            pred, _ = image_histogram_match(depth_data[i, 0, ...], spad_rescaled, weights, sid_obj)
            # break
            # Calculate metrics
            gt = dataset[i]["depth_cropped"].unsqueeze(0)
            # print(gt.dtype)
            # print(pred.shape)
            # print(pred[20:30, 20:30])

            pred_metrics = get_depth_metrics(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(),
                                             gt,
                                             torch.ones_like(gt))

            for j, metric_name in enumerate(metric_list[:-1]):
                metrics[i, j] = pred_metrics[metric_name]

            metrics[i, -1] = np.size(pred)
            # Option to save outputs:
            if save_outputs:
                outputs.append(pred)
            print("\tAvg RMSE = {}".format(np.mean(metrics[:i+1, metric_list.index("rmse")])))

        if save_outputs:
            np.save(os.path.join(output_dir, "midas_{}_outputs.npy".format(hyper_string)), np.array(outputs))

        # Save metrics using pandas
        metrics_df = pd.DataFrame(data=metrics, index=entry_list, columns=metric_list)
        metrics_df.to_pickle(path=os.path.join(output_dir, "midas_{}_metrics.pkl".format(hyper_string)))
        # Compute weighted averages:
        average_metrics = np.average(metrics_df.ix[:, :-1], weights=metrics_df.weight, axis=0)
        average_df = pd.Series(data=average_metrics, index=metric_list[:-1])
        average_df.to_csv(os.path.join(output_dir, "midas_{}_avg_metrics.csv".format(hyper_string)), header=True)
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rmse', 'log_10'))
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
        spad = spad_data[i, ...]
        spad_rescaled = rescale_bins(spad, min_depth, max_depth, sid_obj)
        print("spad_rescaled", spad_rescaled)
        weights = np.ones_like(depth_data[i, 0, ...])
        if use_intensity:
            weights = intensity_data[i, 0, ...]
        # spad_rescaled = preprocess_spad_sid_gmm(spad_rescaled, sid_obj, use_squared_falloff, dc_count > 0.)
        # spad_rescaled = preprocess_spad_sid(spad_rescaled, sid_obj, use_squared_falloff, dc_count > 0.
        #                                     )

        if dc_count > 0.:
            spad_rescaled = remove_dc_from_spad(spad_rescaled,
                                           sid_obj.sid_bin_edges,
                                           sid_obj.sid_bin_values[:-2] ** 2,
                                           lam=1e1 if use_poisson else 1e-1,
                                           eps_rel=1e-5)
        if use_squared_falloff:
            spad_rescaled = spad_rescaled * sid_obj.sid_bin_values[:-2] ** 2
        # print(spad_rescaled)
        pred, _ = image_histogram_match(depth_data[i, 0, ...], spad_rescaled, weights, sid_obj)
        # break
        # Calculate metrics
        gt = data["depth_cropped"]
        print(gt.shape)
        print(pred.shape)
        print(gt[:,:,40, 60])
        print(depth_data[i,0,40,60])
        print("before rmse: ", np.sqrt(np.mean((gt.numpy() - depth_data[i,0,...])**2)))

        before_metrics = get_depth_metrics(torch.from_numpy(depth_data[i,0,...]).unsqueeze(0).unsqueeze(0).float(),
                                           gt,
                                           torch.ones_like(gt))
        pred_metrics = get_depth_metrics(torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float(),
                                         gt,
                                         torch.ones_like(gt))
        if save_outputs:
            np.save(os.path.join(output_dir, "midas_{}[{}]_{}_out.npy".format(dataset_type, entry, hyper_string)),
                    pred)

        print("before:")
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rmse', 'log_10'))
        print(
            "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(before_metrics["delta1"],
                                                                                before_metrics["delta2"],
                                                                                before_metrics["delta3"],
                                                                                before_metrics["rel_abs_diff"],
                                                                                before_metrics["rmse"],
                                                                                before_metrics["log10"]))
        print("after:")


        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('d1', 'd2', 'd3', 'rel', 'rmse', 'log_10'))
        print(
            "{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(pred_metrics["delta1"],
                                                                                pred_metrics["delta2"],
                                                                                pred_metrics["delta3"],
                                                                                pred_metrics["rel_abs_diff"],
                                                                                pred_metrics["rmse"],
                                                                                pred_metrics["log10"]))



