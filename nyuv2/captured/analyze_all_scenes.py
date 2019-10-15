#! /usr/bin/env python3
import os
import cv2
import numpy as np
import scipy.io as sio
import scipy.signal as signal
import torch

# Models
from DenseDepthModel import DenseDepth
from DORN import DORN
from MiDaSModel import get_midas, midas_predict
from capture_utils import loadmat_h5py, z_to_r, r_to_z, rescale_bins, normals_from_depth, \
                          get_closer_to_mod, load_spad, preprocess_spad, load_and_crop_kinect, get_hist_med, \
                          depth_imwrite, savefig_no_whitespace
from models.data.data_utils.sid_utils import SID
from models.loss import get_depth_metrics
from remove_dc_from_spad import remove_dc_from_spad_edge
from weighted_histogram_matching import image_histogram_match, image_histogram_match_variable_bin
# from spad_utils import rescale_bins

from camera_utils import extract_camera_params, project_depth, undistort_img

from models.core.checkpoint import safe_makedir

import h5py
import matplotlib.pyplot as plt

from sacred import Experiment

ex = Experiment("analyze_all_scenes")

@ex.config
def cfg():
    data_dir = "data"
    calibration_file = os.path.join(data_dir, "calibration", "camera_params.mat")
    # Offset is a pixel-wise offset to adjust the crop and compensate for calibration drift
    # bin_width_ps is the bin width in picoseconds of the SPAD for that particular scene
    # min_r and max_r are set to reject direct reflections off the beam splitter.
    scenes = {
        "8_29_kitchen_scene": {"offset": (-10, -8), "bin_width_ps": 16, "min_r": 0.4, "max_r": 9.},
        "8_29_conference_room_scene": {"offset": (-16, -12), "bin_width_ps": 16, "min_r": 0.4, "max_r": 9.},
        "8_30_conference_room2_scene": {"offset": (-16, -12), "bin_width_ps": 16, "min_r": 0.4, "max_r": 9.},
        "8_30_Hallway": {"offset": (0, 0), "bin_width_ps": 16, "min_r": 0.4, "max_r": 9.},
        "8_30_poster_scene": {"offset": (0, 0), "bin_width_ps": 16, "min_r": 0.4, "max_r": 9.},
        "8_30_small_lab_scene": {"offset": (0, 0), "bin_width_ps": 16, "min_r": 0.4, "max_r": 9.},
        "8_31_outdoor3": {"offset": (0, 0), "bin_width_ps": 32, "min_r": 0.4, "max_r": 11.},
    }
    # rgb in [0, 255]
    models = {
        "midas": {"load": lambda model_path, device: get_midas("MiDaS/model.pt", device),
                  "run": lambda model, rgb, depth_range, device: midas_predict(model, rgb/255., depth_range, device)},
        "dorn": {"load": lambda model_path, device: DORN(),
                 "run": lambda model, rgb, depth_range, device: dorn_predict(model, rgb)},
        "densedepth": {"load": lambda model_path, device: load_densedepth(model_path, device),
                       "run": lambda model, rgb, depth_range, device: model.predict(rgb).squeeze()}
    }
    use_intensity = True
    figures_dir = "figures" if use_intensity else "figures_no_intensity"

    cuda_device = "0"                       # The gpu index to run on. Should be a string
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))

def load_densedepth(model_path, device):
    model = DenseDepth()
    return model

def dorn_predict(model, rgb):
    """
    :param rgb: RGB image in [0, 255]
    :return:
    """
    rgb_torch = torch.from_numpy(rgb.transpose(2, 0, 1).copy()).float().unsqueeze(0)
    # print(rgb_torch.shape)
    bgr = model.preprocess(rgb_torch)
    # print(bgr.shape)
    z_init_torch = model.predict(bgr, resize_output=(rgb_torch.shape[2], rgb_torch.shape[3]))
    z_init = z_init_torch.cpu().numpy().squeeze()
    return z_init

@ex.automain
def analyze(figures_dir, data_dir, calibration_file, models, scenes, use_intensity, device):
    kinect_intrinsics, spad_intrinsics, RotationOfSpad, TranslationOfSpad = extract_camera_params(calibration_file)
    RotationOfKinect = RotationOfSpad.T
    TranslationOfKinect = -TranslationOfSpad.dot(RotationOfSpad.T)
    for model_str, load_run in models.items():
        model = load_run["load"](None, device)
        run_model = load_run["run"]
        output_dir = os.path.join(figures_dir, model_str)
        for scene, meta in scenes.items():
            print("Running {}...".format(scene))
            offset = meta["offset"]
            bin_width_ps = meta["bin_width_ps"]
            min_r = meta["min_r"]
            max_r = meta["max_r"]
            rootdir = os.path.join(data_dir, scene)
            scenedir = os.path.join(output_dir, scene)
            safe_makedir(os.path.join(scenedir))

            bin_width_m = bin_width_ps * 3e8 / (2 * 1e12)
            min_depth_bin = np.floor(min_r / bin_width_m).astype('int')
            max_depth_bin = np.floor(max_r / bin_width_m).astype('int')
            # Compensate for z translation only
            min_depth = min_depth_bin * bin_width_m - TranslationOfSpad[2]/1e3
            # print(TranslationOfSpad)
            max_depth = (max_depth_bin + 1) * bin_width_m - TranslationOfSpad[2]/1e3
            sid_obj_init = SID(sid_bins=600, alpha=min_depth, beta=max_depth, offset=0)
            ambient_max_depth_bin = 100

            # RGB from Kinect
            rgb, rgb_cropped, intensity, crop = load_and_crop_kinect(rootdir, kinect_file="kinect.mat")
            if not use_intensity:
                intensity = np.ones_like(intensity)

            # Load all the SPAD and kinect data
            spad = load_spad(os.path.join(rootdir, "spad", "data_accum.mat"))
            spad_relevant = spad[..., min_depth_bin:max_depth_bin]
            spad_single_relevant = np.sum(spad_relevant, axis=(0,1))
            ambient_estimate = np.mean(spad_single_relevant[:ambient_max_depth_bin])
            np.save(os.path.join(scenedir, "spad_single_relevant.npy"), spad_single_relevant)


            # Get ground truth depth
            gt_idx = np.argmax(spad[..., :max_depth_bin], axis=2)
            gt_r = signal.medfilt(np.fliplr(np.flipud((gt_idx * bin_width_m).T)), kernel_size=5)
            mask = (gt_r >= min_depth).astype('float').squeeze()
            gt_z = r_to_z(gt_r, spad_intrinsics["FocalLength"])
            gt_z = undistort_img(gt_z, **spad_intrinsics)
            mask = np.round(undistort_img(mask, **spad_intrinsics))
            # Nearest neighbor upsampling to reduce holes in output
            scale_factor = 2
            gt_z_up = cv2.resize(gt_z, dsize=(scale_factor*gt_z.shape[0], scale_factor*gt_z.shape[1]),
                                 interpolation=cv2.INTER_NEAREST)
            mask_up = cv2.resize(mask, dsize=(scale_factor*mask.shape[0], scale_factor*mask.shape[1]),
                                 interpolation=cv2.INTER_NEAREST)

            # Project GT depth and mask to RGB image coordinates and crop it.
            gt_z_proj, mask_proj = project_depth(gt_z_up, mask_up, (rgb.shape[0], rgb.shape[1]),
                                                 spad_intrinsics["FocalLength"]*scale_factor,
                                                 kinect_intrinsics["FocalLength"],
                                                 spad_intrinsics["PrincipalPoint"]*scale_factor,
                                                 kinect_intrinsics["PrincipalPoint"],
                                                 RotationOfKinect, TranslationOfKinect/1e3)
            gt_z_proj_crop = gt_z_proj[crop[0]+offset[0]:crop[1]+offset[0],
                                       crop[2]+offset[1]:crop[3]+offset[1]]
            gt_z_proj_crop = signal.medfilt(gt_z_proj_crop, kernel_size=5)
            mask_proj_crop = (gt_z_proj_crop >= min_depth).astype('float').squeeze()

            # print("gt_z_proj_crop range:")
            # print(np.min(gt_z_proj_crop))
            # print(np.max(gt_z_proj_crop))

            # Process SPAD
            spad_sid, sid_obj_pred, spad_denoised, spad_corrected = \
                preprocess_spad(spad_single_relevant, ambient_estimate, min_depth, max_depth, sid_obj_init)
            np.save(os.path.join(scenedir, "spad_denoised.npy"), spad_denoised)
            np.save(os.path.join(scenedir, "spad_corrected.npy"), spad_corrected)
            np.save(os.path.join(scenedir, "spad_sid.npy"), spad_sid)

            # Initialize with CNN
            z_init = run_model(model, rgb_cropped, depth_range=(min_depth, max_depth), device=device)
            r_init = z_to_r(z_init, kinect_intrinsics["FocalLength"])

            # Histogram Match
            weights = intensity
            # r_pred, t = image_histogram_match(r_init, spad_sid, weights, sid_obj)
            r_pred, t = image_histogram_match_variable_bin(r_init, spad_sid, weights,
                                                           sid_obj_init, sid_obj_pred)
            z_pred = r_to_z(r_pred, kinect_intrinsics["FocalLength"])

            # print("z_pred range")
            # print(np.min(z_pred))
            # print(np.max(z_pred))

            # Save histograms for later inspection
            intermediates = {
                "init_index": t[0],
                "init_hist": t[1],
                "pred_index": t[2],
                "pred_hist": t[3],
                "T_count": t[4]
            }
            np.save(os.path.join(scenedir, "intermediates.npy"), intermediates)
            # Save processed SPAD data
            bin_edges = np.linspace(min_depth, max_depth, len(spad_single_relevant) + 1)
            bin_values = (bin_edges[1:] + bin_edges[:-1])/2
            spad_metadata = {
                "ambient_estimate": ambient_estimate,
                "init_bin_edges": bin_edges,
                "init_bin_values": bin_values,
                "init_sid_bin_edges": sid_obj_init.sid_bin_edges,
                "init_sid_bin_values": sid_obj_init.sid_bin_values,
                "pred_sid_bin_edges": sid_obj_pred.sid_bin_edges,
                "pred_sid_bin_values": sid_obj_pred.sid_bin_values
            }
            np.save(os.path.join(scenedir, "spad_metadata.npy"), spad_metadata)


            # Mean Match
            med_bin = get_hist_med(spad_sid)
            hist_med = sid_obj_init.sid_bin_values[med_bin.astype('int')]
            r_med_scaled = np.clip(r_init * hist_med/np.median(r_init), a_min=min_depth, a_max=max_depth)
            z_med_scaled = r_to_z(r_med_scaled, kinect_intrinsics["FocalLength"])

            min_max = {}
            for k, img in zip(["gt_r", "r_init", "r_pred", "r_med_scaled"], [gt_r, r_init, r_pred, r_med_scaled]):
                min_max[k] = (np.min(img), np.max(img))
            for k, img in zip(["gt_z", "z_init", "z_pred", "z_med_scaled", "gt_z_proj", "gt_z_proj_crop"],
                              [gt_z, z_init, z_pred, z_med_scaled, gt_z_proj, gt_z_proj_crop]):
                min_max[k] = (np.min(img), np.max(img))
            np.save(os.path.join(scenedir, "mins_and_maxes.npy"), min_max)


            # Save to figures
            print("Saving figures...")
            # spad_single_relevant w/ ambient estimate
            plt.figure()
            plt.bar(range(len(spad_single_relevant)), spad_single_relevant, log=True)
            plt.title("spad_single_relevant".format(scene))
            plt.axhline(y=ambient_estimate, color='r', linewidth=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(scenedir, "spad_single_relevant.pdf"))
            # gt_r and gt_z and gt_z_proj and gt_z_proj_crop and masks
            depth_imwrite(gt_r, os.path.join(scenedir, "gt_r"))
            depth_imwrite(gt_z, os.path.join(scenedir, "gt_z"))
            depth_imwrite(gt_z_proj, os.path.join(scenedir, "gt_z_proj"))
            depth_imwrite(gt_z_proj_crop, os.path.join(scenedir, "gt_z_proj_crop"))
            depth_imwrite(mask, os.path.join(scenedir, "mask"))
            depth_imwrite(mask_proj, os.path.join(scenedir, "mask_proj"))
            depth_imwrite(mask_proj_crop, os.path.join(scenedir, "mask_proj_crop"))
            depth_imwrite(intensity, os.path.join(scenedir, "intensity"))
            np.save(os.path.join(scenedir, "crop.npy"), crop)
            # spad_sid after preprocessing
            plt.figure()
            plt.bar(range(len(spad_sid)), spad_sid, log=True)
            plt.title("spad_sid")
            plt.tight_layout()
            plt.savefig(os.path.join(scenedir, "spad_sid.pdf"))
            # rgb, rgb_cropped, intensity
            cv2.imwrite(os.path.join(scenedir, "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(scenedir, "rgb_cropped.png"), cv2.cvtColor(rgb_cropped, cv2.COLOR_RGB2BGR))
            # r_init, z_init, diff_maps
            depth_imwrite(r_init, os.path.join(scenedir, "r_init"))
            depth_imwrite(z_init, os.path.join(scenedir, "z_init"))
            # r_pred, z_pred, diff_maps
            depth_imwrite(r_pred, os.path.join(scenedir, "r_pred"))
            depth_imwrite(z_pred, os.path.join(scenedir, "z_pred"))
            # r_med_scaled, z_med_scaled, diff_maps
            depth_imwrite(r_med_scaled, os.path.join(scenedir, "r_med_scaled"))
            depth_imwrite(z_med_scaled, os.path.join(scenedir, "z_med_scaled"))
            plt.close('all')

            # Compute metrics
            print("Computing error metrics...")
            # z_init
            # z_init_resized = cv2.resize(z_init, gt_z.shape)
            init_metrics = get_depth_metrics(torch.from_numpy(z_init).float(),
                                             torch.from_numpy(gt_z_proj_crop).float(),
                                             torch.from_numpy(mask_proj_crop).float())
            np.save(os.path.join(scenedir, "init_metrics.npy"), init_metrics)
            # z_pred
            # z_pred_resized = cv2.resize(z_pred, gt_z.shape)
            pred_metrics = get_depth_metrics(torch.from_numpy(z_pred).float(),
                                             torch.from_numpy(gt_z_proj_crop).float(),
                                             torch.from_numpy(mask_proj_crop).float())
            np.save(os.path.join(scenedir, "pred_metrics.npy"), pred_metrics)

            # z_med_scaled
            # z_med_scaled_resized = cv2.resize(z_med_scaled, gt_z.shape)
            med_scaled_metrics = get_depth_metrics(torch.from_numpy(z_med_scaled).float(),
                                                   torch.from_numpy(gt_z_proj_crop).float(),
                                                   torch.from_numpy(mask_proj_crop).float())
            np.save(os.path.join(scenedir, "med_scaled_metrics.npy"), med_scaled_metrics)

