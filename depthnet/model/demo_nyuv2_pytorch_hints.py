import torch
from DORN_pytorch import DORN_nyu_rawhints
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from scipy.signal import fftconvolve
import scipy.io as sio
import argparse
import os
import pdb
import json

# from split_utils import build_index
from loss_numpy import delta, mse, rel_abs_diff, rel_sqr_diff

from DORN_pytorch import DORN_nyu_rawhints

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# parser.add_argument('--filename', type=str, default='./data/NYUV2/demo_01.png', help='path to an image')
parser.add_argument('--rootdir', type=str, default="/home/markn1/spad_single/data/nyu_depth_v2_processed",
                    help="rootdir of dataset")
parser.add_argument('--blacklist', type=str,
                    default="/home/markn1/spad_single/data/nyu_depth_v2_processed/blacklist.txt",
                    help="images to not calculate losses on")
parser.add_argument('--indexfile', type=str, default="/home/markn1/spad_single/data/nyu_depth_v2_processed/test.json",
                    help="index of dataset to load")
parser.add_argument('--outputroot', type=str, default='/home/markn1/DORN/result/NYUV2/pytorch/hints', help='output path')
parser.add_argument('--outputlosses', type=str, default='losses.json',
                    help="records average losses on whole dataset. path is relative to outputroot")
parser.add_argument("--torch-path", type=str, default="./torch_params_nyuv2_first_flip.pth.tar")
parser.add_argument('--max-depth', type=float, default=10.0)
parser.add_argument('--min-depth', type=float, default=0.0)
parser.add_argument('--sid-bins', type=int, default=68)
parser.add_argument('--spad-bins', type=int, default=1024)
parser.add_argument('--cuda-device', type=str, default="0")


def depth_prediction_with_hints(img_file, depth_truth, albedo, mask, net, device):
    rgb, H, W = load_image_cv2(img_file, device)
    # rgb, H, W = load_image_torchvision(filename, device)

    # sid_hist = load_sid_rawhist(depth_truth, mask, args.min_depth, args.max_depth, args.sid_bins, device)
    # sid_hist = load_sid_albedo_hist(depth_truth, albedo, mask, args.min_depth, args.max_depth, args.sid_bins, device)
    sid_hist = simulate_spad(depth_truth, albedo, args.spad_bins, 1e6, 1e5, 70,
                             mask, args.min_depth, args.max_depth, args.sid_bins, device)
    input_ = {"rgb": rgb,
              "sid_hist": sid_hist}
    with torch.no_grad():
        output = net(input_)
        pred = decode_ord(output) # Pred is in numpy
    # Magic stuff
    pred = pred[0,0,:,:] - 1.0
    pred = pred/25.0 - 0.36
    pred = np.exp(pred)
    ord_score = cv2.resize(pred, (W, H), interpolation=cv2.INTER_LINEAR)
    return ord_score


def load_sid_rawhist(depth_truth, mask, min_depth, max_depth, sid_bins, device):
    """
    Mask off invalid depths
    :param depth_truth: numpy array containing the ground truth depth map
    :param min_depth:
    :param max_depth:
    :param sid_bins:
    :param device: torch.device to load the histogram to.
    :return: torch(.cuda).FloatTensor with the histogram
    """
    offset = 1.0 - min_depth
    start = 1.0
    end = max_depth + offset
    sid_bin_edges = [end**(float(i)/sid_bins) for i in range(sid_bins + 1)]
    # print(depth_truth)
    # print(sid_bin_edges)
    # Don't use entries where depth is invalid.
    sid_hist, _ = np.histogram((depth_truth + 1.0), bins=sid_bin_edges, weights=mask, density=True)
    # print(sid_hist)
    sid_hist = sid_hist/np.sum(sid_hist) # Make it a histogram in bins
    sid_hist = torch.from_numpy(sid_hist).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).float()
    sid_hist = sid_hist.to(device)
    return sid_hist


def load_sid_albedo_hist(depth_truth, albedo, mask, min_depth, max_depth, sid_bins, device):
    """
    Mask off invalid depths
    :param depth_truth: numpy array containing the ground truth depth map
    :param min_depth:
    :param max_depth:
    :param sid_bins:
    :param device: torch.device to load the histogram to.
    :return: torch(.cuda).FloatTensor with the histogram
    """
    offset = 1.0 - min_depth
    start = 1.0
    end = max_depth + offset
    sid_bin_edges = [end**(float(i)/sid_bins) for i in range(sid_bins + 1)]
    # print(depth_truth)
    # print(sid_bin_edges)
    # Don't use entries where depth is invalid.
    weights = mask * albedo
    sid_hist, _ = np.histogram((depth_truth + 1.0), bins=sid_bin_edges, weights=weights, density=True)
    # print(sid_hist)
    sid_hist = sid_hist/np.sum(sid_hist) # Make it a histogram in bins
    sid_hist = torch.from_numpy(sid_hist).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).float()
    sid_hist = sid_hist.to(device)
    return sid_hist


def load_sid_albedo_falloff_hist(depth_truth, albedo, mask, min_depth, max_depth, sid_bins, device):
    """
    Mask off invalid depths
    :param depth_truth: numpy array containing the ground truth depth map
    :param min_depth:
    :param max_depth:
    :param sid_bins:
    :param device: torch.device to load the histogram to.
    :return: torch(.cuda).FloatTensor with the histogram
    """
    offset = 1.0 - min_depth
    start = 1.0
    end = max_depth + offset
    sid_bin_edges = [end**(float(i)/sid_bins) for i in range(sid_bins + 1)]
    # print(depth_truth)
    # print(sid_bin_edges)
    # Don't use entries where depth is invalid.
    weights = mask * albedo/(depth_truth**2 + 1e-6)
    sid_hist, _ = np.histogram((depth_truth + 1.0), bins=sid_bin_edges, weights=weights, density=True)
    # print(sid_hist)
    sid_hist = sid_hist/np.sum(sid_hist) # Make it a histogram in bins
    sid_hist = torch.from_numpy(sid_hist).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).float()
    sid_hist = sid_hist.to(device)
    return sid_hist


def makeGaussianPSF(size, fwhm=3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    x0 = size // 2
    return np.roll(np.exp(-4 * np.log(2) * ((x - x0) ** 2) / fwhm ** 2), len(x) - x0)


def rescale_bins(spad_counts, sid_bins, min_depth, max_depth):
    """Use bin numbers to do sid discretization.

    Assign photons to sid bins proportionally according to the amount of overlap between
    the sid bin range and the spad_count bin.
    """
    alpha = 1.
    offset = 1.0 - min_depth
    beta = max_depth + offset

    # Get edges of sid bins in meters
    sid_bin_edges_m = np.array([beta ** (float(i) / sid_bins) for i in range(sid_bins + 1)]) - offset

    # Convert sid_bin_edges_m into units of spad bins
    sid_bin_edges_bin = sid_bin_edges_m * len(spad_counts) / (max_depth - min_depth)

    # Map spad_counts onto sid_bin indices
    sid_counts = np.zeros(sid_bins)
    for i in range(sid_bins):
        left = sid_bin_edges_bin[i]
        right = sid_bin_edges_bin[i + 1]
        curr = left
        while curr != right:
            curr = np.min([right, np.floor(left + 1.)])  # Don't go across spad bins - stop at integers
            sid_counts[i] += (curr - left) * spad_counts[int(np.floor(left))]
            # Update window
            left = curr
    return sid_counts


def simulate_spad(depth_truth, albedo, spad_bins, photon_count, dc_count, fwhm_ps,
                  mask, min_depth, max_depth, sid_bins, device):
    """
    min_depth, max_depth in meters
    fwhm: given in picoseconds
    """
    #     spad_bin_edges = np.linspace(min_depth, max_depth, spad_bins + 1)
    weights = (albedo / (depth_truth ** 2 + 1e-6)) * mask
    depth_hist, _ = np.histogram(depth_truth, bins=spad_bins, range=(min_depth, max_depth), weights=weights)

    # Scale by number of photons
    #     print(spad_counts.shape)
    spad_counts = depth_hist * (photon_count / np.sum(depth_hist))
    # Add ambient/dark counts (dc_count)
    spad_counts += dc_count / spad_bins * np.ones(len(spad_counts))

    # Convolve with PSF
    bin_width_m = float(max_depth - min_depth) / spad_bins  # meters/bin
    bin_width_ps = 2 * bin_width_m * 1e12 / (
        3e8)  # ps/bin, speed of light = 3e8, x2 because light needs to travel there and back.
    fwhm_bin = fwhm_ps / bin_width_ps
    psf = makeGaussianPSF(len(spad_counts), fwhm=fwhm_bin)
    # print(psf)
    # print(spad_counts)
    spad_counts = fftconvolve(psf, spad_counts)[:len(spad_counts)]
    # print(spad_counts)

    # Apply poisson
    spad_counts = np.random.poisson(spad_counts)
    sid_counts = rescale_bins(spad_counts, sid_bins, min_depth, max_depth)
    sid_counts = sid_counts/np.sum(sid_counts)
    # return sid_counts#, spad_counts, depth_hist, psf
    sid_hist = torch.from_numpy(sid_counts).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).float()
    sid_hist = sid_hist.to(device)
    return sid_hist

def load_hints_net(device):
    # net = DORN_nyu_rawhints(alpha = 3.)
    # net = DORN_nyu_rawhints(alpha = 0.3)
    # net = DORN_nyu_rawhints(alpha = 1.0)
    net = DORN_nyu_rawhints(alpha = 0.1)
    # net = DORN_nyu_rawhints(alpha = 4.)
    # net = DORN_nyu_rawhints(alpha = 6.)
    # net = DORN_nyu_rawhints(alpha = 8)
    # net = DORN_nyu_rawhints(alpha = 10.)
    # net.load_state_dict(torch.load(args.torch_path))
    net.to(device)
    net.eval()
    return net


def load_image_cv2(img_file, device):
    rgb_cv2 = cv2.imread(img_file, cv2.IMREAD_COLOR)
    H, W = rgb_cv2.shape[:2]
    rgb_cv2 = rgb_cv2.astype(np.float32)
    rgb_cv2 = rgb_cv2 - np.array([[[103.0626, 115.9029, 123.1516]]]).astype(np.float32)
    rgb_cv2 = cv2.resize(rgb_cv2, (353, 257), interpolation=cv2.INTER_LINEAR)
    rgb = torch.from_numpy(rgb_cv2.transpose(2, 0, 1)).unsqueeze(0).flip([1])
    rgb = rgb.to(device)
    return rgb, H, W


def load_image_torchvision(img_file, device):
    pixel_means = torch.tensor([103.0626, 115.9029, 123.1516]).unsqueeze(-1).unsqueeze(-1)
    transform = transforms.Compose([
        transforms.Resize((257, 353)),  # (Height, Width)
        transforms.ToTensor()
    ])
    rgb_pil = Image.open(img_file)
    W, H = rgb_pil.size
    rgb_torch = transform(rgb_pil) * 255.
    rgb = (rgb_torch - pixel_means).unsqueeze(0)
    rgb = rgb.to(device)
    return rgb, H, W


def decode_ord(data_pytorch):
    """Takes a pytorch tensor, converts to numpy, then
    does the ordinal loss decoding.
    """
    data = data_pytorch.cpu().numpy()
    N = data.shape[0]
    C = data.shape[1]
    H = data.shape[2]
    W = data.shape[3]
    ord_labels = data
    decode_label = np.zeros((N, 1, H, W), dtype=np.float32)
    ord_num = C/2
    for i in range(int(ord_num)):
        ord_i = ord_labels[:,2*i:2*i+2,:,:]
        decode_label = decode_label + np.argmax(ord_i, axis=1)
    return decode_label.astype(np.float32, copy=False)


def convert_to_uint8(img, min_val, max_val):
    return np.uint8((img - min_val)/(max_val - min_val)*255.0)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device: {} (CUDA_VISIBLE_DEVICES = {})".format(device,
                                                                os.environ["CUDA_VISIBLE_DEVICES"]))

    # print("Using device: {}".format(device))
    net = load_hints_net(device)
    pixel_means = np.array([[[103.0626, 115.9029, 123.1516]]])

    with open(args.indexfile, 'r') as f:
        print("Loading index from {}".format(args.indexfile))
        index = json.load(f)

    if args.blacklist is not None:
        print("Loading blacklist from {}".format(args.blacklist))
        with open(args.blacklist, "r") as f:
            blacklist = [line.strip() for line in f.readlines()]

    print("Running tests...")
    loss_fns = []
    loss_fns.append(("mse", mse))
    loss_fns.append(("delta1", lambda p, t, m: delta(p, t, m, threshold=1.25)))
    loss_fns.append(("delta2", lambda p, t, m: delta(p, t, m, threshold=1.25 ** 2)))
    loss_fns.append(("delta3", lambda p, t, m: delta(p, t, m, threshold=1.25 ** 3)))
    loss_fns.append(("rel_abs_diff", rel_abs_diff))
    loss_fns.append(("rel_sqr_diff", rel_sqr_diff))
    npixels = 0.

    total_losses = {loss_name: 0. for loss_name, _ in loss_fns}
    for entry in index:
        if entry in blacklist:
            continue
        print(entry)
        rgb_file = os.path.join(args.rootdir, index[entry]["rgb"])

        depth_truth_file = os.path.join(args.rootdir, index[entry]["rawdepth"])
        depth_truth = cv2.imread(depth_truth_file, cv2.IMREAD_ANYDEPTH)
        depth_truth = depth_truth/1000.
        boolmask = (depth_truth <= args.min_depth) | (depth_truth >= args.max_depth)
        mask = 1.0 - boolmask.astype(float)

        albedo_file = os.path.join(args.rootdir, index[entry]["albedo"])
        albedo = cv2.imread(albedo_file, cv2.IMREAD_COLOR)
        albedo = albedo[:,:,1] # Green channel only

        depth = depth_prediction_with_hints(rgb_file, depth_truth, albedo, mask, net, device)


        # Calculate metrics
        npixels += np.sum(mask)
        for loss_name, loss_fn in loss_fns:
            avg_loss = loss_fn(depth, depth_truth, mask)
            total_losses[loss_name] += avg_loss * np.sum(mask)

        img_id = entry.replace("/", "_")
        if not os.path.exists(args.outputroot):
            os.makedirs(args.outputroot)
        # Write output to file
        depth_img = convert_to_uint8(depth, args.min_depth, args.max_depth)
        cv2.imwrite(str(args.outputroot + '/' + img_id + '_pred.png'), depth_img)

        # Write ground truth to file
        truth_img = convert_to_uint8(depth_truth, args.min_depth, args.max_depth)
        cv2.imwrite(str(args.outputroot + '/' + img_id + '_truth.png'), truth_img)

        #TESTING
        # break
    # Save as a json
    avg_losses = {loss_name: total_losses[loss_name]/npixels for loss_name in total_losses}
    avg_losses["network"] = "dorn_pytorch_albedo_hints_{}_spad".format(net.alpha)
    if "mse" in avg_losses:
        avg_losses["rmse"] = np.sqrt(avg_losses["mse"])
    with open(os.path.join(args.outputroot, args.outputlosses), "w") as f:
        json.dump(avg_losses, f)
    print("avg_losses")
    print(avg_losses)
