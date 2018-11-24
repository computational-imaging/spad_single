import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import argparse
from tqdm import tqdm
import configparser
from configparser import ConfigParser
from models import FusionDenoiseModel, DenoiseModel, Upsample2xDenoiseModel
import scipy
import scipy.io
import os
import pathlib


cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

parser = argparse.ArgumentParser(
        description='PyTorch Deep Sensor Fusion Middlebury Evaluation')
parser.add_argument('--option', default=None, type=str,
                    metavar='NAME', help='Name of model to use with options in config file, \
                    either FusionDenoise, Denoise, \
                    or Upsample2xDenoise)')
parser.add_argument('--config', default='captured.ini', type=str,
                    metavar='FILE', help='name of configuration file')
parser.add_argument('--save_raw', default='None',
                    metavar='1 or 0', help='Save the raw, \
                            3d volume output of the network')
parser.add_argument('--gpu', default=None, metavar='N',
                    help='which gpu')
parser.add_argument('--ckpt_noise_param_idx', nargs='+', default=None,
                    metavar='N', type=str,
                    help='model trained on which noise level to use \
                         (value 1-9, default: all)')
parser.add_argument('--scene', default=None, type=str, nargs='+',
                    metavar='FILE', help='name of scene to use \
                    (default: NONE->all)')

scenedir = 'captured/'
outdir = 'results_captured/'
scenenames = ['checkerboard', 'elephant',
              'hallway', 'kitchen',
              'lamp', 'roll',
              'stairs_ball', 'stairs_walking',
              'stuff']
pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)


def parse_arguments(args):
    config = ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.optionxform = str
    config.read(args.config)

    if args.option is not None:
        config.set('params', 'option', args.option)
    option = config.get('params', 'option')

    if args.gpu:
        config.set('params', 'gpu', args.gpu)
    if args.ckpt_noise_param_idx:
        config.set('params', 'ckpt_noise_param_idx',
                   ' '.join(args.ckpt_noise_param_idx))
    if args.save_raw:
        config.set('params', 'save_raw', args.save_raw)
    if args.scene:
        config.set('params', 'scene', ' '.join(args.scene))

    # read all values from config file
    opt = {}
    opt['gpu'] = config.get('params', 'gpu')
    opt['save_raw'] = config.get('params', 'save_raw')
    opt['ckpt_noise_param_idx'] = \
        config.get('params', 'ckpt_noise_param_idx').split()
    opt['ckpt_noise_param_idx'] = \
        [int(idx) for idx in opt['ckpt_noise_param_idx']]
    if not opt['ckpt_noise_param_idx']:
        opt['ckpt_noise_param_idx'] = np.arange(1, 11)

    opt['option'] = config.get('params', 'option')
    opt['scene'] = config.get('params', 'scene').split()
    if not opt['scene']:
        opt['scene'] = scenenames
    opt['checkpoint'] = []

    if option != 'Upsample2xDenoise':
        for n in opt['ckpt_noise_param_idx']:
            opt['checkpoint'].append(config.get(option,
                                     'ckpt_noise_param_{}'.format(n)))
    else:
        opt['checkpoint'].append(
            config.get(option, 'ckpt_finetune_noise_param_10'))
    return opt


def process_denoise(opt, model, captured_filename, out_filename):
    spad_all = \
        np.asarray(scipy.io.loadmat(captured_filename)['spad_processed_data'])
    spad_all = spad_all[0]

    intensity_all = \
        np.asarray(scipy.io.loadmat(captured_filename)['cam_processed_data'])
    intensity_all = intensity_all[0]
    num_frames = np.minimum(len(spad_all), len(intensity_all))
    num_rows = 256
    num_cols = 256

    for kk in range(num_frames):
        print('Processing frame {} of {}'.format(kk+1, num_frames))
        spad = np.asarray(scipy.sparse.csc_matrix.todense(spad_all[kk]))
        spad = spad.reshape([1536, 256, 256])
        spad = spad.transpose((0, 2, 1))
        intensity = intensity_all[kk].astype(np.float32)/255

        base_pad = 16
        step = 16
        grab = 32
        dim = 64

        spad_var = Variable(torch.from_numpy(spad))
        spad_var = spad_var.unsqueeze(0).unsqueeze(0).type(dtype)
        spad_var_pad = \
            torch.nn.functional.pad(spad_var, (base_pad, 0, base_pad, 0, 0, 0))

        if 'Upsample' not in opt['option']:
            # prepare input/output variables
            out = np.zeros((num_rows, num_cols))
            raw_out = np.zeros((1536, num_rows, num_cols))
            intensity = scipy.misc.imresize(intensity,
                                            (num_rows, num_cols),
                                            mode='F')
            intensity_var = Variable(torch.from_numpy(intensity))
            intensity_var = intensity_var.unsqueeze(0).unsqueeze(0).type(dtype)
            intensity_var_pad = \
                torch.nn.functional.pad(intensity_var,
                                        (base_pad, 0, base_pad, 0))

        else:
            # prepare input/output variables
            out = np.zeros((num_rows*2-1, num_cols*2-1))
            raw_out = np.zeros((1536, num_rows, num_cols))
            intensity = scipy.misc.imresize(intensity,
                                            (num_rows*2, num_cols*2),
                                            mode='F')
            intensity_var = Variable(torch.from_numpy(intensity))
            intensity_var = \
                intensity_var.unsqueeze(0).unsqueeze(0).type(dtype)
            intensity_var_pad =  \
                torch.nn.functional.pad(intensity_var,
                                        (base_pad*2, 0, base_pad*2, 0))

            # interpolate between processed tiles
            x = np.arange(-32, 32)
            y = x
            xv, yv = np.meshgrid(x, y)
            dist = np.sqrt(xv**2 + yv**2)
            kernel = 1-(dist - np.min(dist))/(np.max(dist) - np.min(dist))
            kernel = kernel.astype(np.float32)
            counts_up = np.zeros((511, 511))

        for i in tqdm(range(16)):
            for j in range(16):
                # process overlapping tiles of image and assemble
                if 'FusionDenoise' == opt['option']:
                    spad_input = spad_var_pad[:, :, :,
                                              i*step:(i)*step+dim,
                                              j*step:(j)*step+dim]
                    intensity_input = intensity_var_pad[:, :,
                                                        i*step:(i)*step+dim,
                                                        j*step:(j)*step+dim]
                    denoise_out, sargmax = model(spad_input, intensity_input)
                    denoise_out = denoise_out.data.cpu().numpy().squeeze()
                    tile_out = np.argmax(denoise_out, axis=0)
                    out[i*step:(i+1)*step, j*step:(j+1)*step] = \
                        tile_out[step//2:step//2+step, step//2:step//2+step]
                    raw_out[:, i*step:(i+1)*step, j*step:(j+1)*step] = \
                        denoise_out[:, step//2:step//2+step, step//2:step//2+step]
                elif 'Denoise' == opt['option']:
                    spad_input = spad_var_pad[:, :, :,
                                              i*step:(i)*step+dim,
                                              j*step:(j)*step+dim]
                    denoise_out, sargmax = model(spad_input)
                    denoise_out = denoise_out.data.cpu().numpy().squeeze()
                    tile_out = np.argmax(denoise_out, axis=0)
                    out[i*step:(i+1)*step, j*step:(j+1)*step] = \
                        tile_out[step//2:step//2+step, step//2:step//2+step]
                    raw_out[:, i*step:(i+1)*step, j*step:(j+1)*step] = \
                        denoise_out[:, step//2:step//2+step, step//2:step//2+step]
                elif 'Upsample2xDenoise' in opt['option']:
                    spad_input = spad_var_pad[:, :, :,
                                              i*step:(i)*step+dim,
                                              j*step:(j)*step+dim]
                    intensity_input = \
                        intensity_var_pad[:, :,
                                          2*i*step:2*i*step+2*dim,
                                          2*j*step:2*j*step+2*dim]
                    denoise_out, soft_argmax, hf_depth_out, depth_out = \
                        model(spad_input, intensity_input)

                    tile_out = depth_out.data.cpu().numpy().squeeze()
                    chunk = out[2*i*step:2*i*step+2*grab,
                                2*j*step:2*j*step+2*grab]

                    out[2*i*step:2*i*step+2*grab,
                        2*j*step:2*j*step+2*grab] += \
                        kernel[:chunk.shape[0], :chunk.shape[1]]\
                        * (tile_out[grab:grab+2*grab, grab:grab+2*grab]
                           [:chunk.shape[0], :chunk.shape[1]])

                    counts_up[2*i*step:2*i*step+2*grab,
                              2*j*step:2*j*step+2*grab] += \
                        kernel[:chunk.shape[0], :chunk.shape[1]]

                    chunk = raw_out[:, i*step:i*step+grab, j*step:j*step+grab]
                    raw_out[:, i*step:i*step+grab, j*step:j*step+grab] = \
                        (denoise_out.data.cpu().numpy().squeeze()
                         [:, grab//2:grab//2+grab, grab//2:grab//2+grab]
                         [:, :chunk.shape[1], :chunk.shape[2]])
                else:
                    raise ValueError("'option' parameter set to invalid value")

        # convert to meters
        if 'Upsample2xDenoise' in opt['option']:
            out /= counts_up

        out *= 6 / 1536.
        reconstruction = {'out': out, 'raw_out': raw_out}
        scipy.io.savemat(out_filename.replace('.mat', '_{}.mat'.format(kk)),
                         reconstruction)


def main():
    args = parser.parse_args()
    opt = parse_arguments(args)

    # set gpu
    print('=> setting gpu to {}'.format(opt['gpu']))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu']

    # print options
    print('=> Running scenes {}'.format(', '.join(opt['scene'])))
    model_str = [str(idx) for idx in opt['ckpt_noise_param_idx']]
    print('=> for models trained on noise levels {}'.format(', '
          .join(model_str)))

    # iterate over trained models
    for model_iter, model_param in enumerate(opt['ckpt_noise_param_idx']):
        print('=> Initializing Model')
        model = eval(opt['option'] + 'Model()')
        model.type(dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print('=> Loading checkpoint {}'.format(opt['checkpoint'][model_iter]))
        ckpt = torch.load(opt['checkpoint'][model_iter])
        model_dict = model.state_dict()
        try:
            ckpt_dict = ckpt['state_dict']
        except KeyError:
            print('=> Key error loading state_dict from checkpoint; assuming \
checkpoint contains only the state_dict')
            ckpt_dict = ckpt

        for k in ckpt_dict.keys():
            model_dict.update({k: ckpt_dict[k]})
        model.load_state_dict(model_dict)

        # iterate over each scene
        for scene in opt['scene']:
            opt['curr_scene'] = scene
            print('=> Processing {}'.format(scene))

            captured_filename = \
                '{}{}.mat'.format(scenedir,
                                  scene)
            out_filename = \
                '{}{}_{}_{}.mat'.format(outdir,
                                        scene,
                                        opt['option'],
                                        model_param)

            process_denoise(opt, model, captured_filename,
                            out_filename)


if __name__ == '__main__':
    main()
