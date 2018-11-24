import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import os
from tensorboardX import SummaryWriter
import argparse
from datetime import datetime
from shutil import copyfile
import torchvision.transforms
from tqdm import tqdm
from util.SpadDataset import SpadDataset, RandomCrop, ToTensor
import configparser
from configparser import ConfigParser
from models import FusionDenoiseModel, DenoiseModel, Upsample8xDenoiseModel,\
                  Upsample2xDenoiseModel
import skimage.io


cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

parser = argparse.ArgumentParser(
        description='PyTorch Deep Sensor Fusion Training')
parser.add_argument('--option', default=None, type=str,
                    metavar='NAME', help='Name of model to use with options in config file, \
                    either FusionDenoise, Denoise \
                    Upsample8xDenoise, or Upsample2xDenoise)')
parser.add_argument('--logdir', default=None, type=str,
                    metavar='DIR', help='logging directory \
                    for logging')
parser.add_argument('--log_name', default=None, type=str,
                    metavar='DIR', help='name of tensorboard directory for this run\
                    for logging')
parser.add_argument('--config', default='config.ini', type=str,
                    metavar='FILE', help='name of configuration file')
parser.add_argument('--gpu', default=None, metavar='N',
                    help='which gpu')
parser.add_argument('--noise_param_idx', default=None, metavar='N', nargs='+',
                    help='which noise level we are training on (value 1-10)')
parser.add_argument('--lambda_tv', default=None, metavar='Float',
                    help='TV regularizer strength', type=float)
parser.add_argument('--lambda_up', default=None, metavar='Float',
                    help='Upsample loss strength', type=float)
parser.add_argument('--batch_size', default=None, metavar='N',
                    help='minibatch size for optimization', type=int)
parser.add_argument('--workers', default=None, metavar='N',
                    help='number of dataloader workers', type=int)
parser.add_argument('--epochs', default=None, metavar='N',
                    help='number of epochs to train for', type=int)
parser.add_argument('--lr', default=None, metavar='Float',
                    help='learning rate', type=float)
parser.add_argument('--print_every', default=None, metavar='N',
                    help='Write to log every N iterations', type=int)
parser.add_argument('--save_every', default=None, metavar='N',
                    help='Save checkpoint every N iterations', type=int)
parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--train_files', default=None, type=str, metavar='PATH',
                    help='path to list of train intensity files')
parser.add_argument('--val_files', default=None, type=str, metavar='PATH',
                    help='path to list of validation intensity files')
parser.add_argument('--override_ckpt_lr', default=None, action='store_true',
                    help='if resuming, override learning rate stored in\
                    checkpoint with command line/config file lr value')


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def tv(x):
    return torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
           torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))


def evaluate(model, val_loader, n_iter, model_name='FusionDenoiseModel'):
    model.eval()
    sample = iter(val_loader).next()
    spad = sample['spad']
    intensity = sample['intensity']
    bins = sample['bins']
    rates = sample['rates']

    spad_var = Variable(spad.type(dtype))
    depth_var = Variable(bins.type(dtype))
    intensity_var = Variable(intensity.type(dtype))
    rates_var = Variable(rates.type(dtype))

    if model_name == 'FusionDenoiseModel':
        denoise_out, sargmax = model(spad_var, intensity_var)
    elif model_name == 'DenoiseModel':
        denoise_out, sargmax = model(spad_var)
    else:
        raise ValueError('model_name does not match configured models.')

    lsmax_denoise_out = torch.nn.LogSoftmax(dim=1)(
        denoise_out).unsqueeze(1)
    kl_loss = torch.nn.KLDivLoss()(lsmax_denoise_out, rates_var)

    writer.add_scalar('data/val_loss',
                      kl_loss.data.cpu().numpy()[0], n_iter)
    writer.add_scalar('data/val_rmse', np.sqrt(np.mean((
                      sargmax.data.cpu().numpy() -
                      depth_var.data.cpu()).numpy()**2) /
                      sargmax.size()[0]), n_iter)

    im_est_depth = sargmax.data.cpu()[0:4, :, :, :].repeat(
                   1, 3, 1, 1)
    im_depth_truth = depth_var.data.cpu()[0:4, :, :].repeat(
                     1, 3, 1, 1)
    im_intensity = intensity_var.data.cpu()[0:4, :, :, :].repeat(
                   1, 3, 1, 1)
    to_display = torch.cat((
                 im_est_depth, im_depth_truth, im_intensity), 0)
    im_out = torchvision.utils.make_grid(to_display,
                                         normalize=True,
                                         scale_each=True,
                                         nrow=4)
    writer.add_image('image', im_out, n_iter)
    return


def calc_upsample_loss(model, sargmax, hf_depth_out,
                       denoise_out, bins_hr, rates_var, scale):
        # the low-resolution depth is given by the soft argmax output
        # normalize this, compute the low frequency component,
        # upsample, and then compute the high frequency image
        # we want to predict.
        sargmax_normalized = torch.zeros(sargmax.size())
        for i in range(sargmax.size()[0]):
            min_depth = torch.min(sargmax[i, :, :, :]).data
            max_depth = torch.max(sargmax[i, :, :, :]).data
            sargmax_normalized[i, :, :, :] = \
                (sargmax[i, :, :, :].data - min_depth) \
                / (max_depth - min_depth)

        # normalize high resolution depth
        bins_var_float = \
            Variable(bins_hr.type(torch.cuda.FloatTensor))
        for i in range(bins_var_float.size()[0]):
            min_depth = torch.min(bins_var_float[i, :, :, :])
            max_depth = torch.max(bins_var_float[i, :, :, :])
            bins_var_float[i, :, :, :] = \
                (bins_var_float[i, :, :, :] - min_depth) \
                / (max_depth - min_depth)

        # low pass filter low resolution depth
        lf_depth = \
            model.upsampler.lp_filter(Variable(sargmax_normalized.type(dtype)))

        # bicubic upsampling of the low frequency depth
        lf_depth_np = lf_depth.data.cpu().numpy()
        ups_lf_depth_np = \
            np.zeros((lf_depth_np.shape[0],
                      lf_depth_np.shape[2]*scale,
                      lf_depth_np.shape[3]*scale))
        for i in range(lf_depth_np.shape[0]):
            ups_lf_depth_np[i, :, :] = \
                skimage.transform.rescale(np.squeeze(
                    lf_depth_np[i, :, :, :]),
                    scale, order=3, mode='symmetric', clip=False)

        # now we can get the high frequency depth we should estimate
        ups_lf_depth = \
            Variable(torch.from_numpy(
                     ups_lf_depth_np).unsqueeze(1).type(dtype))
        ups_hf_depth = bins_var_float - ups_lf_depth

        probs = rates_var / torch.sum(rates_var, 2).unsqueeze(1)
        lsmax_denoise_out = \
            torch.nn.LogSoftmax(dim=1)(denoise_out).unsqueeze(1)

        # calculate loss
        kl_loss = torch.nn.KLDivLoss()(lsmax_denoise_out, probs)
        up_loss = torch.nn.L1Loss()(hf_depth_out, ups_hf_depth[:, :, :-1, :-1])

        return kl_loss, up_loss


def upsample_evaluate(model, val_loader, n_iter, scale,
                      model_name='Upsample8xDenoiseModel'):
    model.eval()
    sample = iter(val_loader).next()

    spad = sample['spad']
    rates = sample['rates']
    intensity = sample['intensity']
    bins_hr = sample['bins_hr'].type(dtype)

    spad_var = Variable(spad.type(dtype))
    rates_var = Variable(rates.type(dtype))
    intensity_var = Variable(intensity.type(dtype))

    # Run the model forward to compute scores and loss.
    denoise_out, sargmax, hf_depth_out, depth_out \
        = model(spad_var, intensity_var)
    kl_loss, up_loss = \
        calc_upsample_loss(model, sargmax, hf_depth_out,
                           denoise_out, bins_hr, rates_var, scale)

    writer.add_scalar('data/val_loss_kl',
                      kl_loss.data.cpu().numpy()[0], n_iter)
    writer.add_scalar('data/val_loss_up',
                      up_loss.data.cpu().numpy()[0], n_iter)
    writer.add_scalar('data/val_rmse', np.sqrt(np.mean((
                      depth_out.data.cpu().numpy() -
                      bins_hr.cpu().numpy()[:, :, :-1, :-1].squeeze())**2) /
                      depth_out.data.size()[0]), n_iter)
    im_est_depth = depth_out.data.cpu()[0:4, :, :, :].repeat(
                   1, 3, 1, 1)
    im_depth_truth = bins_hr.cpu()[0:4, :, :-1, :-1].repeat(
                     1, 3, 1, 1)
    im_intensity = intensity_var.data.cpu()[0:4, :, :-1, :-1].repeat(
                   1, 3, 1, 1)
    to_display = torch.cat((
                 im_est_depth, im_depth_truth, im_intensity), 0)
    im_out = torchvision.utils.make_grid(to_display,
                                         normalize=True,
                                         scale_each=True,
                                         nrow=4)
    writer.add_image('image', im_out, n_iter)
    return


def upsample_finetune(model, train_loader, val_loader, optimizer, n_iter,
                      lambda_up, epoch, logfile, val_every=10, save_every=100,
                      scale=8, model_name='Upsample8xDenoiseModel'):

    for sample in tqdm(train_loader):
        model.train()
        spad = sample['spad']
        rates = sample['rates']
        intensity = sample['intensity']
        bins_hr = sample['bins_hr'].type(dtype)

        spad_var = Variable(spad.type(dtype))
        rates_var = Variable(rates.type(dtype))
        intensity_var = Variable(intensity.type(dtype))

        # Run the model forward to compute scores and loss.
        denoise_out, sargmax, hf_depth_out, depth_out \
            = model(spad_var, intensity_var)
        kl_loss, up_loss = \
            calc_upsample_loss(model, sargmax, hf_depth_out,
                               denoise_out, bins_hr, rates_var, scale)
        loss = kl_loss + lambda_up * up_loss

        # Run the model backward and take a step using the optimizer.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_iter += 1

        # log in tensorboard
        writer.add_scalar('data/train_loss_kl',
                          kl_loss.data.cpu().numpy()[0], n_iter)
        writer.add_scalar('data/train_loss_up',
                          up_loss.data.cpu().numpy()[0], n_iter)
        writer.add_scalar('data/train_rmse', np.sqrt(np.mean((
                          depth_out.data.cpu().numpy() -
                          bins_hr.cpu()
                          .numpy()[:, :, :-1, :-1].squeeze())**2) /
                          depth_out.data.size()[0]), n_iter)

        if (n_iter + 1) % val_every == 0:
            model.eval()
            upsample_evaluate(model, val_loader, n_iter, scale, model_name)

        if (n_iter + 1) % save_every == 0:
            save_checkpoint({
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': n_iter,
                 }, filename=logfile +
                 '/epoch_{}_{}.pth'.format(epoch, n_iter))

    return n_iter


def train(model, train_loader, val_loader, optimizer, n_iter,
          lambda_tv, epoch, logfile, val_every=10, save_every=100,
          model_name='FusionDenoiseModel'):

    for sample in tqdm(train_loader):
        model.train()
        spad = sample['spad']
        rates = sample['rates']
        intensity = sample['intensity']
        bins = sample['bins']

        spad_var = Variable(spad.type(dtype))
        depth_var = Variable(bins.type(dtype))
        rates_var = Variable(rates.type(dtype))
        intensity_var = Variable(intensity.type(dtype))

        # Run the model forward to compute scores and loss.
        if model_name == 'FusionDenoiseModel':
            denoise_out, sargmax = model(spad_var, intensity_var)
        elif model_name == 'DenoiseModel':
            denoise_out, sargmax = model(spad_var)
        else:
            raise ValueError('model_name does not match configured models.')

        lsmax_denoise_out = torch.nn.LogSoftmax(dim=1)(
                denoise_out).unsqueeze(1)
        kl_loss = torch.nn.KLDivLoss()(lsmax_denoise_out, rates_var)
        tv_reg = lambda_tv * tv(sargmax)
        loss = kl_loss + tv_reg

        # Run the model backward and take a step using the optimizer.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        n_iter += 1

        # log in tensorboard
        writer.add_scalar('data/train_loss',
                          kl_loss.data.cpu().numpy()[0], n_iter)
        writer.add_scalar('data/train_rmse', np.sqrt(np.mean((
                          sargmax.data.cpu().numpy() -
                          depth_var.data.cpu().numpy())**2) /
                          sargmax.data.size()[0]), n_iter)

        if (n_iter + 1) % val_every == 0:
            model.eval()
            evaluate(model, val_loader, n_iter, model_name)

        if (n_iter + 1) % save_every == 0:
            save_checkpoint({
                'lr': optimizer.param_groups[0]['lr'],
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'n_iter': n_iter,
                 }, filename=logfile +
                 '/epoch_{}_{}.pth'.format(epoch, n_iter))

    return n_iter


def parse_arguments(args):
    opt = {}

    print('=> Reading config file and command line arguments')
    config = ConfigParser()
    config._interpolation = configparser.ExtendedInterpolation()
    config.read(args.config)

    # figure out which model we're working with
    if args.option is not None:
        config.set('params', 'option', args.option)
    option = config.get('params', 'option')

    # handle the case of resuming an older model
    opt['resume_new_log_folder'] = False
    if args.resume:
        config.set(option, 'resume', args.resume)
    opt['resume'] = config.get(option, 'resume')
    if opt['resume']:
        if os.path.isfile(os.path.dirname(opt['resume']) + '/config.ini'):
            print('=> Resume flag set; switching to resumed config file')
            args.config = os.path.dirname(opt['resume']) + '/config.ini'
            config.clear()
            config._interpolation = configparser.ExtendedInterpolation()
            config.read(args.config)
        else:
            opt['resume_new_log_folder'] = True

    if args.gpu:
        config.set('params', 'gpu', args.gpu)
    if args.noise_param_idx:
        config.set('params', 'noise_param_idx', ' '.join(args.noise_param_idx))
    if args.logdir:
        config.set(option, 'logdir', args.logdir)
    if args.log_name:
        config.set(option, 'log_name', args.log_name)
    if args.batch_size:
        config.set(option, 'batch_size', str(args.batch_size))
    if args.workers:
        config.set(option, 'workers', str(args.workers))
    if args.epochs:
        config.set(option, 'epochs', str(args.epochs))
    if args.lambda_tv:
        config.set(option, 'lambda_tv', str(args.lambda_tv))
    if args.lambda_up:
        config.set(option, 'lambda_up', str(args.lambda_up))
    if args.print_every:
        config.set(option, 'print_every', str(args.print_every))
    if args.save_every:
        config.set(option, 'save_every', str(args.save_every))
    if args.lr:
        config.set(option, 'lr', str(args.lr))
    if args.train_files:
        config.set(option, 'train_files', args.train_files)
    if args.val_files:
        config.set(option, 'val_files', args.val_files)

    # read all values from config file
    if 'Upsample' in option:
        opt['lambda_up'] = float(config.get(option, 'lambda_up'))
        opt['resume_msgnet'] = config.get(option, 'resume_msgnet')
        if '2x' in option:
            opt['intensity_scale'] = 2
        if '8x' in option:
            opt['intensity_scale'] = 8
    else:
        opt['lambda_tv'] = float(config.get(option, 'lambda_tv'))
        opt['intensity_scale'] = 1
        opt['resume_msgnet'] = None

    opt['gpu'] = config.get('params', 'gpu')
    opt['noise_param_idx'] = config.get('params', 'noise_param_idx').split()
    opt['noise_param_idx'] = \
        [int(idx) for idx in opt['noise_param_idx']]
    opt['logdir'] = config.get(option, 'logdir')
    opt['log_name'] = config.get(option, 'log_name')
    opt['batch_size'] = int(config.get(option, 'batch_size'))
    opt['workers'] = int(config.get(option, 'workers'))
    opt['epochs'] = int(config.get(option, 'epochs'))
    opt['print_every'] = int(config.get(option, 'print_every'))
    opt['save_every'] = int(config.get(option, 'save_every'))
    opt['lr'] = float(config.get(option, 'lr'))
    opt['train_files'] = config.get(option, 'train_files')
    opt['val_files'] = config.get(option, 'val_files')
    opt['optimizer_init'] = config.get(option, 'optimizer')
    opt['model_name'] = config.get(option, 'model_name')

    # write these values to config file
    cfgfile = open(args.config, 'w')
    config.write(cfgfile)
    cfgfile.close()

    return opt


def main():
    # get arguments and modify config file as necessary
    args = parser.parse_args()
    opt = parse_arguments(args)

    # set gpu
    print('=> setting gpu to {}'.format(opt['gpu']))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = opt['gpu']

    # tensorboard log file
    global writer
    if not opt['resume'] or opt['resume_new_log_folder']:
        now = datetime.now()
        logfile = opt['logdir'] + '/' + opt['log_name'] + '_date_' + \
            now.strftime('%m_%d-%H_%M') + '/'
        writer = SummaryWriter(logfile)
        copyfile('./config.ini', logfile + 'config.ini')
    else:
        logfile = os.path.dirname(opt['resume'])
        writer = SummaryWriter(logfile)
    print('=> Tensorboard logging to {}'.format(logfile))

    model = eval(opt['model_name'] + '()')
    model.type(dtype)

    # initialize optimization tools
    params = filter(lambda p: p.requires_grad, model.parameters())
    if opt['optimizer_init'] != '':
        print('=> Loading optimizer from config...')
        optimizer = eval(opt['optimizer_init'])
    else:
        print('=> Using default Adam optimizer')
        optimizer = torch.optim.Adam(params, opt['lr'])

    # datasets and dataloader
    train_dataset = \
        SpadDataset(opt['train_files'], opt['noise_param_idx'],
                    transform=transforms.Compose(
                    [RandomCrop(32, intensity_scale=opt['intensity_scale']),
                     ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=opt['batch_size'],
                              shuffle=True, num_workers=opt['workers'],
                              pin_memory=True)
    val_dataset = \
        SpadDataset(opt['val_files'], opt['noise_param_idx'],
                    transform=transforms.Compose(
                    [RandomCrop(32, intensity_scale=opt['intensity_scale']),
                     ToTensor()]))
    val_loader = DataLoader(val_dataset, batch_size=opt['batch_size'],
                            shuffle=True, num_workers=opt['workers'],
                            pin_memory=True)

    # resume checkpoint
    n_iter = 0
    start_epoch = 0
    if opt['resume']:
        if os.path.isfile(opt['resume']):
            print("=> loading checkpoint '{}'".format(opt['resume']))
            checkpoint = torch.load(opt['resume'])

            try:
                start_epoch = checkpoint['epoch']
            except KeyError as err:
                start_epoch = 0
                print('=> Can''t load start epoch, setting to zero')
            try:
                if not args.override_ckpt_lr:
                    opt['lr'] = checkpoint['lr']
                print('=> Loaded learning rate {}'.format(opt['lr']))
            except KeyError as err:
                print('=> Can''t load learning rate, setting to default')
            try:
                ckpt_dict = checkpoint['state_dict']
            except KeyError as err:
                ckpt_dict = checkpoint

            model_dict = model.state_dict()
            for k in ckpt_dict.keys():
                model_dict.update({k: ckpt_dict[k]})
            model.load_state_dict(model_dict)
            print('=> Loaded {}'.format(opt['resume']))

            if opt['resume_msgnet']:
                msgnet_checkpoint = torch.load(opt['resume_msgnet'])
                try:
                    msgnet_ckpt_dict = msgnet_checkpoint['state_dict']
                except KeyError as err:
                    msgnet_ckpt_dict = msgnet_checkpoint

                model_dict = model.state_dict()
                for k in msgnet_ckpt_dict.keys():
                    model_dict.update({'upsampler.' + k: msgnet_ckpt_dict[k]})
                model.load_state_dict(model_dict)
                print('=> Loaded {}'.format(opt['resume_msgnet']))

            try:
                optimizer_dict = optimizer.state_dict()
                ckpt_dict = checkpoint['optimizer']
                for k in ckpt_dict.keys():
                    optimizer_dict.update({k: ckpt_dict[k]})
                optimizer.load_state_dict(optimizer_dict)
            except (ValueError, KeyError) as err:
                print('=> Unable to resume optimizer from checkpoint')

            # set optimizer learning rate
            for g in optimizer.param_groups:
                g['lr'] = opt['lr']
            try:
                n_iter = checkpoint['n_iter']
            except KeyError:
                n_iter = 0

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(opt['resume'], start_epoch))
        else:
            print("=> no checkpoint found at '{}'".format(opt['resume']))

    # run training epochs
    print('=> starting training')
    for epoch in range(start_epoch, opt['epochs']):
        print('epoch: {}, lr: {}'.format(epoch,
                                         optimizer.param_groups[0]['lr']))
        if 'Upsample' in opt['model_name']:
            n_iter = upsample_finetune(model, train_loader, val_loader,
                                       optimizer, n_iter,
                                       opt['lambda_up'], epoch, logfile,
                                       val_every=opt['print_every'],
                                       save_every=opt['save_every'],
                                       scale=opt['intensity_scale'],
                                       model_name=opt['model_name'])
        else:
            n_iter = train(model, train_loader, val_loader, optimizer, n_iter,
                           opt['lambda_tv'], epoch, logfile,
                           val_every=opt['print_every'],
                           save_every=opt['save_every'],
                           model_name=opt['model_name'])

        # decrease the learning rate
        for g in optimizer.param_groups:
            g['lr'] *= 0.9

        save_checkpoint({
            'lr': optimizer.param_groups[0]['lr'],
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'n_iter': n_iter,
             }, filename=logfile + '/epoch_{}_{}.pth'.format(epoch, n_iter))


if __name__ == '__main__':
    main()
