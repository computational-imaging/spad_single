import torch
import torch.utils.data
import scipy.io
import numpy as np
import skimage.transform


class ToTensor(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self):
        pass

    def __call__(self, sample):
        rates, spad, bins_hr, intensity, bins = sample['rates'],\
                                                sample['spad'],\
                                                sample['bins_hr'],\
                                                sample['intensity'],\
                                                sample['bins']
        sbr, photons = sample['sbr'], sample['photons']

        rates = torch.from_numpy(rates)
        spad = torch.from_numpy(spad)
        bins_hr = torch.from_numpy(bins_hr)
        intensity = torch.from_numpy(intensity)
        bins = torch.from_numpy(bins)
        return {'rates': rates, 'spad': spad,
                'bins_hr': bins_hr, 'bins': bins,
                'intensity': intensity,
                'sbr': sbr, 'photons': photons}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size, intensity_scale=1):
        self.output_size = output_size
        self.intensity_scale = intensity_scale

    def __call__(self, sample):
        rates, spad, bins_hr, intensity, bins = sample['rates'],\
                                                sample['spad'],\
                                                sample['bins_hr'],\
                                                sample['intensity'],\
                                                sample['bins']
        sbr, photons = sample['sbr'], sample['photons']

        h, w = spad.shape[2:]
        new_h = self.output_size
        new_w = self.output_size
        iscale = self.intensity_scale

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        rates = rates[:, :, top: top + new_h,
                      left: left + new_w]
        spad = spad[:, :, top: top + new_h,
                    left: left + new_w]
        bins_hr = bins_hr.squeeze()[8*top: 8*top + 8*new_h,
                                    8*left: 8*left + 8*new_w]
        bins_hr = skimage.transform.resize(bins_hr,
                                           (iscale * self.output_size,
                                            iscale * self.output_size),
                                           mode='constant')
        bins_hr = bins_hr.reshape([1, iscale * self.output_size,
                                   iscale * self.output_size])
        bins = bins[:, top: top + new_h,
                    left: left + new_w]
        intensity = intensity.squeeze()[8*top: 8*top + 8*new_h,
                                        8*left: 8*left + 8*new_w]
        intensity = skimage.transform.resize(intensity,
                                             (iscale * self.output_size,
                                              iscale * self.output_size),
                                             mode='constant')
        intensity = intensity.reshape([1, iscale * self.output_size,
                                       iscale * self.output_size])

        return {'rates': rates, 'spad': spad,
                'bins_hr': bins_hr, 'bins': bins,
                'intensity': intensity,
                'sbr': sbr, 'photons': photons}


class SpadDataset(torch.utils.data.Dataset):
    def __init__(self, datapath, noise_param, transform=None):
        """__init__
        :param datapath: path to text file with list of
                        training files (intensity files)
        :param transform: transform (callable, optional):
                        Optional transform to be applied
                        on a sample.
        """
        with open(datapath) as f:
            self.intensity_files = f.read().split()
        self.spad_files = []
        for idx, n in enumerate(noise_param):
            self.spad_files.extend([intensity.replace('intensity', 'spad')
                                    .replace('.mat', '_p{}.mat'.format(n))
                                    for intensity in self.intensity_files])

        intensity_files_orig = self.intensity_files.copy()
        for idx, n in enumerate(noise_param):
            if idx > 0:
                self.intensity_files.extend(intensity_files_orig)

        self.transform = transform

    def __len__(self):
        return len(self.spad_files)

    def tryitem(self, idx):
        # simulated spad measurements
        spad = np.asarray(scipy.sparse.csc_matrix.todense(scipy.io.loadmat(
            self.spad_files[idx])['spad'])).reshape([1, 64, 64, -1])
        spad = np.transpose(spad, (0, 3, 2, 1))

        # normalized pulse
        rates = np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['rates']).reshape([1, 64, 64, -1])
        rates = np.transpose(rates, (0, 3, 1, 2))
        rates = rates / np.sum(rates, axis=1)[None, :, :, :]

        intensity = np.asarray(scipy.io.loadmat(
            self.intensity_files[idx])['intensity']).astype(
                np.float32).reshape([1, 512, 512]) / 255

        # low/high resolution depth maps
        bins = (np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['bin']).astype(
                np.float32).reshape([64, 64])-1)[None, :, :] / 1023
        bins_hr = (np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['bin_hr']).astype(
                np.float32).reshape([512, 512])-1)[None, :, :] / 1023

        # sample metainfo
        sbr = np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['SBR']).astype(np.float32)
        photons = np.asarray(scipy.io.loadmat(
            self.spad_files[idx])['photons']).astype(np.float32)
        sample = {'rates': rates, 'spad': spad, 'bins_hr': bins_hr,
                  'intensity': intensity, 'bins': bins, 'sbr': sbr,
                  'photons': photons}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __getitem__(self, idx):
        try:
            sample = self.tryitem(idx)
        except Exception as e:
            print(idx, e)
            idx = idx + 1
            sample = self.tryitem(idx)
        return sample
