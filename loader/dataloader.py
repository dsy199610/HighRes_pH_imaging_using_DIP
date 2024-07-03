import numpy as np
import h5py
from torch.utils.data import Dataset
import pathlib
import logging
import torch.nn.functional as F
from collections import defaultdict


class Dataset_pH_mpMRI(Dataset):

    def __init__(self, rabbit, slice_):
        self.rabbit = rabbit
        self.slice = slice_
        self.data_folders = list(pathlib.Path('data').iterdir())

        self.examples = []
        self.fname2nslice = defaultdict(int)
        for rabbit in sorted(self.data_folders):
            rabbitname = str(rabbit.name)
            if rabbitname not in self.rabbit:
                continue
            for slice in sorted(list(pathlib.Path('data/' + rabbitname).iterdir())):
                slicename = str(slice.name)
                if slicename not in self.slice:
                    continue
                self.examples.append((rabbitname, slicename))
                self.fname2nslice[rabbitname] += 1

        logging.info(' ' * 10)
        logging.info('--+' * 10)
        logging.info('loading rabbits: %s ' % self.fname2nslice)
        logging.info('total slices: %s' % len(self.examples))
        logging.info('--+' * 10)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        rabbit, slice = self.examples[idx]
        path = 'data/' + str(rabbit) + '/' + str(slice) + '/'

        f = h5py.File(path + str(slice) + '_pH.mat', 'r')
        var_name, _ = list(f.items())[0]
        pH = f[var_name][()]

        f = h5py.File(path + str(slice) + '_T1_base.mat', 'r')
        var_name, _ = list(f.items())[0]
        T1 = f[var_name][()]
        T1 = T1 / T1.max()
        T1 = np.transpose(T1, (1, 2, 0))

        if pathlib.Path(path + str(slice) + '_T2.mat').exists():
            f = h5py.File(path + str(slice) + '_T2.mat', 'r')
            var_name, _ = list(f.items())[0]
            T2 = f[var_name][()]
            if T2.max() != 0:
                T2 = T2 / T2.max()
            T2 = np.transpose(T2, (1, 2, 0))
        else:
            T2 = np.zeros(T1.shape)

        if pathlib.Path(path + str(slice) + '_DWI.mat').exists():
            f = h5py.File(path + str(slice) + '_DWI.mat', 'r')
            var_name, _ = list(f.items())[0]
            DWI = f[var_name][()]
            if DWI.max() != 0:
                DWI = DWI / DWI.max()
            DWI = np.transpose(DWI, (1, 2, 0))
        else:
            DWI = np.zeros(T1.shape)

        return pH[None, :, :], T1[None, :, :, :], T2[None, :, :, :], DWI[None, :, :, :], str(rabbit), str(slice)


def interpolate_MRI(T1):
    return F.interpolate(T1, size=(400, 400, 4), mode='trilinear', align_corners=True)


def interpolate_MRI_res(MRI, res):
    return F.interpolate(MRI, size=(res, res, 4), mode='trilinear', align_corners=True)


def Crop_ROI(pH, T1, T2, DWI):
    T1_height, T1_width = T1.shape[2], T1.shape[3]
    pH_height, pH_width = pH.shape[2], pH.shape[3]
    factor_height, factor_width = T1_height // pH_height, T1_width // pH_width

    top, bottom, left, right = 0, 7, 4, 3  # margins that can be cropped
    pH = pH[:, :, top: pH_height - bottom, left: pH_width - right]
    T1 = T1[:, :, top * factor_height: (pH_height - bottom) * factor_height, left * factor_width: (pH_width - right) * factor_width, :]
    T2 = T2[:, :, top * factor_height: (pH_height - bottom) * factor_height, left * factor_width: (pH_width - right) * factor_width, :]
    DWI = DWI[:, :, top * factor_height: (pH_height - bottom) * factor_height, left * factor_width: (pH_width - right) * factor_width, :]
    return pH, T1, T2, DWI