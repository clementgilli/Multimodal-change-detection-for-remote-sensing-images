# General
import os

# Imaging
import numpy as np
import torch

# Local


class Dataset(torch.utils.data.Dataset):
    """
    Dataset from two folders containing all the patches of the training (or validation) set as .npy files of shape
    [1 x H x W x C].
    * B: total number of patches
    * H, w: height and width of the patches
    * C: number of input (or target) channels
    """
    def __init__(self, path_folder_inp, path_folder_tar, mmap=True):

        self.path_folder_inp = path_folder_inp
        self.path_folder_tar = path_folder_tar
        self.list_files_inp = os.listdir(self.path_folder_inp)
        self.list_files_tar = os.listdir(self.path_folder_tar)

        self.inp_patches = [np.load(os.path.join(self.path_folder_inp, f), mmap_mode="r" if mmap else None)
                            for f in self.list_files_inp]
        self.tar_patches = [np.load(os.path.join(self.path_folder_tar, f), mmap_mode="r" if mmap else None)
                            for f in self.list_files_tar]

        self.b = len(self.list_files_inp)
        self.nb_chan_in = self.inp_patches[0].shape[-1]
        self.nb_chan_tar = self.tar_patches[0].shape[-1]

    def __len__(self):
        return self.b

    def __getitem__(self, index):

        batch_in = self.inp_patches[index]
        batch_tar = self.tar_patches[index]

        x = torch.tensor(batch_in)
        y = torch.tensor(batch_tar)

        x = torch.moveaxis(x, 2, 0)
        y = torch.moveaxis(y, 2, 0)

        return x, y


class PackedDataset(torch.utils.data.Dataset):
    """
    Dataset from two .npy files of shape [B x H x W x C] containing all the patches of the training (or validation) set.
    * B: total number of patches
    * H, w: height and width of the patches
    * C: number of input (or target) channels
    """
    def __init__(self, npy_path_inp, npy_path_tar, mmap=True):
        self.npy_path_inp = npy_path_inp
        self.npy_path_tar = npy_path_tar

        self.inp_patches = np.load(self.npy_path_inp, mmap_mode="r" if mmap else None)
        self.tar_patches = np.load(self.npy_path_tar, mmap_mode="r" if mmap else None)

        self.b = self.inp_patches.shape[0]
        self.nb_chan_in = self.inp_patches.shape[-1]
        self.nb_chan_tar = self.tar_patches.shape[-1]

    def __len__(self):
        return self.b

    def __getitem__(self, index):
        batch_in = self.inp_patches[index, ...]
        batch_tar = self.tar_patches[index, ...]

        x = torch.tensor(batch_in)
        y = torch.tensor(batch_tar)

        x = torch.moveaxis(x, 2, 0)
        y = torch.moveaxis(y, 2, 0)

        return x, y
