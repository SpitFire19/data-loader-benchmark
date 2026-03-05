import torch
import numpy as np
import os
import glob
from torch.utils.data import Dataset

class SyntheticDataset(Dataset):
    def __init__(self, data_dir):
        # read the files that contain our dataset
        self.files = sorted([
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.endswith(".npy")
        ])
        # we want to store only memory maps
        self.arrays = [
            np.load(f, mmap_mode="r")
            for f in self.files
        ]
        self.samples_per_file = self.arrays[0].shape[0]
        self.total_samples = len(self.arrays) * self.samples_per_file

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file
        sample = self.arrays[file_idx][sample_idx]
        sample = np.array(sample, copy=True)
        # we choose to return the copy of the data
        # on disk to avoid undefined behavior from PyTorch.
        return torch.from_numpy(sample)