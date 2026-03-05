import numpy as np
import os

KB = 1024
MB = 1024 * KB
GB = 1024 * MB

# Align row size with page size, with
# row size = 1024 * 4 = 4KB (standard page size)
n_columns = 1024
# We want to be able to ditribute the work for
# n_workers = 1, 2, 4, 8 respectively, hence choose 10
n_files = 10
batch_size = 64
def generate_synthetic(num_files, file_size, num_cols, seed = 42):
    """
    Generate num_files of file_size bytes each
    Each file contains equal number of rows with and columns of float32 data
    We keep our files in .npy format as it integrates naturally with NumPy and PyTorch
    """
    np.random.seed(seed)
    dirname = 'stress_test_data'
    os.makedirs(dirname, exist_ok=True)
    dtype_size = np.dtype(np.float32).itemsize
    rows_per_file = file_size // (num_cols * dtype_size)
    # print(f'Wrting {num_files} files of shape {rows_per_file} * {num_cols}')
    for i in range(num_files):
        data=np.random.randn(rows_per_file, num_cols).astype(np.float32)
        np.save(f'{dirname}/data_{i:02d}.npy', data)

generate_synthetic(10, 1 * GB, n_columns)
