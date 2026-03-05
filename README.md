# PyTorch DataLoader Benchmark and Darshan I/O Analysis

This project benchmarks the performance of the PyTorch ```DataLoader```
with different worker configurations and analyzes I/O behavior using
Darshan profiling reports. The goal is to evaluate how data loading 
performance scales with different numbers of workers and to identify
potential I/O bottlenecks in machine learning workflows.

------------------------------------------------------------------------

# Environment Setup

The project requires Python and several Python libraries.

## Requirements

-   Python 3.10+
-   PyTorch
-   NumPy

Please install the dependencies using:

pip install -r requirements.txt

# Dataset Generation

The benchmark uses a synthetic dataset stored on disk.
The dataset generation process is deterministic (with all random seeds being
fixed) to ensure reproducibility.

Run:

python generate.py

This script will:

-   generate **10 GB** of table data in .npy format
-   store the dataset in the directory `stress_test_data/`

A fixed random seed is used so that the dataset is identical across
runs.

------------------------------------------------------------------------

# Running the Benchmark

Run the DataLoader benchmark:

python benchmark.py

The benchmark evaluates DataLoader performance for different worker
configurations: num_workers = [1, 2, 4, 8]

For each configuration:
-   we execute 10 trials
-   we report average runtime and throughput
-   we calculate scaling efficiency as demanded in the task description

# Reproducibility

The following measures were taken to ensure reproducible results:

-   Fixed random seeds were used for Python, NumPy, and PyTorch.
-   The order of worker configurations is randomized to reduce
    filesystem cache bias.
-   Each configuration is executed multiple times and results are
    averaged.

These steps reduce measurement noise and allow results to be reproduced
on a clean system with a high precision.

# GitHub repository structure

`
├── generate.py 
├── benchmark.py 
├── process_data.py
├── requirements.txt 
└── README.md
`
-   `generate_dataset.py` -- creates the synthetic dataset
-   `benchmark.py` -- runs the DataLoader benchmark
-   `process_data.py` -- a torch.utils.data.Dataset implementation
-   `requirements.txt` -- Python dependencies
-   `darshan_reports` -- contain the Darshan reports that we later analyze
-   `report.pdf`      -- final report
------------------------------------------------------------------------

# Benchmark performance Notes

Performance results depend entirely on CPU model, disk type (HDD/SSD) and amount of system resources available.
The benchmark was run on the machine with
* CPU: Intel Core i7 14650HX
* 
The benchmark is thus designed to evaluate **relative scaling behavior**.
