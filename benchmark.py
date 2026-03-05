import time
import numpy as np
import random
import torch
from torch.utils.data import DataLoader
from process_data import SyntheticDataset

random.seed(42) # Set fixed random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

num_workers = [1, 2, 4, 8]
shuffled_workers = random.sample(num_workers, len(num_workers))
# we shuffle our list to prevent filesystem caching bias
batch_size = 64

def benchmark(dataset, num_workers):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    loader_iter = iter(loader)
    next(loader_iter)
    # Skip the first batch in the benchmark
    # to avoid initialization overhead
    start = time.time()
    n_samples = 0

    for batch in loader_iter:
        n_samples += batch.size(0) # perform any simple operation inside the loop

    end = time.time()
    epoch_time = end - start
    throughput = n_samples / epoch_time
    return epoch_time, throughput

if __name__ == '__main__':
    num_trials = 10
    results_summary = {}
    for nw in shuffled_workers:
        dataset = SyntheticDataset('stress_test_data')
        trial_times = []
        throughputs = []
        for trial in range(num_trials):
            print(f'Starting {nw} workers, trial {trial+1}')
            t_n, t_p = benchmark(dataset, nw)
    
            trial_times.append(t_n)
            throughputs.append(t_p) 
            
        if trial_times: # if trials were successful
            avg_t = np.mean(trial_times)
            std_t = np.std(trial_times)
            results_summary[nw] = {'mean': avg_t, 'std': std_t, 'thr':np.mean(throughputs)}

    print("Final benchamark results:")
    for nw in num_workers:
        stats = results_summary[nw]
        print(f"Workers: {nw} | Avg Time: {stats['mean']:.2f}s | Std. error: {stats['std']}|Avg. throughput: {stats['thr']}")
    
    print("\nScaling efficiency:")
    T1 = results_summary[1]['mean']
    for nw in num_workers:
        stats = results_summary[nw]
        Tn = stats['mean']
        efficiency = (T1 / (nw * Tn)) * 100
        print(f"Workers: {nw} Efficiency: {efficiency:.2f}%")
