import torch.multiprocessing as mp
import torch
from collections import Counter
import argparse
import json

def determine_rank(i, m,n,sp,samples):
    torch.manual_seed(i)
    zeros = torch.zeros(m, n)
    for i in range(samples):
        matrix = torch.randn(m, n) # N(0,1)
        probs = torch.rand(m, n) # U(0,1)
        sparse = torch.where(probs < sp, matrix, zeros)
        a = torch.linalg.matrix_rank(sparse).item()
        output.append(a)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--row', type=int, required=True)
    parser.add_argument('--col', type=int, required=True)
    parser.add_argument('--sparsity', type=float, required=True)
    parser.add_argument('--sample_factor', type=int, required=True)
    # Parse the argument
    args = parser.parse_args()
    m = args.row
    n = args.col
    sp = args.sparsity
    sample_factor = args.sample_factor
    
    manager = mp.Manager()
    output = manager.list()
    torch.set_num_threads(1)
    num_processes = 24
    sample_done = 0
    sample_missing = 0
    
    while 1:
        if sample_done == 0:
            sample_missing = sample_factor
        else:
            if (max(output) - min(output) + 1) * sample_factor <= sample_done:
                break
            else:
                sample_missing = (max(output) - min(output) + 1)*sample_factor - sample_done
                
        processes = []
        for i in range(num_processes):
            p = mp.Process(target=determine_rank, args=(i, m, n, sp, int(sample_missing / num_processes),))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        sample_done += sample_missing
            
    counts = dict(Counter(output))
    counts["sample"] = sample_done
    print(f'finish {m}_{n}_{sp}.json file')
    with open(f'prob/{m}_{n}_{sp}.json', 'w') as fp:
        json.dump(counts, fp)
