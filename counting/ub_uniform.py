#Usage: python ub_uniform.py --n "[3072,400,400,10]" --p "[0.075, 0.05, ...]" --name "cifar400"

import numpy as np
import gurobipy as gb
from gurobipy import GRB
from pprint import pprint
from tqdm import tqdm
import torch
from scipy.special import comb
import matplotlib.pyplot as plt
from fractions import Fraction
import time
from PruneModel import prune_model, load_model
from IPython.display import clear_output
from Integrator import newExperiment
import pandas as pd
import subprocess
import json
import argparse
import os
import ast
os.environ['MKL_THREADING_LAYER'] = 'GNU'

dynamic_hyperplanes = {}
memorized_ranks = {}
def num_hyperplanes(l,d):
    if tuple([l,d]) in dynamic_hyperplanes:
        return dynamic_hyperplanes[tuple([l,d])]
    if l == L:
        sum_ = 0
        for k in range(0,n[l]+1):
            p_dis = prob(k,n[l-1],n[l],p_sep[l-1])
            com = Fraction(sum(comb(n[l],j) for j in range(min(k,d)+1)))
            sum_ += p_dis * com
    else:
        sum_ = 0
        for k in range(0,n[l]+1):
            sum__ = 0
            for j in range(0, min(k,d)+1):
                sum__ += Fraction(comb(n[l],j)) * num_hyperplanes(l+1, min(n[l]-j,d,k))
            sum_ += prob(k,n[l-1],n[l],p_sep[l-1]) * sum__
    dynamic_hyperplanes[tuple([l,d])] = sum_
    return sum_

def prob(rank, rows, columns, p, sample_factor = 48):
    min_dimension = max(rows,columns)
#     samples = min_dimension*sample_factor
    if (rows, columns, p) not in memorized_ranks:
        zeros = torch.zeros(rows, columns) # 0
        memorized_ranks[(rows, columns, p)] = dict(zip([i for i in range(0, min_dimension+1)], [0]*(min_dimension+1)))
        subprocess.run(["python", "prob.py", "--row", str(rows), "--col", str(columns), "--sparsity", str(p), "--sample_factor", str(sample_factor)])
        if p == 1:
            with open(f'prob/{rows}_{columns}_{1.0}.json') as json_file:
                prob_dict = json.load(json_file)
        else:
            with open(f'prob/{rows}_{columns}_{p}.json') as json_file:
                prob_dict = json.load(json_file)
        memorized_ranks[(rows, columns, p)]['sample'] = prob_dict["sample"]
        for key in memorized_ranks[(rows, columns, p)].keys():
            if str(key) in prob_dict.keys():
                memorized_ranks[(rows, columns, p)][key] = prob_dict[str(key)] 
    return Fraction(memorized_ranks[(rows, columns, p)][rank],memorized_ranks[(rows, columns, p)]["sample"])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--n', type=str, required=True)
    parser.add_argument('--p', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    
    # Parse the argument
    args = parser.parse_args()
    n = ast.literal_eval(args.n)
    sp = ast.literal_eval(args.p)
    name = args.name

    in_size = n[0]
    L = len(n) - 1
    dct = {}
    val_lst = []
    p_value = []
    
    p_value.append((1, 1))
    num_hyperplane = {}
    dynamic_hyperplanes = {}
    p_sep = [1, 1, 1]
    val_lst.append(num_hyperplanes(1, in_size))
    
    for p in sp:
        # find local sparsity arrangments
        p_value.append((p, p))
        num_hyperplane = {}
        dynamic_hyperplanes = {}
        p_sep = [p, p, 1]
        val_lst.append(num_hyperplanes(1, in_size))
        
    dct["sparsity"] = p_value
    dct["upper_bound"] = val_lst
    df = pd.DataFrame(dct) 
    df.to_csv(name + '_ub_u.csv') 