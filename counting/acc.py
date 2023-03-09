#Usage: python acc.py --n "[784,10,10,10]" --p "[0.075, 0.05, 0.025]" --name "cifar800" --model "Perceptron10" --dataset "MNIST"

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Add arguments
    parser.add_argument('--n', type=str, required=True)
    parser.add_argument('--p', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    
    # Parse the argument
    args = parser.parse_args()
    n = ast.literal_eval(args.n)
    sp = ast.literal_eval(args.p)
    name = args.name + "_ub.csv"
    name_u = args.name + "_ub_u.csv"
    df = pd.read_csv(name)
    df_u = pd.read_csv(name_u)
    
    model_arch = args.model
    dataset = args.dataset
    strategy = "LayerMagWeight"
    model_layers = n[1:]
    sample_points = 3
    method = 'exact'
    model_num = 1
    csv_name = 'testing123'
    times = 2
    in_size = n[0]

    result = {}
    log = {}
    log["global_sparsity"] = ["1"] + sp
    for i in range(times):
        log["U" + str(i)] = []
        log["O" + str(i)] = []
    old = []
    new = []
    acc_lst = []
    acc_temp = []
    for seed in range(times):
        a= newExperiment(model_arch, dataset, strategy, [[1,1]], model_layers, csv_name, sample_points, method, model_num, in_size, seed=seed)
        log["U" + str(seed)].append(a[0])
        log["O" + str(seed)].append(a[0])
        acc_temp.append(a)
    for c in range(len(acc_temp[0])):
        sum_a = 0
        for seed in range(times):
            sum_a += acc_temp[seed][c]
        sum_a = sum_a / times
    old.append(sum_a)
    new.append(sum_a)
    so = [(1,1)]
    uo = [df_u["upper_bound"].loc[0]]
        
    for p in sp:
        # extract local sparsity arrangments
        val = list(df[str(p) + "upper_bound"])
        sparsity = list(df[str(p) + "local_sparsity"])  
        so.append(ast.literal_eval(sparsity[np.argmax(val)]))
        uo.append(max(val))
        compressions = [[1/p, 1/p], [1/ast.literal_eval(sparsity[np.argmax(val)])[0], 1/ast.literal_eval(sparsity[np.argmax(val)])[1]]]
        
        acc_lst = []
        acc_temp = []
        for seed in range(times):
            a= newExperiment(model_arch, dataset, strategy, compressions, model_layers, csv_name, sample_points, method, model_num, in_size, seed=seed)
            log["U" + str(seed)].append(a[0])
            log["O" + str(seed)].append(a[1])
            acc_temp.append(a)
        for c in range(len(acc_temp[0])):
            sum_a = 0
            for seed in range(times):
                sum_a += acc_temp[seed][c]
            sum_a = sum_a / times
            acc_lst.append(sum_a)

        old.append(acc_lst[0])
        new.append(acc_lst[1])
    result["global_sparsity"] = ["1"] + sp
    result["sparsity_U"] = df_u['sparsity']
    result["UB_U"] = df_u["upper_bound"]
    result["ACC_U"] = old
    result["sparsity_O"] = so
    result["UB_O"] = uo
    result["ACC_O"] = new
    df = pd.DataFrame(result)
    df.to_csv(model_arch + '.csv')
    df = pd.DataFrame(log)
    df.to_csv(model_arch + '_acc.csv')