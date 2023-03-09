#Usage: python ub_acc.py --n "[784,10,10,10]" --p "[0.075, 0.05, 0.025]" --name "MNIST10" --model "Perceptron10" --dataset "MNIST"

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
            sum_ += prob(k,n[l-1],n[l],p_sep[l-1]) * sum(comb(n[l],j) for j in range(min(k,d)+1))
    else:
        sum_ = 0
        for k in range(0,n[l]+1):
            sum__ = 0
            for j in range(0, min(k,d)+1):
                sum__ += comb(n[l],j) * num_hyperplanes(l+1, min(n[l]-j,d,k))
            sum_ += prob(k,n[l-1],n[l],p_sep[l-1]) * sum__
    dynamic_hyperplanes[tuple([l,d])] = sum_
    return sum_

def prob(rank, rows, columns, p, sample_factor = 48):
    min_dimension = min(rows,columns)
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
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    
    # Parse the argument
    args = parser.parse_args()
    n = ast.literal_eval(args.n)
    sp = ast.literal_eval(args.p)
    name = args.name
    
    model_arch = args.model
    dataset = args.dataset
    strategy = "LayerMagWeight"
    model_layers = n[1:]
    sample_points = 3
    method = 'exact'
    model_num = 1
    csv_name = 'testing123'
    times = 1

    in_size = n[0]
    L = len(n) - 1
    dct = {}
    result = {}
    old = []
    new = []
    for p in sp:
        # find local sparsity arrangments
        dims = []
        for i in range(len(n) - 2):
            dims.append(n[i] * n[i+1])
        keep = sum(dims) * p
        model = gb.Model()
        model.setParam('OutputFlag', 0)
        p_sep = model.addVars(len(dims), ub = 1, lb = 0)
        model.addConstr(keep == gb.quicksum(dims[i] * p_sep[i] for i in range(len(dims))))
        model.setObjective(p_sep[0], GRB.MINIMIZE)
        model.optimize()
        min_p = model.objVal
        model = gb.Model()
        model.setParam('OutputFlag', 0)
        p_sep = model.addVars(len(dims), ub = 1, lb = 0)
        model.addConstr(keep == gb.quicksum(dims[i] * p_sep[i] for i in range(len(dims))))
        model.setObjective(p_sep[0], GRB.MAXIMIZE)
        model.optimize()
        max_p = model.objVal
        p0_value = np.linspace(min_p, max_p, num=12)
        p1_value = []
        for p0 in p0_value:
            model = gb.Model()
            model.setParam('OutputFlag', 0)
            p_sep = model.addVars(len(dims), ub = 1, lb = 0)
            model.addConstr(keep == gb.quicksum(dims[i] * p_sep[i] for i in range(len(dims))))
            model.addConstr(p_sep[0] == p0)
            model.setObjective(p_sep[1], GRB.MAXIMIZE)
            model.optimize()
            p1_value.append(model.ObjVal)
        p_value = list(zip(p0_value, np.array(p1_value)))
        p_value = p_value[1: len(p_value)-1]
        p_value.sort()

        num_hyperplane = {}
        for ps in p_value:
            dynamic_hyperplanes = {}
            p_sep = [ps[0], ps[1], 1]
            num_hyperplane[ps] = num_hyperplanes(1, in_size) 
        val = list(num_hyperplane.values())
        
        if dataset == "MNIST":
            compressions = [[1/p, 1/p], [1/p_value[np.argmax(val)][0], 1/p_value[np.argmax(val)][1]]]
        elif dataset == "CIFAR10":
            compressions = [[1,1,1/p, 1/p], [1,1,1/p_value[np.argmax(val)][0], 1/p_value[np.argmax(val)][1]]]
        
        acc_lst = []
        acc_temp = []
        for seed in range(times):
            a= newExperiment(model_arch, dataset, strategy, compressions, model_layers, csv_name, sample_points, method, model_num, in_size, seed=seed)
            acc_temp.append(a)
        for c in range(len(acc_temp[0])):
            sum_a = 0
            for seed in range(times):
                sum_a += acc_temp[seed][c]
            sum_a = sum_a / times
            acc_lst.append(sum_a)
        
        dct[str(p) + "local_sparsity"] = p_value
        dct[str(p) + "upper_bound"] = val
        old.append(acc_lst[0])
        new.append(acc_lst[1])
        print("finish " + str(p))  
        df = pd.DataFrame(dct) 
        df.to_csv(name + '_ub.csv') 
    df = pd.DataFrame(dct) 
    df.to_csv(name + '_ub.csv') 
    result["sparsity"] = sp
    result["old"] = old
    result["new"] = new
    df = pd.DataFrame(result)
    df.to_csv(name + '_acc.csv')