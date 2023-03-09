#Usage: python all.py --n "[784,10,10,10]" --p "[0.075, 0.05, 0.025]" --name "cifar800" --model "Perceptron10" --dataset "MNIST"

import numpy as np
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
from ast import literal_eval
import math
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial
os.environ['MKL_THREADING_LAYER'] = 'GNU'
import decimal

dict_comb = {}
def comb_efficient(n,m,exact=True):
    if tuple([n,m]) not in dict_comb:
        dict_comb[tuple([n,m])] = decimal.Decimal(comb(n,m, exact=exact))
    return dict_comb[tuple([n,m])]

def prob(k, C,R, p):
    if tuple([k,C,R,p]) in memorized_prob:
        return memorized_prob[tuple([k,C,R,p])]
    if C >= R:
        if R < k or  k == 0:
            memorized_prob[tuple([k,C,R,p])] = 0
            return 0
        Q = decimal.Decimal(1 - (decimal.Decimal(1 - p)**decimal.Decimal(C)))
        if R == k:
            memorized_prob[tuple([k,C,R,p])] = decimal.Decimal(Q)**decimal.Decimal(k)
            return memorized_prob[tuple([k,C,R,p])]
        memorized_prob[tuple([k,C,R,p])] = (comb_efficient(R,k, exact=True) * decimal.Decimal(Q)**decimal.Decimal(k) * decimal.Decimal(1 - Q) ** decimal.Decimal(R-k))
        return memorized_prob[tuple([k,C,R,p])]
    else:
        if C < k or k == 0:
            memorized_prob[tuple([k,C,R,p])] = 0
            return 0
        Q = decimal.Decimal(1 - decimal.Decimal(1 - p)**decimal.Decimal(R))
        if C == k:
            memorized_prob[tuple([k,C,R,p])] = decimal.Decimal(Q)**decimal.Decimal(k)
            return memorized_prob[tuple([k,C,R,p])]
        memorized_prob[tuple([k,C,R,p])] = (comb_efficient(C,k, exact=True) * decimal.Decimal(Q)**decimal.Decimal(k) * decimal.Decimal(1 - Q) ** decimal.Decimal(C-k))
        return memorized_prob[tuple([k,C,R,p])]
   
def num_hyperplanes(l,d):
    global N_comb
    if tuple([l,d]) in dynamic_hyperplanes:
        return dynamic_hyperplanes[tuple([l,d])]
    if l == L:
        sum_ = 0
        for k in range(0,n[l]+1):
            p_dis = prob(k,n[l-1],n[l],p_sep[l-1])
            com = decimal.Decimal(sum(comb_efficient(n[l],j, exact=True) for j in range(min(k,d)+1)))
            sum_ += p_dis * com
    else:
        sum_ = 0
        for k in range(0,n[l]+1):
            sum__ = 0
            for j in range(0, min(k,d)+1):
                sum__ += comb_efficient(n[l],j, exact=True) * num_hyperplanes(l+1, min(n[l]-j,d,k))
            sum_ += prob(k,n[l-1],n[l],p_sep[l-1]) * sum__
    dynamic_hyperplanes[tuple([l,d])] = sum_
    return sum_

def ss0(s):
    return literal_eval(s)[0]
def ss1(s):
    return literal_eval(s)[1]

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
    n = literal_eval(args.n)
    sp = literal_eval(args.p)
    name = args.name
    model_arch = args.model
    dataset = args.dataset

    in_size = n[0]
    L = len(n) - 1
    dims = []
    for i in range(len(n) - 2):
        dims.append(n[i] * n[i+1])
    dct = {}    
    sparsity_U = [(1,1)]
    sparsity_O = [(1,1)]
    p_sep = [1,1,1]
    dynamic_hyperplanes = {}
    memorized_prob = {}
    x = num_hyperplanes(1, in_size)
    x = float(x.ln())
    ub_U = [x]
    ub_O = [x]
    
    for p in sp:
        keep = sum(dims) * p
        min_p0 = max(keep - dims[1], 0) / dims[0]
        min_p1 = (keep - min_p0 * dims[0]) / dims[1]
        max_p1 = max(keep - dims[0], 0) / dims[1]
        max_p0 = (keep - max_p1 * dims[1]) / dims[0]
        
        
        # model = gb.Model()
        # model.setParam('OutputFlag', 0)
        # p_sep = model.addVars(len(dims), ub = 1, lb = 0)
        # model.addConstr(keep == gb.quicksum(dims[i] * p_sep[i] for i in range(len(dims))))
        # model.setObjective(p_sep[0], GRB.MINIMIZE)
        # model.optimize()
        # min_p = model.objVal
        # model = gb.Model()
        # model.setParam('OutputFlag', 0)
        # p_sep = model.addVars(len(dims), ub = 1, lb = 0)
        # model.addConstr(keep == gb.quicksum(dims[i] * p_sep[i] for i in range(len(dims))))
        # model.setObjective(p_sep[0], GRB.MAXIMIZE)
        # model.optimize()
        # max_p = model.objVal
        p0_value = np.linspace(min_p0, max_p0, num=12)
        # p1_value = []
        # for p0 in p0_value:
        #     model = gb.Model()
        #     model.setParam('OutputFlag', 0)
        #     p_sep = model.addVars(len(dims), ub = 1, lb = 0)
        #     model.addConstr(keep == gb.quicksum(dims[i] * p_sep[i] for i in range(len(dims))))
        #     model.addConstr(p_sep[0] == p0)
        #     model.setObjective(p_sep[1], GRB.MAXIMIZE)
        #     model.optimize()
        #     p1_value.append(model.ObjVal)
        # p_value = list(zip(p0_value, np.array(p1_value)))
        # p_value.sort()
        # low = p_value[0]
        # high = p_value[-1]
        low = [p0_value[1], (keep - p0_value[1] * dims[0]) / dims[1]]
        high = [p0_value[-2], (keep - p0_value[-2] * dims[0]) / dims[1]]
        mid = [p,p]
        dynamic_hyperplanes = {}
        memorized_prob = {}
        p_sep = [low[0], low[1], 1]
        low_ub = num_hyperplanes(1, in_size)
        low_ub = float(low_ub.ln())
        dynamic_hyperplanes = {}
        memorized_prob = {}
        p_sep = [high[0], high[1], 1]
        high_ub = num_hyperplanes(1, in_size)
        high_ub = float(high_ub.ln())
        dynamic_hyperplanes = {}
        memorized_prob = {}
        p_sep = [mid[0], mid[1], 1]
        mid_ub = num_hyperplanes(1, in_size)
        mid_ub = float(mid_ub.ln())
        print(p)
        x_list = [low[0], mid[0],high[0]]
        y_list = [low_ub, mid_ub, high_ub]
        print(x_list, y_list)
        
        poly = lagrange(x_list, y_list)
        d = poly.deriv()
        root = d.roots
        if len(root) != 0:
            if root[0] != 0.0:
                p_new = d.roots[0]
                if p_new < x_list[-1] and p_new > x_list[0]:
                    keep = sum(dims) * p
                    p_new2 = (keep - p_new * dims[0]) / dims[1]
                    dynamic_hyperplanes = {}
                    memorized_prob = {}
                    p_sep = [p_new, p_new2, 1]
                    v = num_hyperplanes(1, in_size) 
                    v = float(v.ln())
                    if v > mid_ub:
                        sparsity_O.append((p_new,p_new2))
                        ub_O.append(v)
                    else:
                        sparsity_O.append(mid)
                        ub_O.append(mid_ub)
                else:
                    sparsity_O.append(mid)
                    ub_O.append(mid_ub)
            else:
                sparsity_O.append(mid)
                ub_O.append(mid_ub)
        else:
            sparsity_O.append(mid)
            ub_O.append(mid_ub)
        sparsity_U.append(mid)
        ub_U.append(mid_ub)
    dct["sparsity_U"] = sparsity_U
    dct["ub_U"] = ub_U
    dct["sparsity_O"] = sparsity_O
    dct["ub_O"] = ub_O
    df = pd.DataFrame(dct)
    df.to_csv(name + '_ub_new.csv')
    
    name = args.name + "_ub_new.csv"
    df = pd.read_csv(name)
    
    model_arch = args.model
    dataset = args.dataset
    strategy = "LayerMagWeight"
    model_layers = n[1:]
    sample_points = 3
    method = 'exact'
    model_num = 1
    csv_name = 'testing123'
    times = 10
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
    if model_arch == "LeNet" or model_arch == "LeNet_C10":
        q = [[1,1,1,1]]
    elif model_arch == "AlexNet":
        q = [[1,1,1,1,1,1,1,1]]
    else:
        q = [[1,1]]
        
    for seed in range(times):
        
        a= newExperiment(model_arch, dataset, strategy, q, model_layers, csv_name, sample_points, method, model_num, in_size, seed=seed)
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
    count = 1
    
    for p in sp:
        sparsity = df["sparsity_O"].loc[count]
        count += 1
        if model_arch == "LeNet" or model_arch == "LeNet_C10":
            compressions = [[1,1,1/p, 1/p], [1,1,1/literal_eval(sparsity)[0], 1/literal_eval(sparsity)[1]]]
        elif model_arch == "AlexNet":
            compressions = [[1,1,1,1,1,1/p,1/p,1], [1,1,1,1,1,1/literal_eval(sparsity)[0], 1/literal_eval(sparsity)[1],1]]
        else:
            compressions = [[1/p, 1/p], [1/literal_eval(sparsity)[0], 1/literal_eval(sparsity)[1]]]
        
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
    result["sparsity_U"] = df['sparsity_U']
    result["UB_U"] = df["ub_U"]
    result["ACC_U"] = old
    result["sparsity_O"] = df['sparsity_O']
    result["UB_O"] = df["ub_O"]
    result["ACC_O"] = new
    df = pd.DataFrame(result)
    df["sparsity1_U"] = df["sparsity_U"].apply(ss0)
    df["sparsity2_U"] = df["sparsity_U"].apply(ss1)
    df["sparsity1_O"] = df["sparsity_O"].apply(ss0)
    df["sparsity2_O"] = df["sparsity_O"].apply(ss1)
    df.index.name = 'number'
    df = df[["global_sparsity", "sparsity1_U", "sparsity2_U", "UB_U", "ACC_U", "sparsity1_O", "sparsity2_O", "UB_O", "ACC_O"]]
    df.to_csv(args.name + '.csv')
    df = pd.DataFrame(log)
    df.to_csv(args.name + '_acc.csv')
