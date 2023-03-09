#python all.py --n "[784,100,100,10]" --p "[0.01, 0.005, 0.0025, 0.001, 0.00075, 0.0005, 0.0001]" --name "mnist100" --model "Perceptron100" --dataset "MNIST"

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

def ss0(s):
    return literal_eval(s)[0]
def ss1(s):
    return literal_eval(s)[1]

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
    x = Fraction(num_hyperplanes(1, in_size))
    x = math.log(x.numerator,2) - math.log(x.denominator,2)
    ub_U = [x]
    ub_O = [x]
    
    for p in sp:
        keep = sum(dims) * p
        min_p0 = max(keep - dims[1], 0) / dims[0]
        min_p1 = (keep - min_p0 * dims[0]) / dims[1]
        max_p1 = max(keep - dims[0], 0) / dims[1]
        max_p0 = (keep - max_p1 * dims[1]) / dims[0]
        
        low = [min_p0, min_p1]
        high = [max_p0, max_p1]
        mid = [p,p]
        dynamic_hyperplanes = {}
        p_sep = [low[0], low[1], 1]
        low_ub = Fraction(num_hyperplanes(1, in_size))
        dynamic_hyperplanes = {}
        p_sep = [high[0], high[1], 1]
        high_ub = Fraction(num_hyperplanes(1, in_size))
        dynamic_hyperplanes = {}
        p_sep = [mid[0], mid[1], 1]
        mid_ub = Fraction(num_hyperplanes(1, in_size))
        low_ub = math.log(low_ub.numerator,2) - math.log(low_ub.denominator,2)
        high_ub = math.log(high_ub.numerator,2) - math.log(high_ub.denominator,2)
        mid_ub = math.log(mid_ub.numerator,2) - math.log(mid_ub.denominator,2)
        
        x_list = [low[0], mid[0], high[0]]
        y_list = [low_ub, mid_ub, high_ub]
        
        poly = lagrange(x_list, y_list)
        d = poly.deriv()
        root = d.roots
        flag = 0
        if len(root) != 0:
            if root[0] != 0.0:
                p_new = d.roots[0]
                if p_new < x_list[-1] and p_new > x_list[0]:
                    keep = sum(dims) * p
                    p_new2 = (keep - p_new * dims[0]) / dims[1]
                    num_hyperplane = {}
                    dynamic_hyperplanes = {}
                    p_sep = [p_new, p_new2, 1]
                    v = num_hyperplanes(1, in_size) 
                    v = math.log(v.numerator,2) - math.log(v.denominator,2)
                    if v > mid_ub and p_new < p:
                        sparsity_O.append((p_new,p_new2))
                        ub_O.append(v)
                    else:
                        flag = 1
                else:
                    flag = 1
            else:
                flag = 1
        else:
            flag = 1
        sparsity_U.append(mid)
        ub_U.append(mid_ub)
        if flag == 1:
            good_u = mid_ub
            good_s = mid 
            spp = np.linspace(min_p0, p, num=30)
            for i in range(30):
                p1 = (keep - spp[i] * dims[0]) / dims[1]
                if p1 > 0:
                    num_hyperplane = {}
                    dynamic_hyperplanes = {}
                    p_sep = [spp[i], p1, 1]
                    ubo = num_hyperplanes(1, in_size)
                    ubo = math.log(ubo.numerator,2) - math.log(ubo.denominator,2)
                    if ubo > good_u:
                        good_u = ubo
                        good_s = (spp[i], p1)
            sparsity_O.append(good_s)
            ub_O.append(good_u)
                
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
    times = 1
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