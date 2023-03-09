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
    n = [784, 100, 100, 100, 10]
    in_size = n[0]
    L = len(n) - 1
    dims = []
    for i in range(len(n) - 2):
        dims.append(n[i] * n[i+1])  
    

    dynamic_hyperplanes = {}
    memorized_prob = {}
    p_sep = [0.0005, 0.0005, 0.0005, 1]
    low_ub = num_hyperplanes(1, in_size)
    
        
    dynamic_hyperplanes = {}
    memorized_prob = {}
    p_sep = [0.0003, 0.000984, 0.000984, 1]
    low_ub2 = num_hyperplanes(1, in_size)
    
    
    
    model_arch = "Perceptron3_100"
    dataset = "MNIST"
    # strategy = "GlobalMagWeight"
    strategy = "LayerMagWeight"
    model_layers = n[1:]
    sample_points = 3
    method = 'exact'
    model_num = 1
    csv_name = 'testing123'
    times = 10
    in_size = n[0]
    
    lst = []
    sum = 0
    q = [[1/0.0005, 1/0.0005, 1/0.0005, 1]]
    # q = [1/0.05]
    for seed in range(times): 
        a= newExperiment(model_arch, dataset, strategy, q, model_layers, csv_name, sample_points, method, model_num, in_size, seed=seed+10)
        sum += a[0]
        lst.append(a)
    
    
    
    lst2 = []
    sum2 = 0
    q = [[1/0.0003, 1/0.000984, 1/0.000984, 1]]
    for seed in range(times): 
        a= newExperiment(model_arch, dataset, strategy, q, model_layers, csv_name, sample_points, method, model_num, in_size, seed=seed+10)
        sum2 += a[0]
        lst2.append(a)
    print(low_ub)
    print(low_ub2)
    # print(lst)
    # print(lst2)
    print(sum / times)
    print(sum2 / times)
