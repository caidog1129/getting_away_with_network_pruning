"""
    Update the weights for the network after pruning
    A. For stably inactive neurons, remove the corresponding weights before and after them.
    B. For stably active neurons, remove the weights before and update the weights and bias:
    We have the following notations:
        0. h1_a: [c1_a]
                 the stably active neurons
        1. w1_a: [c1_a, c0] 
                 the weights before the active neurons. 
           w1_a = {w11_a, w12_a}, whose rank is r1 and r1 < c1_a 
           w11_a: [r1, c0] 
                  the weights before h11_a;
                  the r1 row vectors which can represent the base in R^r1
           w12_a: [c1_a - r1, c0]
                  the weights before h12_a;
                  the rest row vectors can be presented as a linear combination of the w11_a
        2. w12_a = K1*w11_a, 
           K1: [c1_a - r1, r1]
        3. w2_a:  [c2, c1_a] 
                  the weights after the active neurons.
           w2_a = {w21_a, w22_a}
           w21_a: [c2, r1]
                  the weights after h11_a;
           w22_a: [c2, c1_a - r1]
                  the weights after h12_a;
        4. b1_a = {b11_a, b12_a}
           b11_a: [r1, ]
           b12_a: [c1_a - r1, ]
    We update the weights before and after h1_a as below:
        1. remove w12_a from w1_a
        2. merge w22_a into w21_a:
           w21_a' = w21_a + w22_a * K1
        3. update b2:
           b2'    = b2 + w22_a * (b12_a - K1 * b11_a)
        4. remove w22_a
        
"""
import os
import sys

import numpy as np
from numpy.linalg import matrix_rank, norm
import torch
#import sympy

from common.timer import Timer
from dir_lookup import *

DEBUG = False
timer = Timer()

#######################################################################################
# remove the corresponding weights before and after the stably inactive neurons
#######################################################################################
def prune_inactive_per_layer(w1, w2, ind_inact):
    """
    w1: [c1, c0] the weights before the current layer
    w2: [c2, c1] the weithgs after the current layer
    ind_inact: a list of the index of the stably inactive neurons
    """
    w1 = np.delete(w1, ind_inact, axis=0)
    w2 = np.delete(w2, ind_inact, axis=1)
    return w1, w2


def sanity_ckp():
    w1 = np.array(
            [[1,0,2,1],
             [1,1,0,0],
             [2,1,2,1],
             [1,2,3,0],
             [11,12,13,0]])
    b1 = np.arange(5)
    w2 = np.arange(10).reshape(2,5)
    b2 = np.array([11,12])
    act_neurons = np.array([[1, 0], [1,1], [1,2]]).T
    inact_neurons = np.array([[1,4]]).T
    w_names = ['w1', 'w2']
    b_names = ['b1', 'b2']
    return [w1, w2], [b1, b2], act_neurons, inact_neurons, w_names, b_names
   
def weights_bias_from_fcnn_ckp(ckp):
    w_names = sorted([name for name in ckp['state_dict'].keys() 
                            if 'weight' in name and 'features' in name])
    b_names = sorted([name for name in ckp['state_dict'].keys() 
                            if 'bias' in name and 'features' in name])
    w_names.append('classifier.0.weight')
    b_names.append('classifier.0.bias')

    device = ckp['state_dict'][w_names[0]].device

    weights = []
    bias    = []
    for name in w_names:
        weights.append(ckp['state_dict'][name].cpu().numpy())
    for name in b_names:
        bias.append(ckp['state_dict'][name].cpu().numpy())
    return w_names, b_names, weights, bias, device

#######################################################################################
# Load the weights and bias from the checkpoints and the neuron stability from the state files,
# Prune the model according to neuron stability
#   tag: the method to get sbable neurons, whoses results is under the folder
#        'results/${dataset}/${tag}' 
#        e.g:  results-no_preprocess, results-preprocess_all, 
#              results-old-approach, results-preprocess_partial
#######################################################################################
def prune_ckp(model_path,tag):
     
    if DEBUG:
        weights, bias, act_neurons, inact_neurons, w_names, b_names = sanity_ckp()
    else:
        timer.start()
        ckp_path = os.path.join(model_path, 'checkpoint_120.tar')
        pruned_ckp_path = os.path.join(model_path, 'pruned_checkpoint_120.tar')
        MILP_rst      = collect_rst(model_path, tag)
        #stb_path = os.path.join(model_path, 'stable_neurons.npy')
        #stb_path = get_stb_path(ALLPRE, model_path)
        stb_path = get_MILP_stb_path(tag, model_path)
        if not os.path.exists(ckp_path):
            print(ckp_path, 'not exists')
            return
        if not os.path.exists(stb_path):
            print(stb_path, 'not exists')
            return
        ckp = torch.load(ckp_path)
        w_names, b_names, weights, bias, device = weights_bias_from_fcnn_ckp(ckp)
        act_neurons, inact_neurons = read_MILP_stb(model_path, tag) 
        ##timer.stop('loaded the checkpoint')
   
    # Get a new model by applying lossless pruning. 
    #   Note this reduces the model size instead of masking pruned weights and biases as zeros
    pruned_numbers = ''
    all_pruned_ind = []
    rm_max = 0.0
    for l in range(1, len(weights)):
        if len(act_neurons) > 0:
            ind_act  = act_neurons[1,   act_neurons[0,:] == l]
        else:
            ind_act  = []
        if len(inact_neurons) > 0:
            ind_inact = inact_neurons[1, inact_neurons[0,:] == l]
        else:
            ind_inact = []
        w1 = weights[l-1]
        w2 = weights[l]
        b1 = bias[l-1]
        b2 = bias[l]
        prune_ind = []
        c1,c0 = w1.shape
        c2,c1 = w2.shape
        # get the index of stably active neurons to prune
        if len(ind_act) > 0:
            #import pdb;pdb.set_trace()
            w2, b2, prune_ind_act = prune_active_per_layer(w1, w2, b1, b2, ind_act)
            prune_ind.extend(prune_ind_act)
        else:
            prune_ind_act = []
        # all stably inactive neurons to prune
        if len(ind_inact) > 0:
            prune_ind.extend(ind_inact)
        # delete all the neurons to be pruned
        if len(prune_ind) > 0:
            w1_rm = w1[prune_ind,:].reshape(-1)
            w2_rm = w2[:, prune_ind].reshape(-1)
            w_rm_max = np.absolute(np.concatenate([w1_rm, w2_rm])).max()
            if w_rm_max > rm_max:
                rm_max = w_rm_max
            
            w1 = np.delete(w1, prune_ind, axis=0)
            b1 = np.delete(b1, prune_ind, axis=0)
            w2 = np.delete(w2, prune_ind, axis=1)
        print(f'Layer-{l}: prune {len(prune_ind_act)} stably active neurons')
        print(f'layer-{l}: prune {len(ind_inact)} stably inactive neurons')
        pruned_numbers += f'{len(prune_ind_act)}, {len(ind_inact)},,'
        # update the weights and bias
        weights[l-1] = w1
        bias[l-1]    = b1
        weights[l]   = w2
        bias[l]      = b2
        ##timer.stop(f'{l} layer is pruned')
        for ii in prune_ind:
            for jj1 in range(c0):
                ind = [l-1, ii, jj1]
                all_pruned_ind.append(ind)
            for jj2 in range(c2):
                ind = [l, jj2, ii]
                all_pruned_ind.append(ind)
    ckp['all_pruned_index'] = np.array(all_pruned_ind)
    # update the ckeckpoints
    for i, name in enumerate(w_names):
        ckp['state_dict'][name] = torch.from_numpy(weights[i]).cuda(device=device)
    for i, name in enumerate(b_names):
        ckp['state_dict'][name] = torch.from_numpy(bias[i]).cuda(device=device)
    # save the checkpoint 
    torch.save(ckp, pruned_ckp_path)
    
    # append the lossless pruning results into the stat file
    if 'NO RESULT' in MILP_rst[0]:
        _,arch,_,_ = parse_exp_name(model_path)
        MILP_rst = [model_path + ', , ,' + ' , , ,,' * len(arch.split('-')) + ' ,, , ,, ,,'] 
    MILP_rst_path = get_MILP_rst_path(tag, model_path)
    with open(MILP_rst_path, 'w') as f:
        for l in MILP_rst:
            f.write(l + pruned_numbers + '\n')  
    print(MILP_rst_path)
    return np.array(all_pruned_ind), rm_max

if __name__ == '__main__':
    #model_path = 'model_dir/CIFAR100-rgb/dnn_CIFAR100-rgb_400-400_7.500000000000001e-05_0001'
    #model_path = 'model_dir/CIFAR10-rgb/dnn_CIFAR10-rgb_400-400_0.000175_0002'
    model_name = os.path.basename(sys.argv[1])
    dataset = model_name.split('_')[1]
    model_path = os.path.join(model_root, dataset, sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == 'magnitude':
        #magnituded_based_prune_ckp(model_path, ALLPRE)
        diff_MP_LLC(model_path,ALLPRE)
    else:
        prune_ckp(model_path, ALLPRE)
