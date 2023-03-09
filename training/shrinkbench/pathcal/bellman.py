import numpy as np
from .layers import * 

def importances_to_weights(importances):
    weights = []
    for layer in importances.keys():
        if isLinear(layer):
            weights += [importances[layer]['weight']]
    return weights

def mask_list_to_dict(importances, mask):
    ret_mask = {}
    i = 0
    for layer in importances.keys():
        if isLinear(layer):
#             ret_mask[layer] = {'weight':mask[i],'bias':np.ones_like(importances[layer]['bias'])}
            ret_mask[layer] = {'weight':mask[i]}
            i += 1
    return ret_mask

def longest_new_run(importances, fraction):
    # Convert to list
    weights = importances_to_weights(importances)
    # Number of layers
    NUM_LAYER = len(weights)
    # Max length of weight
    MAX_LEN = max([l.shape[-1] for l in weights])
    # Store the longest weight upon current level and node
    longest = np.ones((NUM_LAYER+1,MAX_LEN)) * -1
    longest[NUM_LAYER,:] = 1
    # Store the longes path upon current level and node
    longest_path = longest.tolist()
    longest_path[NUM_LAYER] = [[(NUM_LAYER,i,0)] for i in range(MAX_LEN)]
    # Mask of the weight
    mask = []
    mask_count = 0
    mask_size = 0
    for l in weights:
        mask += [np.zeros_like(l)]
        mask_size += l.flatten().shape[0]
        
    finished = False
    while (mask_count+1)/mask_size < fraction and not finished:
        print(f'{(mask_count)/mask_size*100}%')
        for j in range(weights[0].shape[-1]):
            m, path = longest_new(weights, 0, j, longest, longest_path, mask, new=True) # when current level is full cannot find a new path
            if m == -1:
                finished=True
                break
            for L, v, i in path[:-1]:
                if (mask_count+1)/mask_size < fraction:
                    # Check the repetition
                    if mask[L][i,v] == 0:
                        mask[L][i,v] = 1
                        mask_count += 1
                else:
                    finished = True
                    break
      #print(f'Longest path for node {j} is {m:.7f} with path {path}')
    print(f'Final Percentage Keep : {mask_count/mask_size*100}%')
    return mask_list_to_dict(importances, mask)

def longest_new(layers, L, v, longest, longest_path, mask, new=False):
    if longest[L,v] != -1 and new == False:
        # if there is max value calculated and not new path required
        # we return the value and path from history
        return longest[L,v], longest_path[L][v]
    weights = layers[L][:,v]
    m = -1
    arc = -1
    path = []
    p = []
    if L == len(layers) - 1 and new:
        # if reach top and still need new
        # fetch the max unused arcs
        for i in range(len(weights)):
            # we need to ensure the current arc is unused and 
            # check for the max weight in unused arcs.
            if mask[L][i,v] == 0 and weights[i] > m:
                m = weights[i]
                arc = (L,v,i)
                path = [(L,v,i)]
                p = longest_path[L+1][v]
        # if nothing is found 
        # we return -1 on weights to make sure this arc is not used
        if m == -1: return m, path
    elif new:
        for i in range(len(weights)):
            # picking a new arc in this layer
            if mask[L][i,v] == 0:
                cur_m, cur_path = longest_new(layers, L+1, i, longest, longest_path, mask, new=False)
          # picking a new arc in later layer
            else:
                cur_m, cur_path = longest_new(layers, L+1, i, longest, longest_path, mask, new=True)
            cur = weights[i] * cur_m
            if cur > m:
                m = cur
                arc = (L,v,i)
                path = [(L,v,i)]
                p = cur_path
    else:
        # Iterater through all arcs
        for i in range(len(weights)):
            cur_m, cur_path = longest_new(layers, L+1, i, longest, longest_path, mask, new=False)
            cur = weights[i] * cur_m
            # print(cur,m,cur_path)
            if cur > m:
                m = cur
                arc = (L,v,i)
                path = [(L,v,i)]
                p = cur_path
    path += p
    # There are some cases where m is not the max value return
    # We do not replace the max value in longest
    if m > longest[L,v]:
        longest[L,v] = m
        longest_path[L][v] = path
    return m, path

def longest_botup_run(importances, fraction):
    # Convert to list
    weights = importances_to_weights(importances)
    # Mask of the weight
    mask = []
    mask_count = 0
    mask_size = 0
    for l in weights:
        mask += [np.zeros_like(l)]
        mask_size += l.flatten().shape[0]
    
    finished = False
    while (mask_count + 1)/mask_size < fraction and not finished:
        path = longest_botup(weights)
        # print(weights)
        for layer in range(len(path.keys())-1):
            # print(mask[layer].shape)
            for idx, argm in enumerate(path[layer]):
                # print(argm, idx)
                if mask[layer][argm, idx] == 0:
                    mask[layer][argm, idx] = 1
                    mask_count += 1
                if (mask_count + 1)/mask_size >= fraction:
                    finished = True
                    break
            weights[layer] *= (mask[layer] == 0).astype(int) # Invert mask
        print(f'Percentage keep {mask_count/mask_size*100:.5f}%')
    print(f'Final Percentage keep {mask_count/mask_size*100:.5f}%')
    return mask_list_to_dict(importances, mask)

def longest_botup(layers):
    longest = {len(layers):np.ones((layers[-1].shape[0],1))}
    longest_path = {len(layers):np.array(range(len(layers[-1])))}
    for layer_idx in range(len(layers)-1,-1,-1):
        layer = layers[layer_idx]
        prev_longest = np.tile(longest[layer_idx + 1], layer.shape[-1])
        cur_longest = (prev_longest + layer).max(axis=0,keepdims=True).T
        
        cur_longest_path = (prev_longest + layer).argmax(axis=0)
        # print(cur_longest.shape)
        # print(cur_longest_path.shape)
        longest[layer_idx] = cur_longest
        longest_path[layer_idx] = cur_longest_path
    # print(longest)
    # print(longest_path)
    return longest_path