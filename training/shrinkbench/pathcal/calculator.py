import torch
import numpy as np
# from sklearn import preprocessing
# from .loader import *
from .layers import * 
from .helper import *
"""
===============================================================
====              Calculate Flows Using Weights            ====
===============================================================
"""
def calc_flows(importances):
    prev_layer = None
    prev_flows = None
    for layer in reversed(list(importances.keys())):
        weights = importances[layer]['weight']
        #print(np.amax(weights),np.amin(weights))
        # Calculate the layer based on layer types
        # Only consider conv and linear currently
        if isConv(layer):
            # Get current weights
            if isLinear(prev_layer):
                prev_flows = shrink_flows(prev_flows, weights)
            cur_path, weights = cal_conv_flows(prev_flows, weights)
            
            prev_layer = layer
            prev_flows = cur_path
        elif isLinear(layer):
            # If prev layer is conv requires reshape
            cur_flows = cal_linear_flows(prev_flows, weights)
            if prev_flows is not None:
                weights = np.tile(prev_flows, (weights.shape[1],)).reshape(weights.shape) * weights
            prev_layer = layer
            prev_flows = cur_flows
        
#         scaler = preprocessing.Normalizer()
        new_weights = weights.reshape(len(weights),-1)
        new_weights = scaler.fit_transform(new_weights)
        importances[layer]['weight'] = new_weights.reshape(weights.shape)
        #print(np.amax(new_weights),np.amin(new_weights))
        
    return importances

def cal_linear_flows(prev_flows, weights, threshold=0):
    """
    This method is used for calculate flows for linear layer.
    param:
    prev_flows: the outputing flows from previous layer 
    weights: the weights connecting previous layer to current layer
    threshold: not currently used in this function
    """
    # duplicate and reshape to match the current weights shape for easier calculation
    if prev_flows is None:
        next_flows = weights
    else:
        next_flows = np.tile(prev_flows, (weights.shape[1],)).reshape(weights.shape) * weights
    #next_flows[weights.abs() <= threshold] = 0
    # Sum over all the flows connect to current node
    next_flows = next_flows.sum(axis=0)
    # print(next_flows)
    return next_flows

def cal_conv_flows(prev_flows, weights, threshold=0):
    """
    This method is used for calculate flows for conv layer.
    param:
    prev_flows: the outputing flows from previous layer 
    weights: the weights connecting previous layer to current layer
    threshold: not currently used in this function
    """
    if prev_flows is not None:
        new_weights = weights * prev_flows[:, None, None, None]
    else:
        new_weights = weights
    
    # Calculate next flowss
    weights = np.swapaxes(weights,0,1)
    next_flows = new_weights.reshape(len(weights),-1)
    next_flows = next_flows.sum(axis=-1)
    return next_flows, new_weights

def expend_flows(prev_flows, weights):
    """
    This method is used for reshape the flows when chaning from conv layer to linear layer
    Here we duplicate the flows from conv to match the input shape of linear layer
    param:
    prev_flows: the outputing flows from previous layer 
    weights: the weights connecting previous layer to current layer
    """
    in_shape = weights.shape[-1]
    flow_mat = np.zeros((in_shape // len(prev_flows) , len(prev_flows)))
    flow_mat[:,:] = prev_flows
    
    return flow_mat.T.flatten()

def shrink_flows(prev_flows, weights):
    prev_flows = prev_flows.reshape((weights.shape[0],-1))
    prev_flows = prev_flows.sum(axis=-1).T
    #
    return prev_flows