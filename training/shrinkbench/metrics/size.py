"""Model size metrics
"""

import numpy as np
from . import nonzero, dtype2bits


def model_size(model, as_bits=False):
    """Returns absolute and nonzero model size

    Arguments:
        model {torch.nn.Module} -- Network to compute model size over

    Keyword Arguments:
        as_bits {bool} -- Whether to account for the size of dtype

    Returns:
        int -- Total number of weight & bias params
        int -- Out total_params exactly how many are nonzero
    """

    total_params = 0
    nonzero_params = 0
    for tensor in model.parameters():
        t = np.prod(tensor.shape)
        nz = nonzero(tensor.detach().cpu().numpy())
        if as_bits:
            bits = dtype2bits[tensor.dtype]
            t *= bits
            nz *= bits
        total_params += t
        nonzero_params += nz
    return int(total_params), int(nonzero_params)

def layer_size(tensor, as_bits=False):
    """
    Returns absolute and nonzero module size

    Arguments:
        module -- Network module to compute module size over

    Keyword Arguments:
        as_bits {bool} -- Whether to account for the size of dtype

    Returns:
        int -- Total number of weight & bias params
        int -- Out total_params exactly how many are nonzero
    """
    total_params = 0
    nonzero_params = 0
    t = np.prod(tensor.shape)
    nz = nonzero(tensor.detach().cpu().numpy())
    if as_bits:
        bits = dtype2bits[tensor.dtype]
        t *= bits
        nz *= bits
    total_params += t
    nonzero_params += nz
    return int(total_params), int(nonzero_params)
    
def layerwise_sparsity(model):
    '''
    Returns a string instead of list for easy insert into pd.Datafram
    '''
    sparsityL = []
    for i, tensor in enumerate(model.parameters()):
        if i%2==1:
            continue
        total_size, non_zeros = layer_size(tensor)
        sparsityL.append((non_zeros / total_size)*100)
#         print(f"layer {int((i/200)*100)} {(non_zeros / total_size)*100}%")
        # Fancy maths to convert from 0, 2, 4 to 0, 1, 2 pattern
    sparsity = ""
    for x in sparsityL:
        sparsity += f"{x:.2f}, "
    return sparsity
    