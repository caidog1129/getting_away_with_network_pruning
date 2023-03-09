"""Random  pruning

Implements pruning strategy that randomly prunes weights.
It is intended as a baseline for pruning evalution
"""

import numpy as np
from ..pruning import VisionPruning, LayerPruning
from .utils import map_importances, fraction_mask


def random_mask(tensor, fraction):
    idx = np.random.uniform(0, 1, size=tensor.shape) > fraction
    mask = np.ones_like(tensor)
    mask[idx] = 0.0
    return mask

# The RP used in the Paper
class RandomPruning(VisionPruning):

    def model_masks(self):
        params = self.params()
        # print("Params:", params)
        masks = map_importances(lambda x: random_mask(x, self.fraction), params)
        return masks

class LayerRandom(LayerPruning, VisionPruning):

    def layer_masks(self, module, layer):
        params = self.module_params(module)
        masks = {param: random_mask(value, self.fraction[layer]) 
                 for param, value in params.items() if value is not None and param != 'bias'}
        return masks

    