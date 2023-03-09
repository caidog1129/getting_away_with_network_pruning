"""Magnitude based pruning

Implements pruning strategy that prunes as follows

For a given tensor, sort by absolute value and
keep only the _fraction_ with highest magnitudes
so that overall desired compression is achieved
"""

import numpy as np

from ..pathcal.calculator import *
from ..pathcal import bellman

from ..pruning import (LayerPruning,
                       VisionPruning,
                       GradientMixin,
                       ActivationMixin)
from .utils import (fraction_threshold,
                    fraction_mask,
                    map_importances,
                    map_importances_bias,
                    flatten_importances,
                    importance_masks,
                    importance_masks_bias,
                    activation_importance)

# The MP used in the paper
class GlobalMagWeight(VisionPruning):
    ''' Does not prune bias '''
    def model_masks(self):
        importances = map_importances(np.abs, self.params())
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks
    
class GlobalMagBias(VisionPruning):
    ''' Prunes the bias '''
    def model_masks(self):
        importances = map_importances_bias(np.abs, self.params())
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks_bias(importances, threshold)
        return masks

    
class FlowMagWeight(VisionPruning):
    def model_masks(self):
        importances = map_importances(np.abs, self.params())
        importances = calc_flows(importances)
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks

class FlowMagBias(VisionPruning):

    def model_masks(self):
        importances = map_importances_bias(np.abs, self.params())
        importances = calc_flows(importances)
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagWeight(LayerPruning, VisionPruning):

    def layer_masks(self, module, layer):
        params = self.module_params(module)
        importances = {param: np.abs(value) for param, value in params.items() if param != 'bias'}
        masks = {param: fraction_mask(importances[param], self.fraction[layer])
                 for param, value in params.items() if value is not None and param != 'bias'}
        return masks

# The GP used in the paper
class GlobalMagGrad(GradientMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        grads = self.param_gradients()
        importances = {mod:
                       {p: np.abs(params[mod][p]*grads[mod][p])
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks
    
class MixedMagGrad2(GradientMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        grads = self.param_gradients()
        importances = {mod:
                       {p: (-(params[mod][p])*(grads[mod][p]))+ 2 *(5e-4)*(pow((params[mod][p]),2))
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks

# The XP used in the Paper
class MixedMagGrad(GradientMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        grads = self.param_gradients()
        importances = {mod:
                       {p: (-(params[mod][p])*(grads[mod][p]) )+ (5e-4)*(pow((params[mod][p]),2))
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks
    
class MixedMagGrad3(GradientMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        grads = self.param_gradients()
        importances = {mod:
                       {p: np.abs((-(params[mod][p])*(grads[mod][p]) )+ (5e-4)*(pow((params[mod][p]),2)))
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks

class MixedMagGrad4(GradientMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        grads = self.param_gradients()
        importances = {mod:
                       {p: np.abs(((params[mod][p])*(grads[mod][p]) ))+ (5e-4)*(pow((params[mod][p]),2))
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks



    
class Bellman(VisionPruning):
    """
    Bellman's equations
    """
    def model_masks(self):
        importances = map_importances(np.abs, self.params())
        # Bottom up algorithm
        # masks = bellman.longest_botup_run(importances, self.fraction)
        # Top down new
        masks = bellman.longest_new_run(importances, self.fraction)
        for layer in list(masks.keys()):
            weights = masks[layer]['weight'].flatten()
            im = importances[layer]['weight'].flatten()
            print(f'layer: {layer}\n Pruned {(weights == 0).sum()/weights.shape[0]*100:.2f}%')
        return masks

    
class LayerMagGrad(GradientMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        grads = self.module_param_gradients(module)
        importances = {param: np.abs(value*grads[param]) for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks


class GlobalMagAct(ActivationMixin, VisionPruning):

    def model_masks(self):
        params = self.params()
        activations = self.activations()
        # [0] is input activation
        importances = {mod:
                       {p: np.abs(activation_importance(params[mod][p], activations[mod][0]))
                        for p in mod_params}
                       for mod, mod_params in params.items()}
        flat_importances = flatten_importances(importances)
        threshold = fraction_threshold(flat_importances, self.fraction)
        masks = importance_masks(importances, threshold)
        return masks


class LayerMagAct(ActivationMixin, LayerPruning, VisionPruning):

    def layer_masks(self, module):
        params = self.module_params(module)
        input_act, _ = self.module_activations(module)
        importances = {param: np.abs(activation_importance(value, input_act))
                       for param, value in params.items()}
        masks = {param: fraction_mask(importances[param], self.fraction)
                 for param, value in params.items() if value is not None}
        return masks
