# AARLBench

Open source PyTorch library to analyse effect of neural network pruning methods on model recall and class intensification.
Modified from [Original Shrinkbench Git Repository](https://github.com/jjgo/shrinkbench)



# Installation

First, install the dependencies found in the requirements.txt file


## Modules

The modules are organized as follows:

| submodule | Description |
| ---- | ---- |
| `datasets/` | Standardized dataloaders for supported datasets |
| `experiment/` | Main experiment class with the data loading, pruning, finetuning & evaluation |
| `metrics/` | Utils for measuring accuracy, model size, flops & memory footprint |
| `models/` | Custom architectures not included in `torchvision` |
| `plot/` | Utils for plotting across the logged dimensions |
| `pruning/` | General pruning and masking API.  |
| `scripts/` | Executable scripts for running experiments (see `experiment/`) |
| `strategies/` | Baselines pruning methods, mainly magnitude pruning based |

