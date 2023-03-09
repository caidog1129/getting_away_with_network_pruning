from shrinkbench.experiment import PruningExperiment, PruningClass
import torch
import os

# ShrinkPATH is the path from the directory this file is located to the shrinkbench code
# DATAPATH is the path from the current directory to where the datasets are located
# The only think you might need to change is 'shrinkbench' to whatever the name of the file is that 
# contains the shrinkbench code
os.environ['ShrinkPATH'] = './shrinkbench'
os.environ['DATAPATH'] = './shrinkbench/Training_data'

# These are the compression ratios
compressions = [4, 10, 20, 30, 40, 50]
strategies = ["GlobalMagWeight"]

# These are the datasets and their corresponding nn
all_datasets = {'QMNIST': 'LeNet',
               'Fashion': 'LeNet',
               'CIFAR10': 'resnet56',
               'CIFAR100': 'resnet56_C'}
cur_dataset = 'QMNIST'

def load_expriment(dataset=cur_dataset):
    global cur_dataset
    if dataset not in all_datasets.keys():
        print(f"[ERROR] {dataset} does not exist.")
        return None
    cur_dataset = 'QMNIST'
    # This is the overarching object that is interacted with. 
    exp = PruningClass(dataset=dataset, # Change this to 'Fashion', 'CIFAR10', or 'CIFAR100' 
                    model=all_datasets[dataset],  # LeNetChange this to 'resnet56' for CIFAR10, or 'resnet56_C' for CIFAR100
                    train_kwargs={
                        'optim': 'SGD',
                        'epochs': 30,
                        'lr': 1e-2},
                    dl_kwargs={'batch_size':128},
                    save_freq=1)
    exp.run_init()  # Sets up some stuff, can ignore output
    # print(list(exp.model.conv1.named_parameters())) # List weights of first layer in LeNet
    return exp

"""
Load a trained model before prune/finetune.

This is specific for qmnist, but the naming scheme is the same for the other models
    example file: qmnist3.pt 
        * 'qmnist' is the dataset the model is for
        * '3' is the model number (0-9)
When passing the name to the load_model() function, do not include the .pt at the end
"""
def load_org_model(exp, model=1):
    global cur_dataset
    checkpoint = exp.load_model(cur_dataset.lower() + str(model))
    exp.build_model(all_datasets[cur_dataset]) # Change this when going between different model architectures. 
    exp.to_device()
    exp.model.load_state_dict(checkpoint['model_state_dict'])
    exp.optim.load_state_dict(checkpoint['optim_state_dict'])
    # print(list(exp.model.conv1.named_parameters())) # List weights of first layer in LeNet

"""
Load a pruned/finetuned model.

This is specific for qmnist, but the naming scheme is the same for the other models<br>
example file: qmnist3.c30.pt
'qmnist' is the dataset the model is for<
'3' is the model number (0-9)
c is whether it is after pruning or after finetuning. c for pruning and f for finetuning.
30 is what the compression ratio is. Can be 4, 10, 20, 30, 40, 50 for the CIFAR ones, or 4, 10, 20, 30, 40, 50 for <br>
the QMNIST/Fashion ones
When passing the name to the load_model() function, do not include the .pt at the end
"""
def load_model(exp, model=1, compression=4, stg="pruning"):
    global cur_dataset
    stg = "c" if stg == "pruning" else "f"
    model_name = cur_dataset.lower() + str(model) + '.' + stg + str(compression)
    print(model_name)
    checkpoint = exp.load_model(model_name)
    exp.build_model(all_datasets[cur_dataset]) # Change this when going between different model architectures.
    exp.to_device()

    exp.compression = compression
    exp.strategy = "GlobalMagWeight"
    exp.prune()
    exp.to_device()
    exp.model.load_state_dict(checkpoint['model_state_dict'])
    exp.optim.load_state_dict(checkpoint['optim_state_dict'])
    #print(list(exp.model.conv1.named_parameters())) # List weights of first layer in LeNet