'''
New and Improved Logger
Architecture, Model Seed, Pruning Rate, Acc Before, Acc After
'''

from .accuracy import accuracy
import pandas as pd
from os.path import exists
import os


def csvLog(model, unique_ID, regions, method, layers, csv_name, finetuned=False):
#     round_id = model.round
    seed = model.seed
    arch = model.model_name
    strat = model.strategy
    prune_rate = model.compression
    acc = accuracy(model.model, model.val_dl)[0]
    path = os.environ["ShrinkPATH"]
    path = path + f'/../csvOut/{csv_name}.csv'
    
    if exists(path):
        df = pd.read_csv(path, index_col=[0])

        df.loc[len(df.index)] = [arch, seed, strat, prune_rate, finetuned, unique_ID, acc, regions, method, layers]
        
    else:
        df = pd.DataFrame([arch, seed, strat, prune_rate, finetuned, unique_ID, acc, regions, method, layers]).T
        df.columns = ["Architecture", "Model Seed", "Strategy", "Pruning Rate", "Finetuned", "Unique ID", "Accuracy", "Regions", "Method", "Remaining Sparsity"]
    
    df.to_csv(path)