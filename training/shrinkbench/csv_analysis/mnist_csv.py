import csv
import torch
import os
import pandas as pd
from tqdm import tqdm
from .util import calculate_class_correct, calculate_rank


def mnist_init(exp, path=None):
    # Create model specific class performance csv file
    base_path = os.environ["ResultPATH"]
    df = pd.DataFrame(
        columns=[f"model {path}", "", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Overall"])
    images = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009, "Total Images:", 10000]

    for x in [images]:
        df["Overall"] = [x[11]]
        df[""] = ["Total #"]
        for i in range(10):
            df[f"{i}"] = [x[i]]   
    
    df.to_csv(f'{base_path}/{path}-{exp.model_name}-Accuracy-{exp.strategy}.csv', index=False)

def mnist_log(exp, path=None):
    model = exp.model
    state = exp.state
    testdata = exp.val_dl
    base_path = os.environ["ResultPATH"]

    print(f'\nLoading {state}-{exp.compression} to csv:\n')

    # Class Breakdown dataframe
    class_df = pd.read_csv(rf'{base_path}/{path}-{exp.model_name}-Accuracy-{exp.strategy}.csv')

    # For the Class statistics csv file
    orig_c = [0 for x in range(11)]

    epoch_iter = tqdm(testdata)
    epoch_iter.set_description(f"Analysing Model")

    with torch.set_grad_enabled(False):
        for i, (a, b) in enumerate(epoch_iter, start=1):
            a, b = a.to(exp.device), b.to(exp.device)
            for (x, y) in zip(a, b):
                x, y = x.unsqueeze(0), y.unsqueeze(0)
                y = y.cpu()
                # ##### Machine Guessing #####
                yhat = model(x)
                ps = torch.exp(yhat).cpu()
                probab = list(ps.numpy()[0])
                pred = probab.index(max(probab))  # What the machine guesses
                true = y.numpy()[0]

                # Collecting for class specific csv
                if calculate_class_correct(true, pred):
                    orig_c[true] += 1
                    orig_c[10] += 1


    orig_c = [f"compression {exp.compression}", f"{exp.state}"] + orig_c 
    temp_class_df = pd.DataFrame(
        columns=orig_c)
    temp_class_df[f"compression {exp.compression}"] = [""]





    temp_class_df.to_csv(f'{base_path}/{path}-{exp.model_name}-Accuracy-{exp.strategy}.csv', mode='a', index=False)

