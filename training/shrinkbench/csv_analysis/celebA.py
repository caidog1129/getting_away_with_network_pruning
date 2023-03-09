
import csv
import torch
import os
import pandas as pd
from tqdm import tqdm
from .util import calculate_class_correct, calculate_rank


def celebA_init(exp, path=None):
    # Create general csv file
    df = pd.DataFrame(columns=["Image Index", "Writer ID", "Correct Label"])
    temp = "Val"
    df["Image Index"] = [i for i in range(19962)]
    base_path = os.environ["ResultPATH"]
    df.to_csv(f'{base_path}/{path}-CelebA-Overview-{exp.compression}.csv', index=False)


def celebA_log(exp, path=None):
    model = exp.model
    state = exp.state
    testdata = exp.val_dl
    base_path = os.environ["ResultPATH"]

    print(f'\nLoading {state}-{exp.compression} to csv:\n')

    # For the Overview csv File
    df_output = []
    df_label = []
    df_true_prob = []
    #####

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
                try:
                    true = y.numpy()[0][9]
                    # ##### Ranking #####
                    # Collecting for Overview csv
                    df_label.append(true)
                    df_output.append(pred)
                    df_true_prob.append(round(probab[true], 4) * 100)
                except:
                    print(y.numpy())

    df = pd.read_csv(rf'{base_path}/{path}-CelebA-Overview-{exp.compression}.csv')

    # Load statistics into the overview csv file
    df[f"{state}"] = df_output
    df[f"Correct Probability - {state}"] = df_true_prob
    df["Correct Label"] = df_label

    df.to_csv(f'{base_path}/{path}-CelebA-Overview-{exp.compression}.csv', index=False)
