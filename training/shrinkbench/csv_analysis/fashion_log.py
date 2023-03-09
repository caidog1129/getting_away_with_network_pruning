# Author Aidan Good
# Master Fashion MNIST logging file

import csv
import torch
import os
import pandas as pd
from tqdm import tqdm
from .util import calculate_class_correct, calculate_rank


def fashion_init(exp, path=None):
    # Create general csv file
    df = pd.DataFrame(columns=["Image Index", "Correct Label"])
    temp = "Val"
    df["Image Index"] = [i for i in range(10000)]
    base_path = os.environ["ResultPATH"]
    df.to_csv(f'{base_path}/{path}-Fashion-Overview.csv', index=False)


def fashion_class(exp, path=None):
    # Create model specific class performance csv file
    base_path = os.environ["ResultPATH"]
    df = pd.DataFrame(
        columns=["Model", "x", "0's", "0%", "1's", "1%", "2's", "2%", "3's", "3%", "4's", "4%", "5's", "5%", "6's",
                 "6%", "7's", "7%", "8's", "8%", "9's", "9%", "Overall", "T%"])
    images = [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009, "Total Images:", 10000]

    for x in [images]:
        new_df = pd.DataFrame(
            columns=["Model", "x", "0's", "0%", "1's", "1%", "2's", "2%", "3's", "3%", "4's", "4%", "5's", "5%", "6's",
                     "6%", "7's", "7%", "8's", "8%", "9's", "9%", "Overall", "T%"])
        new_df["x"] = [x[10]]
        new_df["Overall"] = [x[11]]
        for i in range(10):
            new_df[f"{i}'s"] = [x[i]]
        df = pd.concat([df, new_df])

    df.to_csv(f'{base_path}/{path}-Fashion-Class-{exp.compression}.csv', index=False)


def fashion_log(exp, path=None):
    model = exp.model
    state = exp.state
    testdata = exp.val_dl
    base_path = os.environ["ResultPATH"]

    print(f'\nLoading {state}-{exp.compression} to csv:\n')
    # print(f'{exp.model_name}-{temp}-{exp.compression}-Class.csv')

    # Class Breakdown dataframe
    class_df = pd.read_csv(rf'{base_path}/{path}-Fashion-Class-{exp.compression}.csv')

    # For the Class statistics csv file
    temp_class_df = pd.DataFrame(
        columns=["Model", "x", "0's", "0%", "1's", "1%", "2's", "2%", "3's", "3%", "4's", "4%", "5's", "5%", "6's",
                 "6%", "7's", "7%", "8's", "8%", "9's", "9%", "Overall", "T%"])
    temp_class_df["Model"] = [f"For {exp.state}:"]
    orig_c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total_c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    percent_c = []
    #

    # For the Overview csv File
    df_output = []
    df_label = []
    df_prob = [[] for x in range(10)]
    df_rank = []
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
                true = y.numpy()[0]
                # ##### Ranking #####
                df_rank.append(calculate_rank(probab, pred, true))
                # Collecting for Overview csv
                df_label.append(true)
                df_output.append(pred)
                for i, probability in enumerate(probab):
                    probability = round(probability, 6)
                    df_prob[i].append(probability)

                # Collecting for class specific csv
                if calculate_class_correct(true, pred):
                    orig_c[true] += 1
                    orig_c[10] += 1
                total_c[true] += 1
                total_c[10] += 1

    for i in range(len(orig_c)):
        percent_c.append(round(orig_c[i] / total_c[i], 4))

    stand_df = pd.DataFrame(
        columns=["Model", "x", "0's", "0%", "1's", "1%", "2's", "2%", "3's", "3%", "4's", "4%", "5's", "5%", "6's",
                 "6%", "7's", "7%", "8's", "8%", "9's", "9%", "Overall", "T%"])

    for i in range(10):
        stand_df[f"{i}'s"] = [orig_c[i]]
        stand_df[f"{i}%"] = [percent_c[i]]
    stand_df["Overall"] = [orig_c[10]]
    stand_df["T%"] = [percent_c[10]]
    stand_df["x"] = ["Correct:"]

    temp_class_df = pd.concat([temp_class_df, stand_df])

    class_df.append(temp_class_df)
    og = pd.concat([class_df, temp_class_df])
    og.to_csv(f'{base_path}/{path}-Fashion-Class-{exp.compression}.csv', index=False)

   
    # Output to giant dataframe csv file
    df = pd.read_csv(rf'{base_path}/{path}-Fashion-Overview.csv')

    # Load statistics into the overview csv file
    df[f"{exp.state}-{exp.compression} Ratio"] = df_output
    for i, x in enumerate(df_prob):
        df[f"{exp.state}-{exp.compression} Ratio {i} Prob"] = x
    df[f"{exp.state}-{exp.compression} Ratio Rank"] = df_rank
    df["Correct Label"] = df_label

    df.to_csv(f'{base_path}/{path}-Fashion-Overview.csv', index=False)
    
