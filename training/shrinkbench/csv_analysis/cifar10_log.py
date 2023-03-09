import torch
import pandas as pd
import pathlib
import os
from .util import calculate_class_correct, calculate_rank
from tqdm import tqdm


def cifar10_init(exp, path=None):
    # Create general csv file
    df = pd.DataFrame(columns=["Train/Val", "Image Index", "Image Label"])
    temp = "Val"
    df["Image Index"] = [i for i in range(10000)]
    base_path = os.environ["ResultPATH"]
    
    pathlib.Path(f"{base_path}").mkdir(parents=True, exist_ok=True)
    
    df.to_csv(f'{base_path}/{path}-CIFAR10-Overview.csv', index=False)
    
    df = pd.DataFrame(columns=["Image Index"])
    df["Image Index"] = [i for i in range(10000)]
    df.to_csv(f'{base_path}/Model{path}.csv', index=False)
    
def cifar10_class(exp, path=None):
    base_path = os.environ["ResultPATH"]
    column = ["airplane", "automobile", "bird",
              "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    df = pd.DataFrame(
        columns=["Model", "x", "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
                 "truck", "Overall"])
    images = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000, "Total Images:", 10000]

    new_df = pd.DataFrame(
        columns=["Model", "x", "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
                 "truck", "Overall"])
    new_df["x"] = [images[10]]
    new_df["Overall"] = [images[11]]
    for i in range(10):
        new_df[f"{column[i]}"] = [images[i]]
    df = pd.concat([df, new_df])

    df.to_csv(f'{base_path}/{path}-CIFAR10-Class-{exp.compression}.csv', index=False)


def cifar10_log(exp, path=None):
    model = exp.model
    temp = "Val"
    testdata = exp.val_dl
    state = exp.state
    base_path = os.environ["ResultPATH"]
    print(f'\nLoading {exp.state}-{exp.compression} to csv\n')
    # Structure class Specific dataframe
#     class_df = pd.read_csv(rf'{base_path}/{path}-CIFAR10-Class-{exp.compression}.csv')

    # Format for writing to final csv file and create lists for statistics recording
    list_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "Overall"]
    temp_class_df = pd.DataFrame(
        columns=["Model", "x", "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
                 "truck", "Overall"])

    temp_class_df["Model"] = [f"For {exp.state}:"]
    orig_c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    total_c = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    percent_c = ["", "Percent Correct:"]
    #

    # structure giant big spreadsheet dataframe
    df_output = []
    df_label = []
    df_rank = []
    df_prob = [[] for x in range(10)]
    #

    epoch_iter = tqdm(testdata)
    epoch_iter.set_description(f"Running through {temp} data")

    with torch.set_grad_enabled(False):
        for i, (a, b) in enumerate(epoch_iter, start=1):
            a, b = a.to(exp.device), b.to(exp.device)
            for (x, y) in zip(a, b):
                x, y = x.unsqueeze(0), y.unsqueeze(0)
                x, y = x.to(exp.device), y.to(exp.device)
                yhat = model(x)
                # My code
                ps = torch.exp(yhat)
                probab = list(ps.cpu().numpy()[0])
                pred = probab.index(max(probab))
                # #### Calculate ranking #####
                true = y.cpu().numpy()[0]
                df_rank.append(calculate_rank(probab, pred, true))
                # #### True Label, Prediction, and Probability #####
                df_label.append(true)
                df_output.append(pred)
                for i, probability in enumerate(probab):
                    probability = round(probability, 6)
                    df_prob[i].append(probability)
                if calculate_class_correct(true, pred):
                    orig_c[true] += 1
                    orig_c[10] += 1
                total_c[true] += 1
                total_c[10] += 1

    raw_df = pd.DataFrame(
        columns=["Model", "x", "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship",
                 "truck", "Overall"])

#     # Output to class specific dataframe csv file
#     for i, x in enumerate(list_labels):
#         raw_df[x] = [orig_c[i]]
#     raw_df["x"] = ["Correct:"]

#     temp_class_df = pd.concat([temp_class_df, raw_df])

#     og = pd.concat([class_df, temp_class_df])
#     og.to_csv(f'{base_path}/{path}-CIFAR10-Class-{exp.compression}.csv', index=False)

    # Output to giant dataframe csv file
    df = pd.read_csv(rf'{base_path}/{path}-CIFAR10-Overview.csv')
    df[f"{exp.state}-{exp.compression} Ratio"] = df_output
    for i, x in enumerate(df_prob):
        df[f"{exp.state}-{exp.compression} Ratio {i} Prob"] = x
    df[f"{exp.state}-{exp.compression} Ratio Rank"] = df_rank
    df["Image Label"] = df_label
    df.to_csv(f'{base_path}/{path}-CIFAR10-Overview.csv', index=False)
    print(f'\nFinished loading {exp.state}-{exp.compression} to csv\n')
    
    #Output to statistical specific csv file
    df = pd.read_csv(rf'{base_path}/Model{path}.csv')
    df["Image Label"] = df_label
    df[f'{exp.compression} PR'] = df_output
    df.to_csv(f'{base_path}/Model{path}.csv', index=False)
    
    
    ########################################################################################################################
    
    
def cifar10_lth(model, loader, path=None):
    base_path = os.environ["ResultPATH"]
    print(f'\nLoading {path} to csv\n')

    # structure giant big spreadsheet dataframe
    df_output = []
    df_label = []
    df_rank = []
    df_prob = [[] for x in range(10)]
    #

    with torch.no_grad():
        for examples, labels in loader:
            examples = examples.to(torch.cpu())
            labels = labels.squeeze().to(torch.cpu())
            for (x, y) in zip(examples, labels):
                x, y = x.unsqueeze(0), y.unsqueeze(0)
                x, y = x.to(exp.device), y.to(exp.device)
                output = exp.model(x)
                print(output)
                # My code                
                pred = x.argmax(dim=1)
                print(pred)
                print(y)
                # #### Calculate ranking #####
                true = y
                # #### True Label, Prediction, and Probability #####
                df_label.append(true)
                df_output.append(pred)
                if calculate_class_correct(true, pred):
                    orig_c[true] += 1
                    orig_c[10] += 1
                total_c[true] += 1
                total_c[10] += 1


    # Output to giant dataframe csv file
    df = pd.read_csv(rf'{base_path}/{path}-CIFAR10-Overview.csv')
    df[f"{level} Ratio"] = df_output
    df["Image Label"] = df_label
    df.to_csv(f'{base_path}/{path}-CIFAR10-Overview.csv', index=False)
    print(f'\nFinished loading {level} to csv\n')
    

    