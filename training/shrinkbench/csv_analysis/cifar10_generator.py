import torch
import numpy as np
import pandas as pd
import os
import statistics
from tqdm import tqdm


def cifar10_generate(path=None, sample=1, function=None):
    base_path = os.environ["CIFARPATH"]
    hypothesis_list = ["0 Test Statistic", "2 Test Statistic", "4 Test Statistic", "5 Test Statistic",
                       "10 Test Statistic", "20 Test Statistic", "50 Test Statistic"]
    list_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    columns = ["Model"] + list_labels
    load = tqdm(range(sample))
    d_list = [[hypothesis_list[x], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for x in range(7)]
    for x in load:
        df = pd.read_csv(rf'{base_path}/{path}/Model{path}.csv')
        # TODO: refactor using list comprehensions
        model_list = [[0 for x in range(11)] for y in range(7)]
        # Loop through 10000 rows in csv files
        for i in range(10000):
            df.iloc[i][1:] = df.iloc[i][1:].sample(frac=1)
            # df row is now randomized
            # Calculate accuracy of the row by model
            for j, item in enumerate(df.iloc[i][1:]):
                if item == df.iloc[i][0]: # If the item guessed equals the label
                    model_list[j][item] += 1
                    model_list[j][10] += 1
        """
        model_list is the accuracy of a randomly generated 'model'. 
        7 rows by 11 columns.
        Rows are different compression ratios and columns are each class guess.
        10th column (11) is the total of all classes guessed "correctly".
        """

        """
        Perform statistic gathering on the randomly generated model
        calculate accuracies: beta_0^c, beta_0^M, beta_t^c, and beta_t^M
        statsByModel[mm] <- ((beta_0^c - beta_0^M)/beta_0^M) - ((beta_t^c - beta_t^M)/beta_t^M)    
        """
    
        # Rotate the matrix (model_list) and rename to class_list
        class_list = np.transpose(model_list, (1, 0))

        # Now want to calculate non compressed values (beta_0^c - beta_0^M)/beta_0^M)
        random_stat_df = pd.DataFrame(columns=columns)
        for i in range(7):
            temp_df = pd.DataFrame(columns=columns)
            temp_df["Model"] = [hypothesis_list[i]]
            # for j, x in enumerate(class_list[:-1]):
            for j, x in enumerate(class_list[:-1]):
                temp_df[list_labels[j]] = [
                    round((((x[i] / 1000) - (class_list[10][i] / 10000)) / (class_list[10][i] / 10000)) -
                          (((x[0] / 1000) - (class_list[10][0] / 10000)) / (class_list[10][0] / 10000)), 8)]
#                     round((((x[i] / 1000) - (class_list[10][i] / 10000)) / (class_list[10][i] / 10000)), 8)]
            random_stat_df = pd.concat([random_df, temp_df])

        random_stat_df.reset_index(drop=True, inplace=True)
        
        '''
        Now generate a table to calculate the number of times random was less/greater than mean sample statistic
        '''
        mult_two = False
        mean_sample_statistics = pd.read_csv(rf'{base_path}/MeanStats.csv')
        if function == "less":
            df = random_df.le(mean_sample_statistics)
        elif function == "greater":
            df = random_df.ge(mean_sample_statistics)
        else:   # get the minimum of the two
            less_df = random_df.le(mean_sample_statistics)
            greater_df = random_df.ge(mean_sample_statistics)
            df = pd.concat([less_df, greater_df]).min(level=0)
            mult_two = True
        counter = 0
        for i, j in df.iterrows():  # j is a series of each compression ratio
            for index, value in j.items():
                if value and (counter % 11 != 0):
                    d_list[counter // 11][(counter % 11)] += 1
                counter += 1
                
        print(d_list)

    output = pd.DataFrame(d_list, columns=columns)
    for x in list_labels:
        output[x] = output[x] / sample
        if mult_two:
            output[x] = output[x] * 2

    return output

# def cifar10_generate2(path=None, sample=1, function=None):
#     base_path = os.environ["CIFARPATH"]
#     hypothesis_list = ["0 Test Statistic", "2 Test Statistic", "4 Test Statistic", "5 Test Statistic",
#                        "10 Test Statistic", "20 Test Statistic", "50 Test Statistic"]
#     list_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
#     columns = ["Model"] + list_labels
    
#     model_output_df = pd.read_csv(rf'{base_path}/{path}/Model{path}.csv')
#     for i in range(10000):
#         # [i] is the row, and there are 7 columns in each row. Img Label, and 6 different compression ratios
#         model_output_df.iloc[i][]
    
    
    
#     load = tqdm(range(sample))
#     d_list = [[hypothesis_list[x], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for x in range(7)]
#     for x in load:
#         df = pd.read_csv(rf'{base_path}/{path}/Model{path}.csv')
#         # TODO: refactor using list comprehensions
#         model_list = [[0 for x in range(11)] for y in range(7)]
#         # Loop through 10000 rows in csv files
#         for i in range(10000):
#             df.iloc[i][1:] = df.iloc[i][1:].sample(frac=1)
#             # df row is now randomized
#             # Calculate accuracy of the row by model
#             for j, item in enumerate(df.iloc[i][1:]):
#                 if item == df.iloc[i][0]: # If the item guessed equals the label
#                     model_list[j][item] += 1
#                     model_list[j][10] += 1
#         print(model_list)
