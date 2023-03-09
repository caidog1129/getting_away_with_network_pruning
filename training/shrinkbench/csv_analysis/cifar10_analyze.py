import torch
import numpy
import pandas as pd
import os
import statistics


def cifar10_analyze(path=None, compressions=[2, 4, 8]):
    num_pruning_strats = 2
    base_path = os.environ["CIFARPATH"]
    arr = []
    list_labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", "Overall"]
    pruning_methods = ["Global", "Layer"]
    for i, x in enumerate(compressions):
        df = pd.read_csv(rf'{base_path}/{path}/{path}-CIFAR10-Class-{x}.csv')
        # CSV file format: row 0 has totals (not needed since all 1000). row 2, 4, 6, etc have the data.
        # Rows 3, 5, 7, etc are blank
        for j in range(1, 5):
            if j % 2 == 1:  # If odd
                continue
            temp_array = []
            for label in list_labels:
                temp_array.append(df.iloc[j][label])
            arr.append(temp_array)
    """
    Format of arr: A list of list where each sublist contains the raw output of the 10 classes, and each sublist is a 
    different compression algorithm. Each group of compression ratios consists of 3 different list: original, global
    magnitude, and layerwise magnitude. 
    """

    # Clean up arr
    # only care about the second item
    clean_arr = []
    for i, x in enumerate(arr):
        if i % 2 == 1:
            clean_arr.append(arr[i])

    """
    Get the raw output of correct numbers for each model/compression
    """
    original = arr[0]
    correct_list = []
    k = 0
    columns = ["Model"] + list_labels
    df = pd.DataFrame(columns=columns)
    for j, x in enumerate(list_labels):
        df[f"{x}"] = [original[j]]
    df["Model"] = [f"Original Model:"]

    for i, x in enumerate(clean_arr):
        temp_array = []
        for j in range(len(x)):
            temp_array.append(x[j])
        temp_df = pd.DataFrame(columns=columns)
        for j, x in enumerate(list_labels):
            temp_df[f"{x}"] = [temp_array[j]]
        temp_df["Model"] = [f"Global {compressions[i]}"]
        df = pd.concat([df, temp_df])

    """
    Prepare standard deviation array
    """
    clean_arr.insert(0, original)
    stdev_arr = []
    for j in range(len(clean_arr[0])):
        stdev_arr.append([])

    for i, x in enumerate(clean_arr):

        for j in range(len(x)):
            stdev_arr[j].append(clean_arr[i][j])
    stdev_arr = stdev_arr[:-1]

    """
    Calculate standard deviation
    """

    collected_stdev_list = []
    for i, x in enumerate(stdev_arr):
        collected_stdev_list.append(statistics.pstdev(stdev_arr[i]))

    array = numpy.array(collected_stdev_list)
    order = array.argsort()

    stdev_rank = [0] * len(order)
    for i, x in enumerate(order):
        stdev_rank[i] = str(list_labels[order[i]])
    stdev_rank.reverse()

    array = numpy.array(original[:-1])

    order = array.argsort()

    original_rank = [0] * len(order)
    for i, x in enumerate(order):
        original_rank[i] = str(list_labels[order[i]])
    original_rank.append("Original Ranking:")
    stdev_rank.append("StDev Ranking:")

    ranking_df = pd.DataFrame(columns=columns)
    for list in [original_rank, stdev_rank]:
        temp_df = pd.DataFrame(columns=columns)
        for i, x in enumerate(list_labels[:-1]):
            temp_df[x] = [list[i]]
        temp_df["Model"] = [list[10]]
        ranking_df = pd.concat([ranking_df, temp_df])

    df = pd.concat([df, ranking_df])

    temp_df = pd.DataFrame(columns=columns)
    for x in columns:
        temp_df[x] = [""]
    df = pd.concat([df, temp_df])

    """
    Perform statistic gathering
    calculate accuracies: beta_0^c, beta_0^M, beta_t^c, and beta_t^M
    statsByModel[mm] <- ((beta_0^c - beta_0^M)/beta_0^M) - ((beta_t^c - beta_t^M)/beta_t^M)    
    """
    # TODO: EZ refactor using range(1:12) and appending to empty list:
    # Also replace columns with list_label to avoid starting at 1
    airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck, overall = df[columns[1]].tolist(),\
        df[columns[2]].tolist(), df[columns[3]].tolist(), df[columns[4]].tolist(), df[columns[5]].tolist(),\
        df[columns[6]].tolist(), df[columns[7]].tolist(), df[columns[8]].tolist(), df[columns[9]].tolist(),\
        df[columns[10]].tolist(), df[columns[11]].tolist()
    class_list = [airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck, overall]

    
    hypothesis_list = ["0 Test Statistic", "2 Test Statistic", "4 Test Statistic", "5 Test Statistic",
                       "10 Test Statistic", "20 Test Statistic", "50 Test Statistic"]

    # Now want to calculate non compressed values (beta_0^c - beta_0^M)/beta_0^M)
    hypothesis_df = pd.DataFrame(columns=columns)
    for i in range(7):
        temp_df = pd.DataFrame(columns=columns)
        temp_df["Model"] = [hypothesis_list[i]]
        for j, x in enumerate(class_list[:-1]):
            temp_df[list_labels[j]] = [round((((x[0]/1000) - (class_list[10][0]/10000)) / (class_list[10][0]/10000)) -
                                              (((x[i]/1000) - (class_list[10][i]/10000)) / (class_list[10][i]/10000)), 8)]
        hypothesis_df = pd.concat([hypothesis_df, temp_df])

    df = pd.concat([df, hypothesis_df])

    df.to_csv(f'{base_path}/{path}/{path}-CIFAR10-Results.csv', index=False)
    print("CSV file created")