import os
import torch
import pandas as pd
from .util import calculate_class_correct, calculate_rank
from tqdm import tqdm


def cifar100_init(exp, path=None):
    # Create general csv file
    df = pd.DataFrame(columns=["Train/Val", "Image Index", "Image Label"])
    temp = "Val"
    df["Image Index"] = [i for i in range(10000)]
    base_path = os.environ["ResultPATH"]
    df.to_csv(f'{base_path}/{path}-CIFAR100-Overview.csv', index=False)
    # Create model specific class performance csv file

    df = pd.DataFrame(columns=["Image Index"])
    df["Image Index"] = [i for i in range(10000)]
    df.to_csv(f'{base_path}/CIFAR100-Model{path}.csv', index=False)

def cifar100_class(exp, path=None):
    base_path = os.environ["ResultPATH"]
    column = ['apple', 'aquarium_fish', 'baby',
               'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
               'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
               'caterpillar', 'cattle',
               'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
               'crocodile', 'cup',
               'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl',
               'hamster', 'house',
               'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
               'lobster', 'man',
               'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
               'orange', 'orchid',
               'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
               'poppy', 'porcupine',
               'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
               'shark', 'shrew',
               'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
               'sunflower',
               'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
               'tractor', 'train', 'trout',
               'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
               'worm']
    df = pd.DataFrame(columns=[f"Model", "x", 'apple', 'aquarium_fish', 'baby',
                                          'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                                          'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                                          'caterpillar', 'cattle',
                                          'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
                                          'crocodile', 'cup',
                                          'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl',
                                          'hamster', 'house',
                                          'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                                          'lobster', 'man',
                                          'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
                                          'orange', 'orchid',
                                          'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                                          'poppy', 'porcupine',
                                          'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
                                          'shark', 'shrew',
                                          'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                                          'sunflower',
                                          'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
                                          'tractor', 'train', 'trout',
                                          'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                                          'worm', 'Overall'])
    images = [100 for i in range(100)] + ["Total Images:", 10000]
    new_df = pd.DataFrame(columns=[f"Model", "x", 'apple', 'aquarium_fish', 'baby',
                                          'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                                          'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                                          'caterpillar', 'cattle',
                                          'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
                                          'crocodile', 'cup',
                                          'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl',
                                          'hamster', 'house',
                                          'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                                          'lobster', 'man',
                                          'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
                                          'orange', 'orchid',
                                          'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                                          'poppy', 'porcupine',
                                          'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
                                          'shark', 'shrew',
                                          'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                                          'sunflower',
                                          'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
                                          'tractor', 'train', 'trout',
                                          'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                                          'worm', 'Overall'])
    new_df["x"] = [images[100]]
    new_df["Overall"] = [images[101]]
    df = pd.concat([df, new_df])
    df.to_csv(f'{base_path}/{path}-CIFAR100-Class-{exp.compression}.csv', index=False)


def cifar100_log(exp, path=None):
    model = exp.model
    temp = "Val"
    testdata = exp.val_dl
    state = exp.state
    base_path = os.environ["ResultPATH"]
    print(f'\nLoading {exp.state}-{exp.compression} to csv\n')
    # Structure class Specific dataframe
    class_df = pd.read_csv(rf'{base_path}/{path}-CIFAR100-Class-{exp.compression}.csv')
    # Format for writing to final csv file and create lists for statistics recording
    list_labels = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                   'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle',
                   'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup',
                   'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house',
                   'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man',
                   'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid',
                   'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
                   'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew',
                   'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower',
                   'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout',
                   'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

    temp_class_df = pd.DataFrame(columns=[f"Model", "x", 'apple', 'aquarium_fish', 'baby',
                                          'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                                          'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                                          'caterpillar', 'cattle',
                                          'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
                                          'crocodile', 'cup',
                                          'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl',
                                          'hamster', 'house',
                                          'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                                          'lobster', 'man',
                                          'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
                                          'orange', 'orchid',
                                          'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                                          'poppy', 'porcupine',
                                          'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
                                          'shark', 'shrew',
                                          'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                                          'sunflower',
                                          'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
                                          'tractor', 'train', 'trout',
                                          'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                                          'worm', "Overall"])

    temp_class_df["Model"] = [f"For {exp.state}:"]
    orig_c = [0 for i in range(101)]
    total_c = [0 for i in range(101)]
    percent_c = ["", "Percent Correct:"]
    #

    # structure giant big spreadsheet dataframe
    df_output = []
    df_label = []
    df_rank = []
    #     Probability commented out bc too much

#     df_prob = [[] for x in range(100)]
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
                pred = probab.index(max(probab))  # What the machine guesses
                # #### Calculate Ranking #####
                true = y.cpu().numpy()[0]
                df_rank.append(calculate_rank(probab, pred, true))
                # #### True Label, Prediction, and Probability #####
                df_label.append(true)
                df_output.append(pred)
                #     Probability commented out bc too much
#                 for i, probability in enumerate(probab):
#                     probability = round(probability, 6)
#                     df_prob[i].append(probability)
                if calculate_class_correct(true, pred):
                    orig_c[true] += 1
                    orig_c[100] += 1
                total_c[true] += 1
                total_c[100] += 1

    raw_df = pd.DataFrame(columns=[f"Model", "x", 'apple', 'aquarium_fish', 'baby',
                                          'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
                                          'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle',
                                          'caterpillar', 'cattle',
                                          'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab',
                                          'crocodile', 'cup',
                                          'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl',
                                          'hamster', 'house',
                                          'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard',
                                          'lobster', 'man',
                                          'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree',
                                          'orange', 'orchid',
                                          'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate',
                                          'poppy', 'porcupine',
                                          'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal',
                                          'shark', 'shrew',
                                          'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar',
                                          'sunflower',
                                          'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
                                          'tractor', 'train', 'trout',
                                          'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman',
                                          'worm', "Overall"])

    # Output to class specific dataframe csv file
    for i, x in enumerate(list_labels):
        raw_df[x] = [orig_c[i]]
    raw_df["x"] = ["Correct:"]

    temp_class_df = pd.concat([temp_class_df, raw_df])

    og = pd.concat([class_df, temp_class_df])
    og.to_csv(f'{base_path}/{path}-CIFAR100-Class-{exp.compression}.csv', index=False)

    # Output to giant dataframe csv file
    df = pd.read_csv(rf'{base_path}/{path}-CIFAR100-Overview.csv')
    df[f"{exp.state}-{exp.compression} Ratio"] = df_output
#     Probability commented out bc too much
#     for i, x in enumerate(df_prob):
#         df[f"{exp.state}-{exp.compression} Ratio {i} Prob"] = x
    df[f"{exp.state}-{exp.compression} Ratio Rank"] = df_rank
    df["Image Label"] = df_label
    df.to_csv(f'{base_path}/{path}-CIFAR100-Overview.csv', index=False)
    print(f'\nFinished loading {exp.state}-{exp.compression} to csv\n')

    # Output to statistical specific csv file
    df = pd.read_csv(rf'{base_path}/CIFAR100-Model{path}.csv')
    df["Image Label"] = df_label
    df[f'{exp.compression} PR'] = df_output
    df.to_csv(f'{base_path}/CIFAR100-Model{path}.csv', index=False)


    # for i in range(len(orig_c)):
    #     percent_c.append("%.2f %%" % (100 * (orig_c[i] / total_c[i])))
    #
    # # Output to class specific dataframe csv file
    # for i, x in enumerate(list_labels):
    #     temp_class_df[x] = [orig_c[i]]
    #
    # to_append = percent_c
    #
    # df_length = len(temp_class_df)
    #
    # temp_class_df.loc[df_length] = to_append
    #
    # class_df.append(temp_class_df)
    # og = pd.concat([class_df, temp_class_df])
    # og.to_csv(f'CIFAR100-ClassBreakdown-{exp.compression}.csv',
    #           index=False)
    #
    # # Output to giant dataframe csv file
    # df = pd.read_csv(rf'CIFAR100-Overview-{exp.compression}.csv')
    # df[f"{state}"] = df_output
    # df[f"{state} Correct Probability"] = df_prob
    # df[f"Topk Rank - {state}"] = df_rank
    # df["Image Label"] = df_label
    # df.to_csv(f'CIFAR100-Overview-{exp.compression}.csv', index=False)
    #
    # print(f'\nFinished loading {exp.state}-{exp.compression} to csv\n')
