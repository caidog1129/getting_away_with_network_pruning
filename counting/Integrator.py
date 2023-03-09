from MILP_old import MILP
# from MILP import MILP
from PruneModel import prune_model, load_model
from IPython.display import clear_output

import os
import os.path
import sys

parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
module_path = parent_path + '/training'
sys.path.insert(0, module_path)

from shrinkbench.experiment import PruningClass
from shrinkbench.csv_analysis import mnist_init, mnist_log
from shrinkbench.metrics import csvLog, accuracy, layerwise_sparsity
import uuid

# Replace with absolute or relative paths to shrinkbench and trainning data
os.environ['DATAPATH'] = '../training/shrinkbench/Training_data'
os.environ["ShrinkPATH"] = '../training/shrinkbench'
os.environ["ResultPATH"] = '../training/Output'

''' 
Example parameters that must be given

model_arch = "Perceptron25_15"
strategy = "RandomPruning"
compressions = [2, 4, 10]
model_layers = [25, 15, 10]
sample_points = 10
method = 'exact'
model_num = 1
input_size = 748

'''

''' Code below '''

def newExperiment(model_arch, dataset, strategy, compressions, model_layers, csv_name, sample_points=10, method='exact', model_num=1, input_size=784, seed=0, threshold=30):
    
    '''
    Create and prune models of desired architecture using given strategy, count linear regions, and record data into csv file.
    
    Parameters
    ----------
    model_arch: (string)
        An architecture selected from ["Perceptron20", "Perceptron50", "Perceptron100", "Perceptron30_10", "Perceptron25_15"]
    strategy: (string)
        Strategy applied when pruning the model, selected from ["RandomPruning", "GlobalMagWeight", "MixedMagGrad"]
    compressions: (list of ints)
        A list that pass the compression rates selected from [2, 4, 10, 20, 50]
    model_layers: (list of ints)
        List of all non-input layer sizes of the model. Ex: Perceptron50 has two 50-sized layers, so pass [50, 50, 10]
    sample_points: (int)
        Number of sample points wanted for counting linear regions. Defaults to 10
    method: (string)
        Method used to count linear regions. Defaults to 'exact'
    model_num: (int)
        Number of models wanted to create for given parameters. Defaults to 1
    input_size: (int)
        The size of the input to the neural network. Defaults to 784 for MNIST.
    '''
    
    exp = PruningClass(dataset=f'{dataset}',
                        model=f'{model_arch}',
                       seed = seed,
                        train_kwargs={
                            'optim': 'SGD',
                            'epochs': 15,
                            'lr': 1e-2},
                        dl_kwargs={'batch_size':128})
    
    # set up logging infastructure
    exp.run_init()
    exp.fix_seed(seed)
    exp.strategy = strategy
    
    acc_lst = []
    lr_lst = []
    # The number of models to do per strategy
    for i in range(model_num):
        # important for logging and loading models
        exp.round = i
        exp.state = 'Original'
        exp.compression = 0
        exp.pruning = False
        exp.build_model(f"{model_arch}")
        exp.update_optim('SGD', 15, 1e-2)

        # Train the model for x epochs
        exp.run()

        # Loads the best perfomring model and save it
        exp.load_model(checkpoint=True)

        # Save the model
        unique_id_og = uuid.uuid1().hex[1:9]
        exp.save_model(f"{model_arch}-{exp.strategy}-{exp.compression}-{unique_id_og}")
        
        # Count linear regions
#         milp = MILP(exp.model, input_size, model_layers, sample_points)
#         regions = milp.run(method)
        
        # Count remaining weights per layer
        layers = layerwise_sparsity(exp.model)
        
        # Record data to csv files
#         csvLog(exp, unique_id_og, regions, method, layers, csv_name)
#         mnist_init(exp, i)
#         mnist_log(exp, i)
#         acc_lst.append(accuracy(exp.model, exp.val_dl)[0])
        

        for compression in compressions:
            exp.compression = compression

            # Prune the model
            exp.prune()
            exp.state = "Compressed"

            # Note it is unique_id, not unique_id_og
            unique_id = uuid.uuid1().hex[1:9]

            exp.save_model(f"{model_arch}-{exp.strategy}-{exp.compression}-{unique_id}")
            bf_acc = accuracy(exp.model, exp.val_dl)[0]
            exp.state='compressed'
            
#             # Count linear regions
#             milp.update_model(exp.model)
#             regions = milp.run(method)
            
#             # Count remaining weights per layer
#             layers = layerwise_sparsity(exp.model)

#             # Record data to csv files
#             csvLog(exp, unique_id, regions, method, layers, csv_name)
#             mnist_log(exp, i)

            # Finetuning
            exp.update_optim('SGD', 15, 1e-2)
            exp.run()

            exp.load_model(prune=True)
            # Save the model
            unique_id = uuid.uuid1().hex[1:9]
            exp.save_model(f"{model_arch}-{exp.strategy}-finetuned-{exp.compression}-{unique_id}")
            exp.state='finetuned'
            
            # Count linear regions
            # masks = exp.masks
            # milp = MILP(exp.model, masks, input_size, model_layers, sample_points)
            milp = MILP(exp.model, input_size, model_layers, sample_points)
            regions = milp.run(method)
            
            # Returns -1 indicating that we did not count linear regions
            if regions == None:
                regions = -1
            lr_lst.append(regions)
            print(lr_lst)
            # Count remaining weights per layer
            layers = layerwise_sparsity(exp.model)

            # Record data to csv files
#             csvLog(exp, unique_id, regions, method, layers, csv_name, True)
#             mnist_log(exp, i)
            acc = accuracy(exp.model, exp.val_dl)[0]
            acc_lst.append(acc)
            
            #reload original model
            exp.load_model(f"{model_arch}-{exp.strategy}-0-{unique_id_og}")
    
    return acc_lst, lr_lst
