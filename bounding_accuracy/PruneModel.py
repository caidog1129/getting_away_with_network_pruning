import os
import os.path
import sys

parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
module_path = parent_path + '/training'
sys.path.insert(0, module_path)

from shrinkbench.experiment import PruningClass
from shrinkbench.csv_analysis import mnist_init, mnist_log
from shrinkbench.metrics import csvLog, accuracy
import uuid

# Replace with absolute or relative paths to shrinkbench and trainning data
os.environ['DATAPATH'] = '../training/shrinkbench/Training_data'
os.environ["ShrinkPATH"] = '../training/shrinkbench'
os.environ["ResultPATH"] = '../training/Output'

def prune_model(model_arch, strategy, compressions, model_num):
    '''
    Create and prune models of desired architecture using given strategy
    
    Parameters
    ----------
    model_arch:
        An architecture selected from ["Perceptron20", "Perceptron50", "Perceptron100", "Perceptron30_10", "Perceptron25_15"]
    strategy:
        Strategy applied when pruning the model, selected from ["RandomPruning", "GlobalMagWeight", "MixedMagGrad"]
    compressions:
        A list that pass the compression rates selected from [2, 4, 10, 20, 50]
    model_num:
        Number of model for each compression rate
        
    Return
    ------
    model_info:
        A dictionary that helps match compression rate, model number and unique_id for each model.
        model_info[c][i] stores the unique_id of the i-th model with compression rate c.
    exp:
        The pruning class, will be useful when loading model
    '''
    # compressions = [2, 4, 10, 20, 50]
    
    # Initialize model_info
    model_info = {}
    model_info[0] = []
    for compression in compressions:
        model_info[compression] = []
    
    exp = PruningClass(dataset='MNIST',
                        model=f'{model_arch}',
                        train_kwargs={
                            'optim': 'SGD',
                            'epochs': 15,
                            'lr': 1e-2},
                        dl_kwargs={'batch_size':128})
    
    # set up logging infastructure
    exp.run_init()
    exp.fix_seed()
    exp.strategy = strategy
    
    # The number of models to do per strategy
    for i in range(model_num):
        # important for logging and loading models\n",
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
        model_info[exp.compression].append(unique_id_og)

        csvLog(exp, unique_id_og)
        mnist_init(exp, i)
        mnist_log(exp, i)

        for compression in compressions:
            exp.compression = compression

            # Prune the model
            exp.prune()
            exp.state = "Compressed"

            # Note it is unique_id, not unique_id_og
            unique_id = uuid.uuid1().hex[1:9]
            model_info[exp.compression].append(unique_id)

            exp.save_model(f"{model_arch}-{exp.strategy}-{exp.compression}-{unique_id}")
            bf_acc = accuracy(exp.model, exp.val_dl)[0]
            exp.state='compressed'

            csvLog(exp, unique_id)
            mnist_log(exp, i)

            # Finetuning
            exp.update_optim('SGD', 15, 1e-2)
            exp.run()

            exp.load_model(prune=True)
            # Save the model
            unique_id = uuid.uuid1().hex[1:9]
            exp.save_model(f"{model_arch}-{exp.strategy}-finetuned-{exp.compression}-{unique_id}")
            exp.state='finetuned'
            csvLog(exp, unique_id, True)
            mnist_log(exp, i)
            model_info[exp.compression].append(unique_id)
            
            #reload original model
            exp.load_model(f"{model_arch}-{exp.strategy}-0-{unique_id_og}")
            
    return exp, model_info


def load_model(exp, model_arch, strategy, compression, unique_id, finetuned=False):
    '''
    Load the desired model
    
    Parameters
    ----------
    exp:
        Pruning class to load model
    model_arch:
        An architecture selected from ["Perceptron20", "Perceptron50", "Perceptron100", "Perceptron30_10", "Perceptron25_15"]
    strategy:
        Strategy applied when pruning the model, selected from ["RandomPruning", "GlobalMagWeight", "MixedMagGrad"]
    compression:
        A compression rate selected from [2, 4, 10, 20, 50]
    unique_id:
        Unique id generated for the model
    finetuned:
        Whether the model is fintuned or not, default as False
    
    Returns
    -------
    model:
        The desired model
    '''
    
    if finetuned:
        model_name = f"{model_arch}-{strategy}-finetuned-{compression}-{unique_id}"
    else:
        if compression > 0:
            finetuned = True
        model_name = f"{model_arch}-{strategy}-{compression}-{unique_id}"
    
    exp.load_model(file_name=model_name, prune_file=finetuned)
    return exp.model
