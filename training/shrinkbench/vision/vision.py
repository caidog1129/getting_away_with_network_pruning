import json
import torch
import os
import torchvision
import datetime
import string
import random
from torchvision import transforms
from .train import TrainingExperiment

from .. import strategies
from ..metrics import model_size, flops, accuracy
from ..util import printc
from ..csv_analysis import mnist_csv
from ..models import MnistNet, LeNet

"""
Modified Pruning experiment class to separate pruning and training methods

    To Use:
        run() method to run training epochs
        prune() method to prune model
"""


class VisionClass(TrainingExperiment):

    def __init__(self,
                 dataset,
                 model,
                 strategy=None,
                 compression=0,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=False,
                 resume=None,
                 resume_optim=False,
                 on_cpu=False,
                 save_freq=10):
        self.model_name = model

        super(VisionClass, self).__init__(dataset, model, seed, path, dl_kwargs, train_kwargs, debug, pretrained,
                                           resume, resume_optim, save_freq)

        self.add_params(strategy=strategy, compression=compression)

        # Added for built data logging:
        self.strategy = strategy
        self.compression = compression
        self.state = 'Original Model'
        self.loss = 0
        
        self.to_device()
        self.pruning = False

        self.dataset = dataset
        self.dlkwargs = dl_kwargs
        self.train_kwargs = train_kwargs

        self.path = path
        self.save_freq = save_freq

    def apply_pruning(self, strategy, compression):
        """
        Applies the pruning strategy selected. Pruning strategies implemented in shrinkbench/strategies

        Args:
            strategy: The strategy being used to prune the NN
            compression: The Compression ratio selected
        """
        constructor = getattr(strategies, strategy)
        x, y = next(iter(self.train_dl))
        self.pruning = constructor(self.model, x, y, compression=compression)
        self.pruning.apply()
        printc(f"Model Pruned using {strategy} strategy", color='GREEN')
        self.state = f'{strategy} Model'
        
    def run_init(self):
        """ 
        Set up logging before running multilpe times 
        """
        
        self.freeze()
        self.build_logging(self.train_metrics, self.path)
        self.save_metrics()
        self.top_loss = 1
        self.epoch = 0

    def run(self):
        """
        Set up CUDA acceleration and run training epochs
        """
        try:
            self.freeze()
            #printc(f"Running {repr(self)}", color='YELLOW') # Commented out for more compact output
            self.to_device()
            self.run_epochs()
        except:
            print("Error with run")

    def prune(self):
        """
        Calling method to apply the pruning strategy selected.
        """
        self.apply_pruning(self.strategy, self.compression)
        self.pruning = True

    def save_metrics(self):
        self.metrics = self.pruning_metrics()
        with open(self.path / 'metrics.json', 'w') as f:
            json.dump(self.metrics, f, indent=4)

    def pruning_metrics(self):
        metrics = {}
        # Model Size
        size, size_nz = model_size(self.model)
        metrics['size'] = size
        metrics['size_nz'] = size_nz
        metrics['compression_ratio'] = size / size_nz

        x, y = next(iter(self.val_dl))
        x, y = x.to(self.device), y.to(self.device)

        # FLOPS
        ops, ops_nz = flops(self.model, x)
        metrics['flops'] = ops
        metrics['flops_nz'] = ops_nz
        metrics['theoretical_speedup'] = ops / ops_nz

        # Accuracy
        loss, acc1, acc5 = self.run_epoch(False, -1)
        self.log_epoch(-1)

        metrics['loss'] = loss
        metrics['val_acc1'] = acc1
        metrics['val_acc5'] = acc5

        return metrics

    def save_model(self, name = False):
        """
        Save the pytorch model in the specified directory and with specified file name

        """
        path = os.environ['ShrinkPATH']
        if name == False:   
            file_name = f"{self.generate_uid()}-{self.model_name}"
        else:
            file_name = name
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict(),
            'loss': self.loss,
            'epoch': self.epoch
            },
            f"{path}/saved_models/{file_name}.pt")

    def load_model(self, file_name=None):
        """
        Loads a pretrained model already saved. Defaults to an mnist model if left None

        Args:
            model: The name of the model

        """
        self.update_optim(self.train_kwargs["optim"], self.train_kwargs["epochs"], self.train_kwargs["lr"])
        path = os.environ["ShrinkPATH"]

        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        if file_name is not None 
            self.model = torch.load(f"{path}/saved_models/{file_name}.pt")

#         if "resnet18" in self.model_name:
#             self.build_model("resnet18")
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.optim.load_state_dict(checkpoint['optim_state_dict'])
#             print("\nResnet18 Model Loaded")

#         elif "resnet56" in self.model_name:
#             self.build_model("resnet56")
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.optim.load_state_dict(checkpoint['optim_state_dict'])
#             print("\nResnet56 Model Loaded")
#         elif "resnet56_C" in self.model_name:
#             self.build_model("resnet56_C")
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.optim.load_state_dict(checkpoint['optim_state_dict'])
#             print("\nResnet56_C Model Loaded")
#         elif "resnet" in self.model_name:
#             self.build_model("resnet")
#             self.model.load_state_dict(checkpoint['model_state_dict'])
#             self.optim.load_state_dict(checkpoint['optim_state_dict'])
#             print("\nResnet18 Model Loaded")
#         elif "LeNet" in self.model_name:
#             model = LeNet()
#             model.load_state_dict(checkpoint['model_state_dict'])
#             self.optim.load_state_dict(checkpoint['optim_state_dict'])
#             self.model = model
#         else:
#             print("Error, no model found for this")
#             raise Exception

#         self.state = 'Original Model'
#         self.pruning = False
#         self.model.eval()
#         self.build_train(**self.train_kwargs)
#         self.to_device()

    def build_qmnist_dataloader(self):
        """
        Recommend use test10k or test50k for extra labels

        Args:
            type: There are 5 types: train, test, test10k, test50k, and nist

        Returns: creates qmnist_dl
        """
        path = os.environ["ShrinkPATH"]
        dl_kwargs = self.dl_kwargs
        mean, std = 0.1307, 0.3081
        normalize = transforms.Compose([#transforms.Pad(2),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=(mean,), std=(std,))])

        trainset = torchvision.datasets.QMNIST(f"{path}/Training_data/QMNIST",
                                               what='train', compat=True, transform=normalize)

        valset = torchvision.datasets.QMNIST(f"{path}/Training_data/QMNIST",
                                             what='test10k', compat=True, transform=normalize)

        self.val_dl = torch.utils.data.DataLoader(valset,
                                                  batch_size=dl_kwargs['batch_size'],
                                                  shuffle=False)

        self.train_dl = torch.utils.data.DataLoader(trainset,
                                                    batch_size=dl_kwargs['batch_size'],
                                                    shuffle=True)

        self.qmnist_dl = torch.utils.data.DataLoader(
            torchvision.datasets.QMNIST(f"{path}/Training_data/QMNIST",
                                        what='test10k', compat=False, transform=normalize),
            batch_size=dl_kwargs['batch_size'],
            shuffle=False)

    def update_optim(self, optim='SGD', epochs=10, lr=1e-3, weight_decay=0):
        """
        Change the optimizer parameters for finetuning after pruning

        Args:
            epochs: The new number of epochs
            lr: The new Learning Rate

        """
        t_kwargs = {'optim': optim, 'epochs': epochs, 'lr': lr, 'weight_decay':weight_decay}

        self.build_train(**t_kwargs)

    def generate_uid(self):
        """Returns a time sortable UID, overloaded from base class

        Computes timestamp and appends unique identifier

        Returns:
            str -- uid
        """
        if hasattr(self, "uid"):
            return self.uid

        N = 4  # length of nonce
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        nonce = ''.join(random.choices(string.ascii_uppercase + string.digits, k=N))
        self.uid = f"{time}-{nonce}-{self.model_name}"
        return self.uid