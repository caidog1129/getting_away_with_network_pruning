import pathlib
import time

import torch
import torchvision
import torchvision.models
from torch import nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
from tqdm import tqdm
import json

from .base import Experiment
from .. import datasets
from .. import models
from ..metrics import correct
from ..models.head import mark_classifier
from ..util import printc, OnlineStats


class TrainingExperiment(Experiment):
    default_dl_kwargs = {'batch_size': 128,
                         'pin_memory': False,
                         'num_workers': 8
                         }

    default_train_kwargs = {'optim': 'SGD',
                            'epochs': 30,
                            'lr': 1e-3,
                            }

    def __init__(self,
                 dataset,
                 model,
                 seed=42,
                 path=None,
                 dl_kwargs=dict(),
                 train_kwargs=dict(),
                 debug=False,
                 pretrained=False,
                 resume=None,
                 resume_optim=False,
                 save_freq=10):

        # Default children kwargs
        super(TrainingExperiment, self).__init__(seed)
        dl_kwargs = {**self.default_dl_kwargs, **dl_kwargs}
        train_kwargs = {**self.default_train_kwargs, **train_kwargs}

        params = locals()
        params['dl_kwargs'] = dl_kwargs
        params['train_kwargs'] = train_kwargs
        # print(train_kwargs)
        self.add_params(**params)
        # Save params
        self.dataset = dataset
        self.dl_kwargs = dl_kwargs
        self.epoch = 0

        if dataset == 'QMNIST':
            self.build_qmnist_dataloader()
        else:
            self.build_dataloader(dataset, **dl_kwargs)

        self.build_model(model, pretrained, resume)
        self.to_device()


        self.build_train(resume_optim=resume_optim, **train_kwargs)
        self.path = path
        self.save_freq = save_freq

    def run(self):
        self.freeze()
        printc(f"Running {repr(self)}", color='YELLOW')
        self.to_device()
        self.build_logging(self.train_metrics, self.path)
        self.top_loss = 1
        self.run_epochs()
        print("Done running epochs")

    def build_dataloader(self, dataset, **dl_kwargs):
        constructor = getattr(datasets, dataset)
        self.train_dataset = constructor(train=True)
        self.val_dataset = constructor(train=False)
        self.train_dl = DataLoader(self.train_dataset, shuffle=True, **dl_kwargs)  # Changed to false
        self.val_dl = DataLoader(self.val_dataset, shuffle=False, **dl_kwargs)

    def build_model(self, model, pretrained=False, resume=None):
        if isinstance(model, str):
            if hasattr(models, model):
                model = getattr(models, model)()

            elif hasattr(torchvision.models, model):
                # https://pytorch.org/docs/stable/torchvision/models.html
                model = getattr(torchvision.models, model)
                mark_classifier(model)  # add is_classifier attribute
            else:
                raise ValueError(f"Model {model} not available in custom models or torchvision models")

        self.model = model

        if resume is not None:
            self.resume = pathlib.Path(self.resume)
            assert self.resume.exists(), "Resume path does not exist"
            previous = torch.load(self.resume)
            self.model.load_state_dict(previous['model_state_dict'])

    def build_train(self, optim, epochs, resume_optim=False, **optim_kwargs):
        default_optim_kwargs = {
            'SGD': {'momentum': 0.9, 'nesterov': True, 'lr': 1e-3, 'weight_decay' : 0},
            'Adam': {'momentum': 0.9, 'betas': (.9, .99), 'lr': 1e-3,  'weight_decay' : 0}
        }

        self.epochs = epochs

        # Optim
        if isinstance(optim, str):
            constructor = getattr(torch.optim, optim)
            if optim in default_optim_kwargs:
                optim_kwargs = {**default_optim_kwargs[optim], **optim_kwargs}
            try:
                optim = constructor(self.model.parameters(), **optim_kwargs)
            except:
                optim = torch.optim.Adam(self.model.parameters(), optim_kwargs['lr'])
        else:
            print("Something wrong here")

        self.optim = optim
        # print(self.optim)

        if resume_optim:
            assert hasattr(self, "resume"), "Resume must be given for resume_optim"
            previous = torch.load(self.resume)
            self.optim.load_state_dict(previous['optim_state_dict'])

        self.loss_func = nn.CrossEntropyLoss()
        # Assume classification experiment
        self.loss_func = nn.CrossEntropyLoss()
        if self.dataset == 'CelebA':
            weights = [1e-2, 1/0.15]
            class_weights = torch.FloatTensor(weights).to(self.device)
            self.loss_func = nn.CrossEntropyLoss(weight=class_weights)
#             self.loss_func = nn.CrossEntropyLoss()
#         if self.dataset == 'CelebA':
#             self.loss_func = torch.nn.BCELoss()

    def to_device(self):
        # Torch CUDA config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not torch.cuda.is_available():
            printc("GPU NOT AVAILABLE, USING CPU!", color="GREEN")
        else:
            printc("GPU AVAILABLE", color="GREEN")
        self.model.to(self.device)
        cudnn.benchmark = True  # For fast training.

    def checkpoint(self):
        checkpoint_path = self.path / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        epoch = self.log_epoch_n
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, checkpoint_path / f'checkpoint.pt')
        
    def prune_checkpoint(self):
        checkpoint_path = self.path / 'checkpoints'
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        epoch = self.log_epoch_n
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': self.optim.state_dict()
        }, checkpoint_path / f'checkpoint-{self.compression}.pt')


    def run_epochs(self):

        since = time.time()
        try:
            for epoch in range(self.epochs):
                
                epoch = self.epoch
                self.epoch += 1
                printc(f"\nStart epoch {epoch}", color='YELLOW')
                self.train(epoch)
                loss, acc1, acc5 = self.eval(epoch)
                # Checkpoint epochs
                # Model checkpointing based on best val loss/acc
                if loss < self.top_loss:
                    self.top_loss = loss
                    if not self.pruning:
                        self.checkpoint()
                    else:
                        self.prune_checkpoint()
                # TODO Early stopping
                # TODO ReduceLR on plateau?
                self.log(timestamp=time.time() - since)
                self.log_epoch(epoch)

        except KeyboardInterrupt:
            printc(f"\nInterrupted at epoch {epoch}. Tearing Down", color='RED')

    def run_epoch(self, train, epoch=0):
        if train:
            self.model.train()
            prefix = 'train'
            dl = self.train_dl
        else:
            prefix = 'val'
            dl = self.val_dl
            self.model.eval()

        total_loss = OnlineStats()
        acc1 = OnlineStats()
        acc5 = OnlineStats()

        # Create the fancy little progress bar with tqdm(dl)
        # Iterates over the dataloaded picked above in train/val
        # print(dl)

        epoch_iter = tqdm(dl)
        epoch_iter.set_description(f"{prefix.capitalize()} Epoch {epoch}/{self.epochs}")
        # print(epoch_iter)

        with torch.set_grad_enabled(train):
            for i, (x, y) in enumerate(epoch_iter, start=1):

                x, y = x.to(self.device), y.to(self.device)
                yhat = self.model(x)

                if self.dataset == 'CelebA':
                    y = y[:, 9]
#                     ids = torch.Tensor([9] * len(y)).to(self.device).long()
#                     y = y.gather(1, ids.view(-1, 1))
                    # y = y.float()

                loss = self.loss_func(yhat, y)
                if train:
                    loss.backward()
                    self.optim.step()
                    self.optim.zero_grad()

                if self.dataset == 'CelebA':
                    c1, c2 = correct(yhat, y, (1, 1))
                    
                    total_loss.add(loss.item() / dl.batch_size)
                    acc1.add(c1 / dl.batch_size)
                    acc5.add(c2 / dl.batch_size)
                    
                    epoch_iter.set_postfix(loss=total_loss.mean, top1=acc1.mean)
#                     for x, y in zip(yhat, y):
#                         if x < 0.5 and y < 0.5:
#                             cor += 1
#                         elif x > 0.5 and y > 0.5:
#                             cor += 1

                else:
                    c1, c5 = correct(yhat, y, (1, 5))

                    total_loss.add(loss.item() / dl.batch_size)
                    acc1.add(c1 / dl.batch_size)
                    acc5.add(c5 / dl.batch_size)

                    epoch_iter.set_postfix(loss=total_loss.mean, top1=acc1.mean, top5=acc5.mean)

        self.log(**{
            f'{prefix}_loss': total_loss.mean,
            f'{prefix}_acc1': acc1.mean,
            f'{prefix}_acc5': acc5.mean,
        })

        return total_loss.mean, acc1.mean, acc5.mean

    def train(self, epoch=0):
        return self.run_epoch(True, epoch)

    def eval(self, epoch=0):
        return self.run_epoch(False, epoch)

    @property
    def train_metrics(self):
        return ['epoch', 'timestamp',
                'train_loss', 'train_acc1', 'train_acc5',
                'val_loss', 'val_acc1', 'val_acc5',
                ]

    def __repr__(self):
        if not isinstance(self.params['model'], str) and isinstance(self.params['model'], torch.nn.Module):
            self.params['model'] = self.params['model'].__module__

        assert isinstance(self.params['model'], str), f"\nUnexpected model inputs: {self.params['model']}"
        return json.dumps(self.params, indent=4)
