import torchvision.models as models
import torch
from torch import nn


def resnet():
    resnet18 = models.resnet18()

    lin = nn.Linear(in_features=512, out_features=10, bias=True)
    resnet18.fc = lin

    return resnet18