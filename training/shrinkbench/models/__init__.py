import os
import pathlib


from .head import replace_head
from .mnistnet import MnistNet, LeNet5, LeNet, ToyLeNet
from .cifar_resnet import (resnet20,
                           resnet32,
                           resnet44,
                           resnet56,
                           resnet110,
                           resnet1202,
                           resnet56_C)
# from .cifar_resnet import (resnet20_100,
#                            resnet32_100,
#                            resnet44_100,
#                            resnet56_100,
#                            resnet110_100,
#                            resnet1202_100)

from .cifar_vgg import vgg_bn_drop, vgg_bn_drop_100

from .celebA_resnet import resnet18

from .resnet18 import resnet

from .perceptron import *

from .OddBallModels import *