import gurobipy as gb
from gurobipy import GRB
import numpy as np
import random
from tqdm import tqdm
import math
import os
import pandas as pd
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import os
import os.path
import sys

parent_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
module_path = parent_path + '/training'
sys.path.insert(0, module_path)

from shrinkbench.experiment import PruningClass
import torch
import torch.nn as nn
import torch.optim as optim
from torch import relu, sigmoid, tanh, selu
import matplotlib.pyplot as plt

def normalize(v):
    # return v
    if np.prod(v) == 0:
        return v
    return v / np.sqrt(v.dot(v))

def gs(A):
    n = len(A)
    A[:,0] = normalize(A[:,0])

    for i in range(1,n):
        Ai = A[:,i]
        for j in range(0, i):
            Aj = A[:,j]
            t = Ai.dot(Aj)
            Ai = Ai - t * Aj
        A[:,i] = normalize(Ai)
    return A
        
class MILP():
    def __init__(self, model, in_size, layer_dims, var):  
        """
        Params:
        model: the model we use
        in_size: input size of model
        layer_dims: list of diminsions of model
        var: number of sample points for the subspace, 0 indicates we do not use sample points (or we use the whole space)
        """
        self.seed = 1
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        
        self.in_size = in_size
        self.layer_dims = layer_dims
        self.var = var
        
        self.modelnn = model
        
        self.w = {}
        self.b = {}
        ''' Changed from original code because shrinkbench perceptrons layers have different names'''
        for i in range(len(self.layer_dims)):
            self.w[i] = self.modelnn.state_dict()[f'fc{i+1}.weight']
            self.b[i] = self.modelnn.state_dict()[f'fc{i+1}.bias']

        # Generates random sample points using numpy
#         if self.var != 0:
#             self.p = []
#             v = []
#             for i in range(self.var):
#                 self.p.append(np.random.random_sample(self.in_size))
#             for i in range(self.var - 1):
#                 v.append(self.p[i+1] - self.p[i])
#             v = np.array(v)
#             self.v = gs(v)

       
        """ UPDATE: Generates random sample points using points from MNIST val dataset """
        if self.var != 0:
            pr_cls = PruningClass(dataset='MNIST', model='Perceptron10',  
                    train_kwargs={'optim': 'SGD', 'epochs': 15, 'lr': 1e-2},
                    dl_kwargs={'batch_size':128})
            
            mnist = pr_cls.val_dataset
            self.p = []
            v = []
            for i in range(self.var):
                self.p.append(mnist[random.randint(0,len(mnist))][0].flatten().squeeze().numpy())
            for i in range(self.var - 1):
                v.append(self.p[i+1] - self.p[i])
            v = np.array(v)
            self.v = gs(v)
            
    def update_model(self, model):
        self.modelnn = model
        
        self.w = {}
        self.b = {}
        ''' Changed from original code because shrinkbench perceptrons layers have different names'''
        for i in range(len(self.layer_dims)):
            self.w[i] = self.modelnn.state_dict()[f'fc{i+1}.weight']
            self.b[i] = self.modelnn.state_dict()[f'fc{i+1}.bias']
        
    def run(self, method, trials = 0, activation_list = []):
        if method == "exact":
            return self.exact()
        if method == "none":
            pass
        if method == "redemacher":
            return self.redemacher(100)
            
    def exact(self):
        """
        Exact counting number of linear regions on the space or subspace 
        This version uses indicator constraint, which is more accurate but slower
        """
        # We define a callback function to be used by the solver
        def callback_function(model, where):
            global linear_regions
            if where == gb.GRB.Callback.MIPSOL:
                f_value = model.cbGetSolution(f)
                if f_value > 0:
                    s = dict()
                    for i in range(len(self.layer_dims)):
                        solution = model.cbGetSolution(z[i])
                        s[i] = solution
                    c = gb.quicksum((1 - s[i][j]) + 2 * (s[i][j] - 0.5) * z[i][j] for i in range(len(self.layer_dims)) for j in range(self.layer_dims[i])) <= sum(self.layer_dims)-1
                    model.cbLazy(c)
                    linear_regions += 1
                    
                    
                    if linear_regions == 100:
                        print("LR is 100") 
                    if linear_regions > 1000 and linear_regions % 1000 == 1:
                        print("Lr now is:", linear_regions)
                    
        self.model = gb.Model()
        self.model.setParam('OutputFlag', 0)
        global linear_regions
        linear_regions = 0
        lambda_sep = {}
        
        # We set the ub and lb to the max and min pixel values in the MNIST dataset
        x = self.model.addVars(self.in_size, ub = 2.9, lb = -0.5, name = 'input')
        f = self.model.addVar(lb = -GRB.INFINITY)
        
        if self.var != 0:
            for i in range(self.var-1):
                lambda_sep[i] = self.model.addVar(lb = -GRB.INFINITY)

            for i in range(self.in_size):
                self.model.addConstr(x[i] == self.p[0][i] + gb.quicksum(lambda_sep[j] * self.v[j][i] for j in range(self.var-1)))

        z={}
        h={}
        g={}
        hbar = {}
        
        # Make the network compact
        # Make an array keeping track of pruned neurons to 
        count = 0
        for dim in self.layer_dims:
            z[count] = self.model.addVars(dim, vtype = GRB.BINARY, name ='indicator')
            h[count] = self.model.addVars(dim, name = 'value')
            hbar[count] = self.model.addVars(dim)
            g[count] = self.model.addVars(dim, lb = -GRB.INFINITY, name = 'true_value')
            count += 1

        for i in range(self.layer_dims[0]):
            # print(self.w[0][1][1], x,
            self.model.addConstr(gb.quicksum(self.w[0][i][j] * x[j] for j in range(self.in_size)) + self.b[0][i] == g[0][i], name = 'calulation')
            self.model.addConstr((z[0][i] == 0) >> (h[0][i] == 0), name = '0 indicator')
            self.model.addConstr((z[0][i] == 1) >> (h[0][i] == g[0][i]), name = '1 indicator')
            self.model.addConstr((z[0][i] == 0) >> (g[0][i] <= 0), name = '0 constraint')
            # self.model.addConstr(f <= g[0][i] + 100 * (1 - z[0][i]))
            # self.model.addConstr(f <= -g[0][i] + 100 * (z[0][i]))
            self.model.addConstr((z[0][i] == 1) >> (f <= g[0][i]), name = 'f 1 constraint')
            self.model.addConstr((z[0][i] == 0) >> (f <= -g[0][i]), name = 'f 0 constraint')

        count = 1
        for input_dim, output_dim in zip(self.layer_dims, self.layer_dims[1:]):
            for i in range(output_dim):
                if count == len(self.layer_dims) - 1:
                    self.model.addConstr(gb.quicksum(self.w[count][i][j] * h[count-1][j] for j in range(input_dim)) + self.b[count][i] == g[count][i], name = 'calculation')
                    self.model.addConstr(z[count][i] == 1)
                else:
                    self.model.addConstr(gb.quicksum(self.w[count][i][j] * h[count-1][j] for j in range(input_dim)) + self.b[count][i] == g[count][i], name = 'calculation')
                    self.model.addConstr((z[count][i] == 0) >> (h[count][i] == 0), name = '0 indicator')
                    self.model.addConstr((z[count][i] == 1) >> (h[count][i] == g[count][i]), name = '1 indicator')
                    self.model.addConstr((z[count][i] == 0) >> (g[count][i] <= 0), name = '0 constraint')
                    # self.model.addConstr(f <= h[count][i] + 100 * (1 - z[count][i]))
                    self.model.addConstr((z[count][i] == 1) >> (f <= g[count][i]), name = 'f 1 constraint')
                    self.model.addConstr((z[count][i] == 0) >> (f <= -g[count][i]), name = 'f 0 constraint')
            count += 1

        self.model.setObjective(f, GRB.MAXIMIZE)
        self.model.params.LazyConstraints = 1
        self.model.optimize(callback_function)
        return linear_regions  
    
    def redemacher(self, trials):
        """
        Counting linear regions using redemacher
        """
        value_choice = [1, -1]
        l = []

        for i in tqdm(range(trials)):
            value = []
            for layer in self.layer_dims:      
                value_temp = np.random.choice(value_choice, size=layer)
                value.append(value_temp)

            self.model = gb.Model()
            self.model.setParam('OutputFlag', 0)
            x = self.model.addVars(self.in_size, lb = -1, ub = 5, name = 'input')
            f = self.model.addVar(lb = -GRB.INFINITY)

            z={}
            h={}
            g={}
            hbar = {}

            count = 0
            for dim in self.layer_dims:
                z[count] = self.model.addVars(dim, vtype = GRB.BINARY, name ='indicator')
                h[count] = self.model.addVars(dim, name = 'value')
                hbar[count] = self.model.addVars(dim)
                g[count] = self.model.addVars(dim, lb = -GRB.INFINITY, name = 'true_value')
                count += 1

            for i in range(self.layer_dims[0]):
                self.model.addConstr(gb.quicksum(self.w[0][i][j] * x[j] for j in range(self.in_size)) + self.b[0][i] == g[0][i], name = 'calulation')
                self.model.addConstr((z[0][i] == 0) >> (h[0][i] == 0), name = '0 indicator')
                self.model.addConstr((z[0][i] == 1) >> (h[0][i] == g[0][i]), name = '1 indicator')
                self.model.addConstr((z[0][i] == 0) >> (g[0][i] <= 0), name = '0 constraint')
                self.model.addConstr(f <= h[0][i] + 100 * (1 - z[0][i]))

            count = 1
            for input_dim, output_dim in zip(self.layer_dims, self.layer_dims[1:]):
                for i in range(output_dim):
                    if count == len(self.layer_dims) - 1:
                        self.model.addConstr(gb.quicksum(self.w[count][i][j] * h[count-1][j] for j in range(input_dim)) + self.b[count][i] == g[count][i], name = 'calculation')
                        self.model.addConstr(z[count][i] == 1)
                    else:
                        self.model.addConstr(gb.quicksum(self.w[count][i][j] * h[count-1][j] for j in range(input_dim)) + self.b[count][i] == g[count][i], name = 'calculation')
                        self.model.addConstr((z[count][i] == 0) >> (h[count][i] == 0), name = '0 indicator')
                        self.model.addConstr((z[count][i] == 1) >> (h[count][i] == g[count][i]), name = '1 indicator')
                        self.model.addConstr((z[count][i] == 0) >> (g[count][i] <= 0), name = '0 constraint')
                        self.model.addConstr(f <= h[count][i] + 100 * (1 - z[count][i]))
            count += 1

            count = 0
            result = 0
            for dim in self.layer_dims:
                r = gb.quicksum(((z[count][i] * 2 - 1) * value[count][i]) for i in range(dim))
                result += r
                count += 1

            self.model.setObjective(result, GRB.MAXIMIZE)
            self.model.optimize()
            l.append(self.model.ObjVal)
        
        values, counts = np.unique(l, return_counts=True)
        return 2 ** (np.dot(values,counts) / trials)