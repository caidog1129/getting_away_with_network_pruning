import torch.nn as nn
import torch.nn.functional as F

# MNIST Perceptrons 
class Perceptron10(nn.Module):
    def __init__(self):
        super(Perceptron10, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Perceptron20(nn.Module):
    def __init__(self):
        super(Perceptron20, self).__init__()
        self.fc1 = nn.Linear(28*28, 20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Perceptron40(nn.Module):
    def __init__(self):
        super(Perceptron20, self).__init__()
        self.fc1 = nn.Linear(28*28, 40)
        self.fc2 = nn.Linear(40,40)
        self.fc3 = nn.Linear(40,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron50(nn.Module):
    def __init__(self):
        super(Perceptron50, self).__init__()
        self.fc1 = nn.Linear(28*28, 50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron100(nn.Module):
    def __init__(self):
        super(Perceptron100, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Perceptron3_100(nn.Module):
    def __init__(self):
        super(Perceptron3_100, self).__init__()
        self.fc1 = nn.Linear(28*28, 100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,100)
        self.fc4 = nn.Linear(100,10)
        self.fc4.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
class Perceptron200(nn.Module):
    def __init__(self):
        super(Perceptron200, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Perceptron200100(nn.Module):
    def __init__(self):
        super(Perceptron200100, self).__init__()
        self.fc1 = nn.Linear(28*28, 200)
        self.fc2 = nn.Linear(200,100)
        self.fc3 = nn.Linear(100,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Perceptron400200(nn.Module):
    def __init__(self):
        super(Perceptron400200, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc2 = nn.Linear(400,200)
        self.fc3 = nn.Linear(200,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Perceptron600300(nn.Module):
    def __init__(self):
        super(Perceptron600300, self).__init__()
        self.fc1 = nn.Linear(28*28, 600)
        self.fc2 = nn.Linear(600,300)
        self.fc3 = nn.Linear(300,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Perceptron800400(nn.Module):
    def __init__(self):
        super(Perceptron800400, self).__init__()
        self.fc1 = nn.Linear(28*28, 800)
        self.fc2 = nn.Linear(800,400)
        self.fc3 = nn.Linear(400,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron400(nn.Module):
    def __init__(self):
        super(Perceptron400, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc2 = nn.Linear(400,400)
        self.fc3 = nn.Linear(400,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron800(nn.Module):
    def __init__(self):
        super(Perceptron800, self).__init__()
        self.fc1 = nn.Linear(28*28, 800)
        self.fc2 = nn.Linear(800,800)
        self.fc3 = nn.Linear(800,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Perceptron30_10(nn.Module):
    def __init__(self):
        super(Perceptron30_10, self).__init__()
        self.fc1 = nn.Linear(28*28, 30)
        self.fc2 = nn.Linear(30,10)
        self.fc3 = nn.Linear(10,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron25_15(nn.Module):
    def __init__(self):
        super(Perceptron25_15, self).__init__()
        self.fc1 = nn.Linear(28*28, 25)
        self.fc2 = nn.Linear(25,15)
        self.fc3 = nn.Linear(15,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Perceptron10_10(nn.Module):
    def __init__(self):
        super(Perceptron25_15, self).__init__()
        self.fc1 = nn.Linear(28*28, 10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
############# CIFAR10 Models ####################

class Perceptron20_C10(nn.Module):
    def __init__(self):
        super(Perceptron20_C10, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 20)
        self.fc2 = nn.Linear(20,20)
        self.fc3 = nn.Linear(20,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron50_C10(nn.Module):
    def __init__(self):
        super(Perceptron50_C10, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 50)
        self.fc2 = nn.Linear(50,50)
        self.fc3 = nn.Linear(50,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron100_C10(nn.Module):
    def __init__(self):
        super(Perceptron100_C10, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 100)
        self.fc2 = nn.Linear(100,100)
        self.fc3 = nn.Linear(100,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron30_10_C10(nn.Module):
    def __init__(self):
        super(Perceptron30_10_C10, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 30)
        self.fc2 = nn.Linear(30,10)
        self.fc3 = nn.Linear(10,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron25_15_C10(nn.Module):
    def __init__(self):
        super(Perceptron25_15, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 25)
        self.fc2 = nn.Linear(25,15)
        self.fc3 = nn.Linear(15,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Perceptron10_10_C10(nn.Module):
    def __init__(self):
        super(Perceptron25_15, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 10)
        self.fc2 = nn.Linear(10,10)
        self.fc3 = nn.Linear(10,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron200_C10(nn.Module):
    def __init__(self):
        super(Perceptron200_C10, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 200)
        self.fc2 = nn.Linear(200,200)
        self.fc3 = nn.Linear(200,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron400_C10(nn.Module):
    def __init__(self):
        super(Perceptron400_C10, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 400)
        self.fc2 = nn.Linear(400,400)
        self.fc3 = nn.Linear(400,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Perceptron800_C10(nn.Module):
    def __init__(self):
        super(Perceptron800_C10, self).__init__()
        self.fc1 = nn.Linear(32*32*3, 800)
        self.fc2 = nn.Linear(800,800)
        self.fc3 = nn.Linear(800,10)
        self.fc3.is_classifier = True
        
    def forward(self,x):
        # flatten image input
        x = x.view(-1,32*32*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x