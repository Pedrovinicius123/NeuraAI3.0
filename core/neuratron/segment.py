import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class Neuratron(nn.Module):
    def __init__(self, i:int, j:int, lr:float, is_input:bool=True):
        #init function
        super(Neuratron, self).__init__()
        self.generate_brain(i, j)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.sigmoid = nn.Sigmoid()
        self.is_input = is_input

    def generate_brain(self, i:int, j:int):
        #function to generate brain
        self.using = nn.Linear(i, j)
        self.total = nn.Linear(i, j)
        
    def forward(self, X, Y_shape:int) -> np.ndarray:
        self.using.weight.data = torch.from_numpy(self.total.weight.detach().numpy()[:Y_shape, :X.shape[1]])
        self.using.bias.data = torch.from_numpy(self.total.bias.detach().numpy()[:Y_shape])

        return self.sigmoid(self.using(X.float()))

    def fit(self, X:np.ndarray, Y:np.ndarray=None, lr:float=0.0001, criterion=None):
        Y = torch.from_numpy(Y).float()

        if not self.is_input:       
            self.optimizer.zero_grad()
        
            loss = criterion(X.unsqueeze(0), Y.unsqueeze(0)) if len(Y.shape) == 0 else criterion(X, Y)
            loss.backward()
        
        self.optimizer.step()
        self.total.weight.data[:self.using.weight.shape[0], :self.using.weight.shape[1]] = self.using.weight
        self.total.bias.data[:self.using.bias.shape[0]] = self.using.bias

        return loss
