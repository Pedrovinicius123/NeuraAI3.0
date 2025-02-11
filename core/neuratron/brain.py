from neuratron.segment import Neuratron
from alive_progress import alive_bar

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Brain:
    def __init__(self, input_neuratrons:int, inner_neuratron_shape:tuple, lr:float):
        self.inputs = []
        self.inner_shape = inner_neuratron_shape
        self.lr = lr
        self.develop_brain(input_neuratrons)

        self.output_neuratron = Neuratron(self.inner_shape[0], self.inner_shape[1], lr=lr,  is_input=False)        
        self.criterion = nn.MSELoss()
        

    def develop_brain(self, input_neuratrons:int):
        i, j = self.inner_shape

        for item in range(input_neuratrons):
            self.inputs.append(Neuratron(i, j, lr=self.lr))

    def forward(self, X:np.ndarray, Y_shape:int):
        i, j = self.inner_shape
        self.outputs = []
        sum_tot = np.zeros((X.shape[0], X.shape[1]))

        for neura in self.inputs:
            neur = neura(torch.from_numpy(X), X.shape[1]).detach().numpy()
            sum_tot += neur

        final_output = self.output_neuratron(torch.from_numpy(sum_tot), Y_shape)
        return final_output, sum_tot

    def sigmoid_derivative(self, A):
        A = A.detach().numpy()
        return A * (1-A)

    def fit(self, X:np.ndarray, Y:np.ndarray, epochs:int):
        Y_shape = Y.shape[1]
        final_loss = None

        with alive_bar(epochs) as bar:
            for i in range(epochs):
                final_output, sum_tot = self.forward(X, Y_shape)
                final_loss = self.output_neuratron.fit(final_output, Y=Y, lr=self.lr, criterion=self.criterion)       
                
                gradientW = self.output_neuratron.using.weight.grad
                new_gradientW = np.dot(sum_tot, np.dot(gradientW.detach().numpy(), self.output_neuratron.using.weight.detach().numpy().T))


                for neura in self.inputs:
                    neura.using.weight.data = torch.from_numpy(np.dot(X.T, new_gradientW))
                    neura.using.bias.data = torch.from_numpy(new_gradientW)

                bar()

        print(f'LOSS: {final_loss}')
