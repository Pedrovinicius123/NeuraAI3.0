import numpy as np
from scipy.special import expit

class Neuratron:
    def __init__(self, i:int, j:int) -> None:
        # Generate Bias and structure (Weights)
        self.structure = self.generate_brain(i, j)
        self.bias = self.generate_brain(i, j)


    def generate_brain(self, i:int, j:int) -> np.ndarray:
        # Generator method
        struct = np.random.rand(i, j)

        if i < j:
            for i_line in range(i):
                struct[i_line, i_line] = 0

        else:
            for j_col in range(j):
                struct[j_col, j_col] = 0

        return struct

    def process_anterior_data(self, X:np.ndarray) -> np.ndarray:
        # Processing method of the Neuratron Segment
        struct = self.structure[:X.shape[1], :X.shape[0]]
        bias = self.bias[:X.shape[0], :X.shape[0]]

        self.Activation = expit(np.dot(X, struct) + bias)
        return self.Activation, struct, bias


    def backpropagation(self, Y:np.ndarray, struct:np.ndarray, bias:np.ndarray, learning_rate:float) -> None:
        # Backpropagation function
        error = self.Activation - Y
        delta1 = self.sigmoidal_deriv(error)

        struct -= (delta1.dot(struct.T) * learning_rate).T
        bias -= delta1 * learning_rate

        self.structure[:struct.shape[0], :struct.shape[1]] = struct
        self.bias[:bias.shape[0], :bias.shape[1]] = bias

    def learn(self, X:np.ndarray, Y:np.ndarray, epochs:int):
        for i in range(epochs):
            activation, struct, bias = self.process_anterior_data(X)
            self.backpropagation(Y, struct, bias, learning_rate=0.0001)

            print(f"Epoch {i}")


    @staticmethod
    def sigmoidal_deriv(X:np.ndarray):
        # Sigmoidal function derivative
        return expit(X) * expit(1 - X)
