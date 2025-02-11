from neuratron.brain import Brain
import numpy as np
import pickle, os

from sklearn.model_selection import train_test_split
from torch.nn import MSELoss as Loss
import torch

def train_and_save_model(model:Brain, filename:str, X:np.ndarray, target:np.ndarray, input_neuratrons:int=3, inner_neuratron_shape:tuple=(100, 100), lr=0.01, epochs:int=1000):
    X_train, X_test, Y_train, Y_test = train_test_split(X, target, test_size=0.3, shuffle=True)

    model.fit(X_train, Y_train, epochs=epochs)
    y_pred, sum_tot = model.forward(X_test, Y_test.shape[1])
    
    print(Y_test)
    criterion = Loss()
    loss = criterion(y_pred, torch.from_numpy(Y_test))

    print(f'Loss: {loss}')

    try:
        with open(os.path.join('models', filename), 'wb') as file:
            pickle.dump(model, file)

    except FileNotFoundError:
        os.mkdir('models')

        with open(os.path.join('models', filename), 'wb') as file:
            pickle.dump(model, file)
