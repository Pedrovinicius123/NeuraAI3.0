import os, pickle, torch
import numpy as np
from neuratron.brain import Brain
from train import train_and_save_model as fit

loss = torch.nn.CrossEntropyLoss()

def train_data_(model_name:str, n_strings:int, input_neuratrons:int=10, inner_neuratron_shape:int=(300, 300), lr:float=0.001, x_train:np.ndarray, y_train:np.ndarray):
    strings = generate_random_strings(30)
    neura = Brain(input_neuratrons=input_neuratrons, inner_neuratron_shape=inner_neuratron_shape, lr=lr)

    for x_inner, y_inner in zip(x_train, y_train):
        fit(neura, model_name, x_inner, y_inner, 30, inner_neuratron_shape=(300, 300), lr=0.001, epochs=1500)

            
def recusive_train_data(epochs, filename:str):
    results = []
    model = None

    with open(os.path.join('models', filename), 'rb') as file:
        model = pickle.load(file)
        samples = generate_random_strings(n_strings=30)
        samples = vectorize_samples(samples)

        sample_size_i = max(item.shape[0] for item in samples)
        sample_size_j = max(item.shape[1] for item in samples)
        new_samples = []

        for sample in samples:
            tot = np.zeros((sample_size_i, sample_size_j))
            tot[:sample.shape[0], :sample.shape[1]] = sample
            new_samples.append(tot)

        results = [model.forward(sample, sample.shape[1]) for sample in new_samples]

        for result, sample in zip(results, new_samples):
            model.fit(result[0].detach().numpy(), sample, epochs=epochs)


    with open(os.path.join('models', filename), 'wb') as file:
        pickle.dump(model, file)


            
