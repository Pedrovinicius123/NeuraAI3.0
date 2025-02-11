import nltk, os, pickle
import numpy as np

from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from neuratron.brain import Brain
from train import train_and_save_model as fit
from faker import Faker

faker = Faker()
vectorizer = CountVectorizer()

nltk.download('punkt')
nltk.download('punkt_tab')

def generate_random_strings(n_strings:int):
    texts = []
    for i in range(n_strings):
        texts.append(faker.text())

    return texts

def vectorize_samples(samples:list):
    vectorized_samples = []    
    
    for sample in samples:
        print(str(sample.encode()).split(' '))
        raw=b''.join(['\x00'+e if len(e)==1 else e for e in sample.encode.split(' ')])
        print(raw)
        vectorized_samples.append(np.fromstring(raw))
        

    return vectorized_samples

def train_data_(n_strings:int, input_neuratrons:int=10, inner_neuratron_shape:int=(300, 300), lr:float=0.001):
    samples = generate_random_strings(n_strings=n_strings)
    vectorized_samples = vectorize_samples(samples)
    
    neura = Brain(input_neuratrons=input_neuratrons, inner_neuratron_shape=inner_neuratron_shape, lr=lr)
    for x, y in zip(vectorized_samples, vectorized_samples):
        fit(neura, 'neura1.1.pkl', x, y, 30, inner_neuratron_shape=(300, 300), lr=0.001, epochs=1500)

def recusive_train_data(epochs:int=1000):
    results = []
    model = None

    with open(os.path.join('models', 'neura1.1.pkl'), 'rb') as file:
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
        print(results.tobytes().decode())

        for result, sample in zip(results, new_samples):
            model.fit(result[0].detach().numpy(), sample, epochs=epochs)


    with open(os.path.join('models', 'neura1.1.pkl'), 'wb') as file:
        pickle.dump(model)

    recusive_train_data(int=1000)   

if __name__ == '__main__':
    train_data_(1)
    recusive_train_data()
