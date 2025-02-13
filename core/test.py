import nltk, os, pickle, binascii, codecs
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
        raw = binascii.hexlify(sample.encode())
        vectorized_samples.append(np.frombuffer(raw, np.uint8).reshape(-1,1))

    return vectorized_samples

def train_data_(model_name:str, n_strings:int, input_neuratrons:int=10, inner_neuratron_shape:int=(300, 300), lr:float=0.001, x:np.ndarray=None, y:np.ndarray=None):
    if x == None:    
        samples = generate_random_strings(n_strings=n_strings)
        x = vectorize_samples(samples)

    neura = Brain(input_neuratrons=input_neuratrons, inner_neuratron_shape=inner_neuratron_shape, lr=lr)

    if y == None:      
        for x_inner, y_inner in zip(x, x):
            fit(neura, model_name, x_inner, y_inner, 30, inner_neuratron_shape=(300, 300), lr=0.001, epochs=1500)

    else:
        for x_inner, y_inner in zip(x, y):
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

def test_model(filename:str):
    with open(os.path.join('models', filename), 'rb') as file:
        model = pickle.load(file)
        sample_strings = generate_random_strings(n_strings=30)
        vectorized = vectorize_samples(sample_strings)

        results = []
        vhex = np.vectorize(hex)
        
        for string in vectorized:
            results.append(np.round(model.forward(string, string.shape[1])[0].detach().numpy()).astype(int))


if __name__ == '__main__':
    train_data_('lingua1.0.pkl', 10)
    recusive_train_data(1000, 'lingua1.0.pkl')
    test_model('lingua1.0.pkl')
