from neuratron.brain import Brain
from scipy.special import expit
from sklearn.datasets import load_iris

import numpy as np

X, y = load_iris(return_X_y=True)

# X, y = np.random.randint(10, size=(10, 3)), np.random.randint(3, size=(10, 1))

print(X.shape)

neura = Brain(input_neuratrons=10, inner_neuratron_shape=(100, 100), lr=0.0001)
neura.fit(X, y, epochs=100000)
