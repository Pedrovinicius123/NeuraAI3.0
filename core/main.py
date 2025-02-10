from neuratron.brain import Brain
from scipy.special import expit
import numpy as np

from sklearn.model_selection import train_test_split

x_size = np.random.randint(1, 10)
y_size = np.random.randint(1, 10)
z_size = np.random.randint(1, 10)

x, y, z = np.linspace(0, x_size, 500).reshape(-1,1), np.linspace(0, y_size, 500).reshape(-1,1), np.linspace(0, z_size, 500).reshape(-1,1)

X = np.concatenate((x, y, z), 1)
target = np.linspace(0, 1, 500).reshape(-1,1)

X_train, X_test, Y_train, Y_test = train_test_split(X, target, test_size=0.3, shuffle=True)

# X, y = np.random.randint(10, size=(10, 3)), np.random.randint(3, size=(10, 1))

print(X_train.shape, Y_train.shape)

neura = Brain(input_neuratrons=100, inner_neuratron_shape=(100, 100), lr=0.00001)
neura.fit(X_train, Y_train, epochs=10000)
