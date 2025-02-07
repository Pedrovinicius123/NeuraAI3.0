from neuratron.segment import Neuratron
import numpy as np

X = np.random.rand(10, 4)
Y = np.random.rand(10, 1)

neura = Neuratron(10, 20)
neura.learn(X, Y, epochs=2000)