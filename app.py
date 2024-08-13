import numpy as np
from source.NeuralNetwork import NeuralNetwork

import matplotlib.pyplot as plt

nn = NeuralNetwork(hidden_layer_sizes=(100,), learning_rate=0.001, max_iter=1000, shuffle=True)

# x = np.random.randn(1024, 2)
# y = np.random.randint(0, 1 + 1, (1024, 1))

x = np.array(
  [[0, 0],
   [0, 1],
   [1, 0],
   [1, 1]]
)

y = np.array(
  [[0],
   [1],
   [1],
   [0]]
)

nn.fit(x, y)

print(nn.predict(x))

plt.plot(nn.history['epoch'], nn.history['loss'])
plt.show()
