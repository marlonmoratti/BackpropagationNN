from source.NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

nn = NeuralNetwork(hidden_layer_sizes=(100,), learning_rate=0.001, max_iter=100, shuffle=True, random_state=42)

x, y = load_breast_cancer(return_X_y=True)
x = StandardScaler().fit_transform(x)
y = y.reshape(-1, 1)
nn.fit(x, y)

y_pred = np.where(nn.predict(x) > 0.5, 1, 0)
print(balanced_accuracy_score(y, y_pred))

plt.plot(nn.history['epoch'], nn.history['loss'])
plt.show()
