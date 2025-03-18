from source.NeuralNetwork import NeuralNetwork
from sklearn.datasets import load_digits
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

nn = NeuralNetwork(
    hidden_layer_sizes=(200,),
    learning_rate=0.001,
    max_iter=20,
    shuffle=True,
    random_state=42
)

x, y = load_digits(return_X_y=True)
x = StandardScaler().fit_transform(x / 16.)
y = y.reshape(-1, 1)

nn.fit(x, y)

y_pred = np.argmax(nn.predict(x), axis=1)
print('Balanced Accuracy in the digits dataset:', balanced_accuracy_score(y, y_pred))

plt.plot(nn.history['epoch'], nn.history['loss'])
plt.title('Loss Curve')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.show()
