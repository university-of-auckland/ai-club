from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from VisualizeNN import DrawNN
import keras.utils

from sklearn.neural_network import MLPClassifier

training_set_inputs = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 1], [0, 0, 0]])
training_set_outputs = np.array([[0, 1, 1, 1, 1, 0, 0]]).T
X = training_set_inputs
y = training_set_outputs

classifier = MLPClassifier(hidden_layer_sizes=(4,), alpha=0.01, tol=0.001, random_state=1)
classifier.fit(X, y.ravel())

network_structure = np.hstack(([X.shape[1]], np.asarray(classifier.hidden_layer_sizes), [y.shape[1]]))

# Draw the Neural Network with weights
network=DrawNN(network_structure, classifier.coefs_)
network.draw()

# Admission data:
# - exam 1 score (x1)
# - exam 2 score (x2)
# - admitted (y)
data = np.loadtxt('/Users/djim087/aiuoa/datasets/students_1.txt', delimiter=',')

# Separate features (x1, x2) from target (y)
X, y = np.hsplit(data, np.array([2]))
y_shape = y.shape[1]
y = keras.utils.to_categorical(y)

model = Sequential()
model.add(Dense(2, activation='sigmoid', input_dim=2))

# Output layer
model.add(Dense(2, activation='sigmoid'))

# For a binary classification problem
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=10, batch_size=34)

# Get layer size.
layer_size = []
for layer in model.layers:
    layer_size.append(int(layer.get_output_at(0).shape[1]))
layer_size.pop()
print(layer_size)

# Draw the Neural Network with weights
network_structure = np.hstack(([X.shape[1]], np.asarray(layer_size), [y_shape]))
weights = []
for i in range(0, len(model.get_weights())):
    if "bias" not in model.weights[i].name:
        weights.append(model.get_weights()[i])
network = DrawNN(network_structure, weights)
network.draw()
