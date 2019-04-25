from sklearn.neural_network import MLPClassifier
from scipy.io import arff
import numpy as np
import matplotlib.pyplot as plt
import sys
import math

# confusion matrix is a 10 x 10 array with rows as true class

print("Loading arff data...")
data = arff.loadarff('mnist_784.arff')
print("done")

# convert data into useful numpy format
# picture j is stored at location data[0][j]
# last element of this array is the image's digit
# images are size 28 x 28

print("converting arff data to numpy format...")
rows = len(data[0])
print(f"Number of images in a data set is {rows}")
e = np.dtype(data[0][0])
cols = len(e.names) - 1
print(f"Number of pixels in each image is {cols}")
X = np.zeros((rows,cols), np.float)
y = np.zeros(rows, np.float)
for j in range(rows):
    tmplist = data[0][j].tolist()
    X[j] = np.asarray(tmplist[:-1], np.float)
    y[j] = np.float(tmplist[-1])
print("done")

# rescale the range of data to 0-1
X = X / 255

# extract the training set to get X_train and y_train...
# first 60,000 images for training
X_train = X[:60000]
y_train = y[:60000]

# extract the test set to get X_test and y_test...
# last 10,000 images for test
X_test = X[60000:]
y_test = y[60000:]

# define and train the model, make predictions
mlp = MLPClassifier(hidden_layer_sizes=(100,100), learning_rate_init=0.1, max_iter=400, alpha=1e-4, solver='sgd', verbose=10, tol=1e-4, random_state=1)
mlp.fit(X_train, y_train)
yhat = mlp.predict(X_test)

correct = np.sum(y_test == yhat) / y_test.shape[0]