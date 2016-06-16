from keras.datasets import mnist
from keras.utils import np_utils
import numpy as np

def load_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1], X_train.shape[2]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1], X_test.shape[2]))
    X_train -= np.mean(X_train)
    X_test -= np.mean(X_test)

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)

    return (X_train, Y_train), (X_test, Y_test)


