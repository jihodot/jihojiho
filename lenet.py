from keras.utils import np_utils
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.layers import Dense, Flatten

from keras.layers import Dropout
from keras.optimizers import SGD
from keras.layers import Conv2D, MaxPooling2D

class LeNet:
    @staticmethod
    def build(width, height, depth, classes, weightsPath=None):
                # initialize the model
        model = Sequential()
        model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(width, height, depth)))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(120, activation='relu'))

        model.add(Dense(84, activation='relu'))

        model.add(Dense(10, activation='softmax'))

        return model
