# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K

class LeNet:
    @staticmethod
    def build():
        #common process-----------------------------------------------------------------------
        model = Sequential()
        model.add(Dense(input_dim=28*28, units=650, activation='relu'))

        #model.add(Dropout(0.8))#prevent overfitting add dropout
        model.add(Dense(units=650, activation='relu'))
        #model.add(Dropout(0.8))
        model.add(Dense(units=650, activation='relu'))
        #model.add(Dropout(0.8))

        #for i in range(25):
        # model.add(Dense(units=701, activation='relu'))

        model.add(Dense(units=10, activation='softmax'))

        return model
