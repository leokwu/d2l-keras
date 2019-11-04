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

class HousePricesNet:
    @staticmethod
    def build():
        model = Sequential()
        # model.add(Dense(units=650, activation='relu'))
        model.add(Dense(units=650))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.3))# prevent overfitting add dropout
        # model.add(Dense(units=650, activation='relu'))
        model.add(Dense(units=650))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.3))
        # model.add(Dense(units=650, activation='relu'))
        model.add(Dense(units=650))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dropout(0.3))

        # for i in range(4):
        #    model.add(Dense(units=1024, activation='relu'))

        model.add(Dense(1))
        # return the constructed network architecture
        return model
