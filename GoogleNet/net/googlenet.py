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
from keras.layers.core import Lambda
from keras.layers.core import Reshape

class GoogleNet:
    def build(chanDim, input_shape, num_classes):
        # common process-----------------------------------------------------------------------
        model = Sequential()
        if chanDim == -1:
            model.add(Lambda(lambda x: K.resize_images(x, 32, 32, data_format="channels_last"), input_shape=input_shape, output_shape(32, 32, 1)))
        if chanDim == 1:
            model.add(Lambda(lambda x: K.resize_images(x, 32, 32, data_format="channels_first"), input_shape=input_shape, output_shape=(1, 32, 32)))
        model.add(Conv2D(32, kernel_size=(3, 3),
                         activation='relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        return model
