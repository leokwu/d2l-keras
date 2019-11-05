# import the necessary packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import AveragePooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.layers.normalization import BatchNormalization


class AlexNet:
    def build(chanDim, input_shape, num_classes):
        # common process-----------------------------------------------------------------------
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(11, 11), strides=(1, 1),
                         activation='relu',
                         padding='same',
                         input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(192, (5, 5), padding='same', strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(384, (3, 3), padding='same', activation='relu', strides=(1, 1)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(384, (3, 3), padding='same', activation='relu', strides=(1, 1)))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu', strides=(1, 1)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),  padding='valid'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Flatten())
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(num_classes, activation='softmax'))

        return model
