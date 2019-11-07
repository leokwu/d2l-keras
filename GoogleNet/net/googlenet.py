# import the necessary packages
from keras.models import Sequential
from keras.models import Model
from keras import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.layers import Lambda
from keras.layers import Reshape
from keras.layers import Input

class GoogleNet:
    def build(chanDim, input_shape, num_classes):
        # common process-----------------------------------------------------------------------
        inputs = Input(input_shape)
        if chanDim == -1:
            x = Lambda(lambda image: K.resize_images(image, 32, 32, data_format="channels_last"), input_shape=input_shape, output_shape=(32, 32, 1))(inputs)
        if chanDim == 1:
            x = Lambda(lambda image: K.resize_images(image, 32, 32, data_format="channels_first"), input_shape=input_shape, output_shape=(1, 32, 32))(inputs)
        x = Conv2D(32, kernel_size=(3, 3),
                         activation='relu')(x)
        BatchNormalization(axis=chanDim)(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Dropout(0.25)(x)
        x = Flatten()(x)
        outputs = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=outputs)
        return model
