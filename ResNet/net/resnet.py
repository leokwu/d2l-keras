# import the necessary packages
from keras.models import Sequential
from keras.models import Model
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
import tensorflow as tf
from keras.layers import concatenate
from keras.layers import add
from keras.layers.convolutional import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Layer
from keras.layers import ReLU
from keras.layers import Activation


class Residual():
    def __init__(self, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = Conv2D(num_channels, kernel_size=(3, 3), padding='same',
                               strides=strides)
        self.conv2 = Conv2D(num_channels, kernel_size=(3, 3), padding='same')
        if use_1x1conv:
            self.conv3 = Conv2D(num_channels, kernel_size=(1, 1),
                                   strides=strides)
        else:
            self.conv3 = None
        self.bn1 = BatchNormalization(axis=-1)
        self.bn2 = BatchNormalization(axis=-1)

    def __call__(self, X):
        Y = ReLU()(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        return ReLU()(Y + X)

class ResNet:
    def build(chanDim, input_shape, num_classes):
        # common process-----------------------------------------------------------------------
        inputs = Input(input_shape)
        x = Lambda(lambda img: tf.image.resize(img, (224, 224)))(inputs)

        b1 = Conv2D(64, kernel_size=(7, 7), strides=2, padding='same')(x)
        b1 = BatchNormalization(axis=-1)(b1)
        # b1 = Activation('relu')(b1)
        b1 = ReLU()(b1)
        b1 = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(b1)        

        b2 = resnet_block(b1, 64, 2, first_block=True)
        b2 = resnet_block(b2, 128, 2)
        b2 = resnet_block(b2, 256, 2)
        b2 = resnet_block(b2, 512, 2)

        b3 = GlobalAveragePooling2D()(b2)
      
        outputs = Dense(num_classes, activation='softmax')(b3)
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model
def resnet_block(x, num_channels, num_residuals, first_block=False):
    y = x
    for i in range(num_residuals):
        if i == 0 and not first_block:
            y = Residual(num_channels, use_1x1conv=True, strides=2)(y)
        else:
            y = Residual(num_channels)(y)
    return y
