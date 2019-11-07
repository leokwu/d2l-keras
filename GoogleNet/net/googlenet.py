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

class Inception():
    # c1 - c4为每条线路里的层的输出通道数
    def __init__(self, c1, c2, c3, c4, **kwargs):
        super(Inception, self).__init__(**kwargs)
        # 线路1，单1 x 1卷积层
        self.p1_1 = Conv2D(c1, kernel_size=1, activation='relu')
        # 线路2，1 x 1卷积层后接3 x 3卷积层
        self.p2_1 = Conv2D(c2[0], kernel_size=1, activation='relu')
        self.p2_2 = Conv2D(c2[1], kernel_size=3, padding='same',
                              activation='relu')
        # 线路3，1 x 1卷积层后接5 x 5卷积层
        self.p3_1 = Conv2D(c3[0], kernel_size=1, activation='relu')
        self.p3_2 = Conv2D(c3[1], kernel_size=5, padding='valid',
                              activation='relu')
        # 线路4，3 x 3最大池化层后接1 x 1卷积层
        self.p4_1 = MaxPooling2D(pool_size=3, strides=1, padding='same')
        self.p4_2 = Conv2D(c4, kernel_size=1, activation='relu')

    # def forward(self, x):
    def __call__(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return concatenate((p1, p2, p3, p4), axis=-1)  # 在通道维上连结输出


class GoogleNet:
    def build(chanDim, input_shape, num_classes):
        # common process-----------------------------------------------------------------------
        inputs = Input(input_shape)
        x = Lambda(lambda img: tf.image.resize(img, (96, 96)))(inputs)
        b1 = Conv2D(64, kernel_size=7, strides=2, padding='valid', activation='relu')(x)
        b1 = MaxPooling2D(pool_size=3, strides=2, padding='same')(b1)
        
        b2 = Conv2D(64, kernel_size=1, activation='relu')(b1)
        b2 = Conv2D(192, kernel_size=3, padding='same', activation='relu')(b2)
        b2 = MaxPooling2D(pool_size=3, strides=2, padding='same')(b2)
        
        b3 = Inception(64, (96, 128), (16, 32), 32)(b2)
        b3 = Inception(128, (128, 192), (32, 96), 64)(b3)
        b3 = MaxPooling2D(pool_size=3, strides=2, padding='same')(b3)
        
        b4 = Inception(192, (96, 208), (16, 48), 64)(b3)
        b4 = Inception(160, (112, 224), (24, 64), 64)(b4)
        b4 = Inception(128, (128, 256), (24, 64), 64)(b4)
        b4 = Inception(128, (128, 256), (24, 64), 64)(b4)
        b4 = Inception(256, (160, 320), (32, 128), 128)(b4)
        b4 = MaxPooling2D(pool_size=3, strides=2, padding='same')(b4)

        b5 = Inception(256, (160, 320), (32, 128), 128)(b4)
        b5 = Inception(384, (192, 384), (48, 128), 128)(b5)
        b5 = GlobalAveragePooling2D()(b5)

        outputs = add(b1, b2, b3, b4, b5, Dense(num_classes))
        model = Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model
