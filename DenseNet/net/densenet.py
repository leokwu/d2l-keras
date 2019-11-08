# import the necessary packages

from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K


def conv_block(x, num_channels):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(num_channels, kernel_size=(3, 3), padding='same')(x)
    return x

def transition_block(x, num_channels):
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(num_channels, kernel_size=(1, 1), padding='same')(x)
    x = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=2)(x)
    return x
    


class DenseBlock(keras.layers.Layer):
    def __init__(self, num_convs, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        self.num_convs = num_convs
        self.num_channels = num_channels        

    def call(self, X):
        x = X
        for _ in range(self.num_convs):       
            Y  = conv_block(x, self.num_channels)
            Z = K.concatenate([X, Y], axis = -1)  # 在通道维上将输入和输出连结
        return Z

    def compute_output_shape(self, input_shape):
        print(input_shape[1], input_shape[2], self.num_chn)
        return (input_shape[0], input_shape[1], input_shape[2], self.num_channels)

class DenseNet:
    def build(chanDim, input_shape, num_classes):

        inputs = keras.Input(input_shape)
        x = keras.layers.Lambda(lambda img: tf.image.resize(img, (128, 128)))(inputs)

        b1 = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=2, padding='same')(x)
        b1 = keras.layers.BatchNormalization(axis=-1)(b1)
        b1 = keras.layers.ReLU()(b1)
        b1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(b1)

        num_channels, growth_rate = 64, 32
        num_convs_in_dense_blocks = [4, 4, 4, 4]
        b2 = b1
        for i, num_convs in enumerate(num_convs_in_dense_blocks):
            b2 = DenseBlock(num_convs, growth_rate)(b2)
            # 上一个稠密块的输出通道数
            num_channels += num_convs * growth_rate
            # 在稠密块之间加入通道数减半的过渡层
        b3 = b2
        if i != len(num_convs_in_dense_blocks) - 1:
            num_channels //= 2
            b3 = transition_block(b3, num_channels)
        
        b4 = keras.layers.BatchNormalization()(b3)
        b4 = keras.layers.Activation('relu')(b4)
        b4 = keras.layers.GlobalAveragePooling2D()(b4)
        # b2 = AveragePooling2D()(b2)
        # b3 = keras.layers.Flatten()(b2)

        outputs = keras.layers.Dense(num_classes, activation='softmax')(b4)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.summary()
        return model


