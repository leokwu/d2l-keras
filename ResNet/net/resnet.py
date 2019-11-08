# import the necessary packages

from tensorflow import keras
import tensorflow as tf

class Residual(keras.layers.Layer):
    def __init__(self, num_channels, use_1x1conv=False, strides=1, *args, **kwargs):
        super(Residual, self).__init__(*args, **kwargs)
        self.num_chn = num_channels
        self.conv1 = keras.layers.Conv2D(num_channels, kernel_size=3, padding='same', strides=strides)
        self.conv2 = keras.layers.Conv2D(num_channels, kernel_size=3, padding='same')
        self.relu = keras.layers.ReLU()
        self.relu2 = keras.layers.ReLU()
        
        if use_1x1conv:
           self.conv3 = keras.layers.Conv2D(num_channels, kernel_size=1, strides=strides)
        else:
           self.conv3 = None
            
        self.bn1 = keras.layers.BatchNormalization()
        self.bn2 = keras.layers.BatchNormalization()
        
    def call(self, inputs, training=None, mask=None):
        Y = self.relu(self.bn1(self.conv1(inputs)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
           inputs = self.conv3(inputs)
        print('>>>>>>>>>>>>>', Y.shape, inputs.shape)  
        Y =  self.relu2(Y + inputs)
        print(">>>>>>>>>>>>y", Y.shape)
        return Y

    def compute_output_shape(self, input_shape):
        print(input_shape[1], input_shape[2], self.num_chn)
        return (input_shape[0], input_shape[1], input_shape[2], self.num_chn)


class ResNet:
    def build(chanDim, input_shape, num_classes):

        inputs = keras.Input(input_shape)
        x = keras.layers.Lambda(lambda img: tf.image.resize(img, (128, 128)))(inputs)

        b1 = keras.layers.Conv2D(64, kernel_size=(7, 7), strides=2, padding='same')(x)
        b1 = keras.layers.BatchNormalization(axis=-1)(b1)
        b1 = keras.layers.ReLU()(b1)
        b1 = keras.layers.MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(b1)

        b2 = resnet_block(b1, 64, 2, first_block=True)
        b2 = resnet_block(b2, 128, 2)
        b2 = resnet_block(b2, 256, 2)
        b2 = resnet_block(b2, 512, 2)

        b3 = keras.layers.GlobalAveragePooling2D()(b2)
        # b2 = AveragePooling2D()(b2)
        # b3 = keras.layers.Flatten()(b2)

        outputs = keras.layers.Dense(num_classes, activation='softmax')(b3)
        model = keras.Model(inputs=inputs, outputs=outputs)

        model.summary()
        return model


def resnet_block(x, num_channels, num_residuals, first_block=False):
    for i in range(num_residuals):
        if i == 0 and not first_block:
            x = Residual(num_channels, use_1x1conv=True, strides=2)(x)
        else:
            x = Residual(num_channels)(x)
    return x

