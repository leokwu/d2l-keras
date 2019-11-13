# import the necessary packages
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras.backend as K




class MobileNet_FineTuning:
    def build(chanDim, input_shape, num_classes):
        # common process-----------------------------------------------------------------------
        inputs = keras.Input(input_shape)
        base_model = keras.applications.mobilenet_v2.MobileNetV2(weights='imagenet', input_tensor=inputs, include_top=False)
        x = base_model.output
        x = keras.layers.GlobalAveragePooling2D()(x)
        # b5 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), padding='valid')(b5)
        # b6 = Flatten()(b5)
        # outputs = Dense(num_classes, activation='softmax')(b6)

        outputs = keras.layers.Dense(num_classes, activation='softmax')(x)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.summary()
        return model
