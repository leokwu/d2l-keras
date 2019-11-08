from keras.utils import np_utils
from keras.datasets import fashion_mnist
from net.resnet import ResNet
import keras
from keras import backend as K

batch_size = 8
num_classes = 10
epochs = 5

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    chanDim = 1
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    chanDim = -1

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


model = ResNet.build(chanDim, input_shape, num_classes)
# model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# start train-----------------------------------
# model.fit(x_train, y_train, batch_size=1000, epochs=20)
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Training set accuracy--------------------------------
result = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0)
print('\nTrain Acc:', result[1])

# Testing set accuracy---------------------------------
result = model.evaluate(x_test, y_test, batch_size=batch_size)
print('\nTest Acc:', result[1])
