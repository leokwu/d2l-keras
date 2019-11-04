from keras.utils import np_utils
from keras.datasets import mnist
from net.lenet import LeNet

def load_data():
    #(x_train, y_train), (x_test, y_test) = mnist.load_data('mnist.npz')
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[0: number]
    y_train = y_train[0: number]
    x_train = x_train.reshape(number, 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    # convert class vectors to binary class matrices
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train
    x_test = x_test
    x_train = x_train / 255
    x_test = x_test / 255
    #x_test = np.random.normal(x_test)#for random test data
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape)

model = LeNet.build()
#model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
#model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#start train-----------------------------------
#model.fit(x_train, y_train, batch_size=1000, epochs=20)
model.fit(x_train, y_train, batch_size=100, epochs=20)

#Training set accuracy--------------------------------
result = model.evaluate(x_train, y_train, batch_size=10000)
print('\nTrain Acc:', result[1])

#Testing set accuracy---------------------------------
result = model.evaluate(x_test, y_test, batch_size=10000)
print('\nTest Acc:', result[1])
