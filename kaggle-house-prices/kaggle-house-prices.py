import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from net.housepricesnet import HousePricesNet
from keras.optimizers import SGD, Adam
from keras import backend as K
import matplotlib.pyplot as plt

# INIT_LR = 1e-4
INIT_LR = 5
BS = 8
EPOCHS = 100


train_data = pd.read_csv('dataset/train.csv')
test_data = pd.read_csv('dataset/test.csv')

print('train_data.shape: %s test_data.shape: %s\n' % (train_data.shape, test_data.shape))
print('train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]: %s\n' % (train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]))

all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))

print('all_features: %s\n' % (all_features))

numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 标准化后，每个特征的均值变为0，所以可以直接用0来替换缺失值
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# dummy_na=True将缺失值也当作合法的特征值并为其创建指示特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# print('all_features.shape: %s\n' % (all_features.shape))
print('all_features.shape: ', (all_features.shape))

n_train = train_data.shape[0]
train_features = np.array(all_features[:n_train].values)
test_features = np.array(all_features[n_train:].values)
train_labels = np.array(train_data.SalePrice.values).reshape((-1, 1))

print('train_features: %s\n test_features: %s\n train_labels: %s\n' % (train_features, test_features, train_labels))

def log_rmse(y_true, y_pred):
    # 将小于1的值设成1，使得取对数时数值更稳定
    clipped_preds = K.clip(y_pred, 1, float('inf'))
    rmse = K.sqrt(K.mean(K.pow((K.log(clipped_preds) - K.log(y_true)), 2)))
    print('rmse: %s \n', rmse)
    return rmse

opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model = HousePricesNet.build()


# model.compile(loss='mse', optimizer=opt, metrics=[log_rmse])
model.compile(loss='mse', optimizer='adam', metrics=[log_rmse])

# H = model.fit_generator((train_features, train_labels),
# 	validation_data=test_features, steps_per_epoch=len(train_features) // BS,
# 	epochs=EPOCHS)
H = model.fit(train_features, train_labels, validation_split=0.15, batch_size=50, epochs=EPOCHS)

# np.set_printoptions(threshold=np.inf)

predictions = model.predict(test_features, batch_size=BS)
print('predictions: \n', predictions)

test_data['SalePrice'] = pd.Series(predictions.reshape(1, -1)[0])
submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
submission.to_csv('submission.csv', index=False)

result = model.evaluate(train_features, train_labels, batch_size=100)
print('\nTrain log rmse:', result[1])



plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, EPOCHS), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, EPOCHS), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, EPOCHS), H.history["log_rmse"], label="train_log_rmse")
plt.plot(np.arange(0, EPOCHS), H.history["val_log_rmse"], label="val_log_rmse")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/log_rmse")
plt.legend(loc="lower left")
plt.savefig("plot.png")
plt.show()
