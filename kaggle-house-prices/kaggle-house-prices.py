import numpy as np
import pandas as pd
from net.housepricesnet import HousePricesNet

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


model = HousePricesNet.build()


