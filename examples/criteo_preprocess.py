import sys
import pandas
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

if len(sys.argv) < 2:
    print("usage: process_data.py input_file output_file")

data = pandas.read_csv(sys.argv[1], sep='\t', header=None)
target = ['label']
dense_features = ['I' + str(i) for i in range(1, 14)]
sparse_features = ['C' + str(i) for i in range(1, 27)]
data.columns = target + dense_features + sparse_features

data[sparse_features] = data[sparse_features].fillna('-1', )
data[dense_features] = data[dense_features].fillna(0, )

for feat in dense_features:
    print(feat, data[feat].min(), data[feat].max())
mms = MinMaxScaler(feature_range=(0, 1))
data[dense_features] = mms.fit_transform(data[dense_features])
for feat in sparse_features:
    lbe = LabelEncoder()
    data[feat] = lbe.fit_transform(data[feat])

data.to_csv(sys.argv[2], float_format='%.6f')
