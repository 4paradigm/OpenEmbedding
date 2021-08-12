import pandas
import tensorflow as tf
import deepctr.models
import deepctr.feature_column

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True) # 输入的数据文件
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--save', required=True)
args = parser.parse_args()

data = pandas.read_csv(args.data)
inputs = dict()
feature_columns = list()
for name in data.columns:
    if name[0] == 'C':
        inputs[name] = data[name]
        feature_columns.append(deepctr.feature_column.SparseFeat(name,
            vocabulary_size=data[name].max() + 1, embedding_dim=64, dtype='int64'))
    elif name[0] == 'I':
        inputs[name] = data[name]
        feature_columns.append(deepctr.feature_column.DenseFeat(name, 1, dtype='float32'))

optimizer = tf.keras.optimizers.Adagrad(args.learning_rate)
model = deepctr.models.DeepFM(feature_columns, feature_columns, task='binary', 
    l2_reg_linear=0, l2_reg_embedding=0, l2_reg_dnn=0)
model.compile(optimizer, 'binary_crossentropy', metrics=['AUC'])
model.fit(inputs, data['label'], batch_size=args.batch_size, epochs=3, verbose=2)
model.save(args.save, overwrite=True, include_optimizer=False)
