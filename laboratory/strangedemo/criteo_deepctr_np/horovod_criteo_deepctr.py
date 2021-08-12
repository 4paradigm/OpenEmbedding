import pandas
import tensorflow as tf
import deepctr.models
import deepctr.feature_column
import horovod.tensorflow.keras as hvd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True) # 输入的数据文件
parser.add_argument('--learning_rate', required=True, type=float)
parser.add_argument('--batch_size', required=True, type=int)
parser.add_argument('--save', required=True)
args = parser.parse_args()

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank() % len(gpus)], 'GPU')

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

# 使用 horovod 实现数据并行
optimizer = hvd.DistributedOptimizer(optimizer, op=hvd.Sum)
n = data.shape[0] // hvd.size() * hvd.size()
for key in inputs.keys():
    inputs[key] = inputs[key][hvd.rank():n:hvd.size()]
labels = data['label'][hvd.rank():n:hvd.size()]
callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0),
             hvd.callbacks.MetricAverageCallback()]
model.compile(optimizer, "binary_crossentropy", metrics=['AUC'])
model.fit(inputs, labels, callbacks=callbacks,
          batch_size=args.batch_size, epochs=3, verbose=2)

if hvd.rank() == 0:
    model.save(args.save, overwrite=True, include_optimizer=False)
