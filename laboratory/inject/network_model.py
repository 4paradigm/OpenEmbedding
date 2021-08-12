import os
import pandas
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import deepctr.models
import deepctr.feature_column

import argparse
parser = argparse.ArgumentParser()
default_data = os.path.dirname(os.path.abspath(__file__)) + '/train100.csv'
parser.add_argument('--data', default=default_data) # 输入的数据文件
parser.add_argument('--optimizer', default='Adam')
parser.add_argument('--model', default='DeepFM')
parser.add_argument('--checkpoint', default='', help='checkpoint 保存路径') # include optimizer
parser.add_argument('--load', default='', help='要恢复的 checkpoint 路径') # include optimizer
parser.add_argument('--save', default='', help='分布式 serving model 保存的路径') # not include optimizer
args = parser.parse_args()
if not args.optimizer.endswith(')'):
    args.optimizer += '()' # auto call args.optimizer


# process data
hvd.init()
data = pandas.read_csv(args.data)
n = data.shape[0] // hvd.size()
data = data.iloc[hvd.rank() * n: hvd.rank() * n + n]
inputs = dict()
feature_columns = list()
for name in data.columns:
    if name[0] == 'C':
        inputs[name] = data[name] % 65536
        feature_columns.append(deepctr.feature_column.SparseFeat(name,
            vocabulary_size=65536, embedding_dim=9, dtype='int64'))
    elif name[0] == 'I':
        inputs[name] = data[name]
        feature_columns.append(deepctr.feature_column.DenseFeat(name, 1, dtype='float32'))


# compile distributed model
optimizer = eval("tf.keras.optimizers." + args.optimizer)
optimizer = hvd.DistributedOptimizer(optimizer)
model = eval("deepctr.models." + args.model)(feature_columns, feature_columns, task='binary')
model.compile(optimizer, "binary_crossentropy", metrics=['AUC'], experimental_run_tf_function=False)


# load --> fit --> save
callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
              hvd.callbacks.MetricAverageCallback() ]
if args.checkpoint and hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(args.checkpoint + '{epoch}'))
if args.load:
    model.load_weights(args.load)

model.fit(inputs, data['label'], batch_size=8, epochs=5, callbacks=callbacks, verbose=1)

if args.save and hvd.rank() == 0:
    model.save(args.save, include_optimizer=False)
