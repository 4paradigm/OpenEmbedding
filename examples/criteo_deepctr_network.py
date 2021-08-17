import os
import pandas
import tensorflow as tf
import deepctr.models
import deepctr.feature_column
import horovod.tensorflow.keras as hvd
import openembedding.tensorflow as embed
print('OpenEmbedding: ', embed.__version__)


import argparse
parser = argparse.ArgumentParser()
default_data = os.path.dirname(os.path.abspath(__file__)) + '/train100.csv'
parser.add_argument('--data', default=default_data)
parser.add_argument('--batch_size', default=8, type=int)
parser.add_argument('--optimizer', default='Adam')
parser.add_argument('--model', default='DeepFM')
parser.add_argument('--checkpoint', default='', help='checkpoint save path') # include optimizer
parser.add_argument('--load', default='', help='checkpoint path to restore') # include optimizer
parser.add_argument('--save', default='', help='distributed serving model save path') # not include optimizer
parser.add_argument('--export', default='', help='standalone serving model save path') # not include optimizer
args = parser.parse_args()
if not args.optimizer.endswith(')'):
    args.optimizer += '()' # auto call args.optimizer


# Assign GPU according to rank.
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[hvd.local_rank()], True)


# Process data.
data = pandas.read_csv(args.data)
n = data.shape[0] // hvd.size() * hvd.size()
data = data.iloc[hvd.rank():n:hvd.size()]
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


# Compile distributed model.
optimizer = eval("tf.keras.optimizers." + args.optimizer)
optimizer = embed.distributed_optimizer(optimizer)
optimizer = hvd.DistributedOptimizer(optimizer, op=hvd.Sum)
model = eval("deepctr.models." + args.model)(feature_columns, feature_columns, task='binary')
model = embed.distributed_model(model)
model.compile(optimizer, "binary_crossentropy", metrics=['AUC'], experimental_run_tf_function=False)


# load --> fit --> save
callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
              hvd.callbacks.MetricAverageCallback() ]
if args.checkpoint and hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(args.checkpoint + '{epoch}'))
if args.load:
    model.load_weights(args.load)

model.fit(inputs, data['label'], batch_size=args.batch_size, epochs=5, callbacks=callbacks, verbose=2)

if args.save and hvd.rank() == 0:
    model.save(args.save, include_optimizer=False)
if args.export and hvd.rank() == 0:
    model.save_as_original_model(args.export, include_optimizer=False)
