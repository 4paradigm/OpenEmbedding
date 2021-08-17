import os
import pandas
import tensorflow as tf
import deepctr.models
import deepctr.feature_column
import openembedding.tensorflow as embed
print('OpenEmbedding: ', embed.__version__)


import argparse
parser = argparse.ArgumentParser()
default_data = os.path.dirname(os.path.abspath(__file__)) + '/train100.csv'
parser.add_argument('--data', default=default_data)
parser.add_argument('--batch_size', default=8, type=int)
# Currently, MirroredStrategy does not support this.
# parser.add_argument('--prefetch', action='store_true')
parser.add_argument('--optimizer', default='Adam')
parser.add_argument('--model', default='DeepFM')
parser.add_argument('--checkpoint', default='', help='checkpoint save path') # include optimizer
parser.add_argument('--load', default='', help='checkpoint path to restore') # include optimizer
parser.add_argument('--save', default='', help='distributed serving model save path') # not include optimizer
parser.add_argument('--export', default='', help='standalone serving model save path') # not include optimizer
args = parser.parse_args()
if not args.optimizer.endswith(')'):
    args.optimizer += '()' # auto call args.optimizer

# Process data
data = pandas.read_csv(args.data)
data = data.iloc[:data.shape[0] // args.batch_size * args.batch_size]
inputs = dict()
feature_columns = list()
for name in data.columns:
    inputs[name] = tf.reshape(data[name], [-1, args.batch_size, 1])
    if name[0] == 'C':
        inputs[name] = tf.cast(inputs[name] % 65536, dtype=tf.int64)
        feature_columns.append(deepctr.feature_column.SparseFeat(name,
            vocabulary_size=65536, embedding_dim=9, dtype='int64'))
    elif name[0] == 'I':
        inputs[name] = tf.cast(inputs[name], dtype=tf.float32)
        feature_columns.append(deepctr.feature_column.DenseFeat(name, 1, dtype='float32'))
train_batch_target = tf.reshape(data['label'], [-1, args.batch_size])
dataset = tf.data.Dataset.from_tensor_slices((inputs, train_batch_target))


# Compile distributed model
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    optimizer = eval("tf.keras.optimizers." + args.optimizer)
    optimizer = embed.distributed_optimizer(optimizer)

    model = eval("deepctr.models." + args.model)(feature_columns, feature_columns, task='binary')
    model = embed.distributed_model(model, sparse_as_dense_size=args.batch_size)
    model.compile(optimizer, "binary_crossentropy", metrics=['AUC'], experimental_run_tf_function=False)


# load --> fit --> save
callbacks = list()
if args.checkpoint:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(args.checkpoint + '{epoch}'))
if args.load:
    model.load_weights(args.load)


# Currently, MirroredStrategy does not support this.
# if args.prefetch:
#     dataset = embed.pulling(dataset, model).prefetch(4)
model.fit(dataset, batch_size=args.batch_size, epochs=5, callbacks=callbacks, verbose=2)

if args.save:
    model.save(args.save, include_optimizer=False)
if args.export:
    model.save_as_original_model(args.export, include_optimizer=False)
