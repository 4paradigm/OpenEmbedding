import os
import pandas
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import openembedding.tensorflow as embed


class CriteoLR(tf.keras.Model):
    def __init__(self):
        super(CriteoLR, self).__init__()
        self.embeddings = embed.Embedding(-1, 1,
              embeddings_initializer=tf.keras.initializers.Zeros(), num_shards=16)
        self.concatenate = tf.keras.layers.Concatenate()
        self.sigmoid = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        fields = []
        for name, tensor in inputs.items():
            if name[0] == 'C':
                fields.append(self.embeddings(tensor))
            else:
                fields.append(tf.reshape(tensor, [-1, 1, 1]))
        return self.sigmoid(self.concatenate(fields))


import argparse
parser = argparse.ArgumentParser()
default_data = os.path.dirname(os.path.abspath(__file__)) + '/train100.csv'
parser.add_argument('--data', default=default_data)
parser.add_argument('--optimizer', default='Adam')
parser.add_argument('--checkpoint', default='') # include optimizer
parser.add_argument('--load', default='') # include optimizer
parser.add_argument('--save', default='') # not include optimizer
# 因为使用了 subclass Model，所以不支持导出为 tensorflow 原始模型。
# parser.add_argument('--export', default='') # not include optimizer
args = parser.parse_args()
if not args.optimizer.endswith(')'):
    args.optimizer += '()' # auto call args.optimizer


# process data
hvd.init()
data = pandas.read_csv(args.data)
n = data.shape[0] // hvd.size() * hvd.size()
data = data.iloc[hvd.rank():n:hvd.size()]
inputs = dict()
for name in data.columns:
    if name[0] == 'C':
        inputs[name] = (data[name] + int(name[1:]) * 1000000007) % (2**63) # hash encoding
    elif name[0] == 'I':
        inputs[name] = data[name]


# compile distributed model
optimizer = eval("tf.keras.optimizers." + args.optimizer)
optimizer = embed.distributed_optimizer(optimizer)
optimizer = hvd.DistributedOptimizer(optimizer, op=hvd.Sum)
model = CriteoLR()
model = embed.distributed_model(model)
model.compile(optimizer, "binary_crossentropy", metrics=['AUC'], experimental_run_tf_function=False)


# load --> fit --> save
callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
              hvd.callbacks.MetricAverageCallback() ]
if args.checkpoint and hvd.rank() == 0:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(args.checkpoint + '{epoch}'))
if args.load:
    model.load_weights(args.load)

model.fit(inputs, data['label'], batch_size=8, epochs=5, callbacks=callbacks, verbose=2)
if args.save and hvd.rank() == 0:
    model.save(args.save, include_optimizer=False)
