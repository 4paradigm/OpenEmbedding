import os
import pandas
import tensorflow as tf
import openembedding.tensorflow as embed
print('OpenEmbedding', embed.__version__)


class CriteoLR(tf.keras.Model):
    def __init__(self):
        super(CriteoLR, self).__init__()
        # input_dim = -1 means that the input range is the natural number range of int64 [0, 2**63-1].
        # If input_dim = -1, the server will uses hash table to store Embedding layer,
        self.embeddings = embed.Embedding(input_dim=-1, output_dim=1,
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
parser.add_argument('--checkpoint', default='') # include optimizer
parser.add_argument('--load', default='') # include optimizer
parser.add_argument('--save', default='') # not include optimizer
# subclass model not support exporting to tensorflow original model
# parser.add_argument('--export', default='') # not include optimizer
args = parser.parse_args()


# Process data
data = pandas.read_csv(args.data)
inputs = dict()
for name in data.columns:
    if name[0] == 'C':
        inputs[name] = (data[name] + int(name[1:]) * 1000000007) % (2**63) # hash encoding
    elif name[0] == 'I':
        inputs[name] = data[name]


# Compile distributed model
optimizer = tf.keras.optimizers.Adam()
optimizer = embed.distributed_optimizer(optimizer)
model = CriteoLR()
model = embed.distributed_model(model)
model.compile(optimizer, "binary_crossentropy", metrics=['AUC'], experimental_run_tf_function=False)


# load --> fit --> save
callbacks = list()
if args.checkpoint:
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(args.checkpoint + '{epoch}'))
if args.load:
    model.load_weights(args.load)

model.fit(inputs, data['label'], batch_size=8, epochs=5, callbacks=callbacks, verbose=2)
if args.save:
    model.save(args.save, include_optimizer=False)
