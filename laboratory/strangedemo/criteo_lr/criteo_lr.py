import pandas
import tensorflow as tf

def CriteoLR(features, input_dim):
    embeddings = tf.keras.layers.Embedding(input_dim, 1,
          embeddings_initializer=tf.keras.initializers.Zeros())
    fields = list()
    for name, tensor in features.items():
        if name[0] == 'C':
            fields.append(embeddings(tensor))
        else:
            fields.append(tf.reshape(tensor, [-1, 1, 1]))
    concat = tf.keras.layers.concatenate(fields)
    output = tf.keras.layers.Dense(1, activation='sigmoid')(concat)
    return tf.keras.models.Model(inputs=features, outputs=[output])

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--learning_rate', type=float, required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--save', required=True)
args = parser.parse_args()
data = pandas.read_csv(args.data)
inputs = dict()
features = dict()
vocabulary_size = 0
for name in data.columns:
    if name[0] == 'C':
        inputs[name] = data[name] + vocabulary_size
        features[name] = tf.keras.Input(shape=[1], name=name, dtype=tf.int64)
        vocabulary_size += data[name].max() + 1
    elif name[0] == 'I':
        inputs[name] = data[name]
        features[name] = tf.keras.Input(shape=[1], name=name, dtype=tf.float32)

# compile distributed model
optimizer = tf.keras.optimizers.Adagrad(args.learning_rate)
model = CriteoLR(features, vocabulary_size)
model.compile(optimizer, 'binary_crossentropy')
model.fit(inputs, data['label'], batch_size=args.batch_size, epochs=5, verbose=2)
model.save(args.save, overwrite=True, include_optimizer=False)
