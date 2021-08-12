import sys
import pandas
import tensorflow as tf

if len(sys.argv) < 4:
    print('usage: criteo_tfrecord.py input pid np')

def serialize_example(train, j):
    fea_desc = {}
    for name, column in train.items():
        if name[0] == 'I':
            # dense feature
            fea_desc[name] = tf.train.Feature(float_list=tf.train.FloatList(value=[float(column[j])]))
        else:
            # label or sparse feature
            fea_desc[name] = tf.train.Feature(int64_list=tf.train.Int64List(value=[int(column[j])]))
    example_proto = tf.train.Example(features=tf.train.Features(feature=fea_desc))
    return example_proto.SerializeToString()

data = pandas.read_csv(sys.argv[1])
pid = int(sys.argv[2])
np = int(sys.argv[3])

target = ['label']
dense_features = ['I' + str(i) for i in range(1, 14)]
sparse_features = ['C' + str(i) for i in range(1, 27)]
columns = target + dense_features + sparse_features
train = {name:data[name] for name in columns}

count = 1000000
for start in range(count * pid, data.shape[0], count * np):
    end = start + count
    if end > data.shape[0]:
        end = data.shape[0]
    name = str(start // count + 1)
    while len(name) < 5:
        name = '0' + name
    with tf.io.TFRecordWriter("./tfrecord/tf-part.{}".format(name)) as writer:
        for j in range(start, end):
            example = serialize_example(train, j)
            writer.write(example)

if pid == 0:
    with open('./tfrecord/meta', 'w') as writer:
        for name in sparse_features:
            writer.write(name, data[name].max() + 1)
