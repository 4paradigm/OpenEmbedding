
import os 
import sys
import time
import psutil
import pathlib
import pandas
from deepctr.models import WDL, DeepFM, xDeepFM
from deepctr.feature_column import SparseFeat, DenseFeat
import tensorflow as tf
import horovod.tensorflow.keras as hvd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--optimizer', default='Adagrad', choices=['Adam', 'Adagrad','Ftrl','SGD'])
parser.add_argument('--model', default="DeepFM", choices=["WDL", 'DeepFM', 'xDeepFM'])
parser.add_argument('--embedding_dim', default=9, type=int)
parser.add_argument('--batch_size', default=4096, type=int)
parser.add_argument('--epochs', default=5, type=int)

parser.add_argument('--cpu', action='store_true')
parser.add_argument('--server', action='store_true')
parser.add_argument('--cache', action='store_true')
parser.add_argument('--prefetch', action='store_true')
parser.add_argument('--server_concurrency', default=28, type=int)

parser.add_argument('--profile', default='')
parser.add_argument('--master_endpoint', default='')
parser.add_argument('--bind_ip', default='')

parser.add_argument('--checkpoint', default='') # include optimizer
parser.add_argument('--load', default='') # include optimizer
parser.add_argument('--save', default='') # not include optimizer
parser.add_argument('--export', default='') # not include optimizer

# For paper experiment only
parser.add_argument('--pmem', default='')
parser.add_argument('--cache_size', default=500, type=int)
parser.add_argument('--auto_persist', action='store_true')


args = parser.parse_args()
hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if args.cpu:
        tf.config.experimental.set_visible_devices([], 'GPU')
    else:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

print("hvd.local_rank():%d" % hvd.local_rank())
print("hvd.rank():%d" % hvd.rank())
print("hvd.size():%d" % hvd.size())


if args.server:
    import deepctr
    import openembedding.tensorflow as embed
    if args.bind_ip:
        embed.flags.bind_ip = args.bind_ip
    if args.master_endpoint:
        embed.flags.master_endpoint = args.master_endpoint
        embed.flags.wait_num_servers = 1
    if args.pmem:
        embed.flags.config = ('{"server":{"server_concurrency":%d'
               ',"pmem_pool_root_path":"%s", "cache_size":%d } }') % (
              args.server_concurrency, args.pmem, args.cache_size)
    else:
        embed.flags.config = '{"server":{"server_concurrency":%d } }' % (
              args.server_concurrency)

    # 将embeddings_regularizer当作activity_regularizer，大规模embedding不适用embeddings_regularizer。
    class PSEmbedding(embed.Embedding):
        def __init__(self, input_dim, output_dim,
              embeddings_initializer=None, embeddings_regularizer=None, **kwargs):
            sparse_as_dense = False
            if args.cache and input_dim < args.batch_size:
                sparse_as_dense = True
            embeddings_initializer = tf.keras.initializers.zeros()
            super(PSEmbedding, self).__init__(input_dim, output_dim, 
                    embeddings_initializer=embeddings_initializer,
                    activity_regularizer=embeddings_regularizer,
                    sparse_as_dense=sparse_as_dense,
                    num_shards=1,
                    **kwargs)
    deepctr.inputs.Embedding = PSEmbedding
else:
    import deepctr
    # 将embeddings_regularizer当作activity_regularizer，大规模embedding不适用embeddings_regularizer。
    class TFEmbedding(tf.keras.layers.Embedding):
        def __init__(self, input_dim, output_dim,
              embeddings_initializer=None, embeddings_regularizer=None, **kwargs):
            embeddings_initializer = tf.keras.initializers.zeros()
            super(TFEmbedding, self).__init__(input_dim, output_dim, 
                    embeddings_initializer=embeddings_initializer,
                    activity_regularizer=embeddings_regularizer, 
                    **kwargs)
    deepctr.inputs.Embedding = TFEmbedding


if args.pmem:
    def save_server_model(model, filepath, include_optimizer=True):
        if args.auto_persist:
            embed.persist_server_model(model, filepath, 2)
        else:
            embed.persist_server_model(model, filepath, 0)
    embed.exb.save_server_model = save_server_model
    embed.exb.load_server_model = embed.restore_server_model

    class AutoPersist(tf.keras.callbacks.Callback):
        def __init__(self, path):
            self.path = path
            self.persist_no = 0

        def on_train_batch_end(self, batch, logs=None):
            if embed.should_persist_server_model(self.model):
                self.persist_no += 1
                self.model.save(self.path + str(self.persist_no))


target = ['label']
dense_features = ['I' + str(i) for i in range(1, 14)]
sparse_features = ['C' + str(i) for i in range(1, 27)]

def get_csv_dataset():
    vocabulary = {}
    data = pandas.read_csv(args.data)
    num_lines = data.shape[0]
    num_local_lines = int(num_lines / hvd.size()) // args.batch_size * args.batch_size
    local_start = hvd.local_rank() * num_local_lines
    local_end = local_start + num_local_lines
    print("num_lines:%d, num_local_lines:%d" % (num_lines, num_local_lines))
    print("local_start:%d, local_end:%d" % (local_start, local_end))
    for name in sparse_features:
        vocabulary[name] = data[name].max() + 1

    train = data.iloc[local_start:local_end]
    train_batch_input = {}
    for name in dense_features + sparse_features:
        train_batch_input[name] = tf.reshape(train[name], [-1, args.batch_size, 1])
        if name[0] == 'I':
            train_batch_input[name] = tf.cast(train_batch_input[name], dtype="float32")
        else:
            train_batch_input[name] = tf.cast(train_batch_input[name], dtype="int64")
    train_batch_target = tf.reshape(train[target].values, [-1, args.batch_size])
    dataset = tf.data.Dataset.from_tensor_slices((train_batch_input, train_batch_target))
    return dataset.prefetch(16), vocabulary


dense_b = ['x', 0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
dense_k = ['x', 5775, 257675, 65535, 969, 23159456, 431037, 56311, 6047, 29019, 11, 231, 4008, 7393]
def get_tf_dataset():
    vocabulary = {}
    columns = {'label':tf.io.FixedLenFeature([1], tf.int64)}
    for name in dense_features:
        columns[name] = tf.io.FixedLenFeature([1], tf.float32)
    for name in sparse_features:
        columns[name] = tf.io.FixedLenFeature([1], tf.int64)

    for line in open(args.data + '/meta'):
        feature, count = line.split()
        vocabulary[feature] = int(count)
    files = []
    num_examples = 0
    for i, path in enumerate(sorted(pathlib.Path(args.data).glob('tf-*'))):
        num_examples += 1000000
        if i % hvd.size() == hvd.rank():
            files.append(str(path))
            print(str(path))

    parallel = 4
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.interleave(lambda path: tf.data.TFRecordDataset(path, buffer_size=2**20),
        cycle_length=parallel, block_length=1024, num_parallel_calls=parallel)
    dataset = dataset.batch(args.batch_size).repeat()
    def parser(examples):
        tensors = tf.io.parse_example(examples, columns)
        labels = tensors.pop('label')
        return tensors, labels
    dataset = dataset.map(parser, num_parallel_calls=parallel)
    dataset = dataset.prefetch(parallel * 4)
    steps_per_epoch = num_examples // hvd.size() // args.batch_size + 1
    return dataset, vocabulary, steps_per_epoch


def get_origin_dataset():
    vocabulary = { name:-1 for name in sparse_features + dense_features }
    for line in open(args.data + '/meta'):
        feature, count = line.split()
        vocabulary[feature] = int(count)
    files = []
    num_examples = 0
    for i, path in enumerate(sorted(pathlib.Path(args.data).glob('day_*'))):
        num_examples += 150000000
        if i % hvd.size() == hvd.rank() and not str(path).endswith('.gz'):
            files.append(str(path))
            print(str(path))
    parallel = 4
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.interleave(lambda path: tf.data.TextLineDataset(path, buffer_size=2**20),
        cycle_length=parallel, block_length=1024, num_parallel_calls=parallel)
    dataset = dataset.batch(args.batch_size).repeat()
    def mapper(lines):
        dtypes = [int()] + [float() for name in dense_features] + [str() for name in sparse_features]
        columns = tf.io.decode_csv(lines, field_delim='\t', record_defaults=dtypes) 
        examples = {}
        for i, name in enumerate(target + dense_features + sparse_features):
            if name[0] == 'I':
                examples[name] = tf.cast((columns[i] - dense_b[i]) / dense_k[i], dtype=tf.float32)
            elif name[0] == 'C':
                examples[name] = tf.strings.to_hash_bucket_fast(columns[i], 2**62)
        return examples, columns[0]
    dataset = dataset.map(mapper, num_parallel_calls=parallel * 2)
    dataset = dataset.prefetch(parallel * 6)
    steps_per_epoch = num_examples // hvd.size() // args.batch_size + 1
    return dataset, vocabulary, steps_per_epoch


def get_model(vocabulary):
    feature_columns = []
    for name in sparse_features:
        feature_columns.append(SparseFeat(name,
              vocabulary_size=vocabulary[name], embedding_dim=args.embedding_dim, dtype='int64'))
    for name in dense_features:
        feature_columns.append(DenseFeat(name, 1, dtype='float32'))

    if 'Adam' == args.optimizer: 
        optimizer = tf.keras.optimizers.Adam()
    elif 'Ftrl' == args.optimizer:
        optimizer = tf.keras.optimizers.Ftrl()
    elif 'SGD' == args.optimizer:
        optimizer = tf.keras.optimizers.SGD()
    elif 'Adagrad' == args.optimizer:
        optimizer = tf.keras.optimizers.Adagrad()
    else:
        print('unknown optimizer ', args.optimizer)

    if args.server:
        optimizer = embed.distributed_optimizer(optimizer)
    if hvd.size() > 1:
        optimizer = hvd.DistributedOptimizer(optimizer, op=hvd.Sum)
    
    if args.model == 'WDL':
        fc_sizes = (512, 256, 128, 32)
    elif args.model in {'DeepFM', 'xDeepFM'}:
        fc_sizes = (400, 400, 400)
    else:
        print("unknown model ", args.model)
    model = eval(args.model)(feature_columns, feature_columns,
          task='binary', dnn_hidden_units=fc_sizes, l2_reg_linear=0, l2_reg_embedding=0)

    if args.server:
        model = embed.distributed_model(model)
    if hvd.size() > 1:
        model.compile(optimizer, "binary_crossentropy", metrics=['AUC'], experimental_run_tf_function=False)
    else:
        model.compile(optimizer, "binary_crossentropy", metrics=['AUC'])
    return model


def get_callbacks():
    callbacks = []
    if hvd.size() > 1:
        callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                      hvd.callbacks.MetricAverageCallback() ]
    if args.profile:
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
              log_dir=args.profile, profile_batch='100,110')
        callbacks.append(tensorboard_callback)
    if args.checkpoint and hvd.rank() == 0:
        if args.auto_persist:
            callbacks.append(AutoPersist(args.checkpoint))
        else:
            callbacks.append(tf.keras.callbacks.ModelCheckpoint(args.checkpoint + '{epoch}')) #include optimizer
    return callbacks


def print_rss():
    print(psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024, 'GB', flush=True)


if __name__ == "__main__":
    print_rss()
    
    with tf.device('CPU:0'):
        if args.data.endswith('csv'):
            dataset, vocabulary = get_csv_dataset()
        elif args.data.endswith('1T'):
            dataset, vocabulary, steps_per_epoch = get_origin_dataset()
        else:
            dataset, vocabulary, steps_per_epoch = get_tf_dataset()
    print_rss()

    model = get_model(vocabulary)
    if args.load:
        model.load_weights(args.load)
    print_rss()
    
    if args.data.endswith('csv'):
        with tf.device('CPU:0'):
            if args.server and args.prefetch:
                dataset = embed.pulling(dataset, model).prefetch(4)
        history = model.fit(dataset, verbose=2,
              callbacks=get_callbacks(), epochs=args.epochs)
    else:
        with tf.device('CPU:0'):
            if args.server and args.prefetch:
                dataset = embed.pulling(dataset, model, steps=args.epochs * steps_per_epoch).prefetch(4)
        history = model.fit(dataset, verbose=2,
                callbacks=get_callbacks(), epochs=args.epochs, steps_per_epoch=steps_per_epoch)
    print_rss()
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Embedding) and layer.output_dim < 2**31:
            print(layer.name, layer.call(tf.constant(layer.input_dim - 1)))
 
    if args.save and hvd.rank() == 0:
        model.save(args.save, include_optimizer=False)
        print_rss()

    if args.export and hvd.rank() == 0:
        model.save_as_original_model(args.export, include_optmizer=False)
        print_rss()
