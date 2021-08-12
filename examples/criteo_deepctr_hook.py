import os
import pandas
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import openembedding.tensorflow as embed
import deepctr.models
import deepctr.feature_column


import argparse
parser = argparse.ArgumentParser()
default_data = os.path.dirname(os.path.abspath(__file__)) + '/train100.csv'
parser.add_argument('--data', default=default_data)
parser.add_argument('--optimizer', default='Adam')
parser.add_argument('--model', default='DeepFM')
parser.add_argument('--checkpoint', default='') # include optimizer
parser.add_argument('--load', default='') # include optimizer
parser.add_argument('--save', default='') # not include optimizer

parser.add_argument('--batch_size', default=8, type=int)
# 因为 server 使用了 hash table 存储数据，所以不支持导出为 tensorflow 原始模型。
# parser.add_argument('--export', default='') # not include optimizer
args = parser.parse_args()
if not args.optimizer.endswith(')'):
    args.optimizer += '()' # auto call args.optimizer


# hook deepctr.inputs.Embedding
class HookEmbedding(embed.Embedding):
    def __init__(self, input_dim=-1, output_dim=9,
            embeddings_initializer=None, embeddings_regularizer=None, **kwargs):
        # input_dim = -1 表示 input 范围是 int64 的自然数范围 [0, 2**63 - 1]
        # input_dim = -1 时，server 使用 hash table 存储 Embedding 层数据
        # server 端不支持 embeddings_regularizer
        # 可以通过 num_shards 指定全局的 shard 数量，默认等于 server 数量
        super(HookEmbedding, self).__init__(input_dim, output_dim, 
                embeddings_initializer=embeddings_initializer, 
                activity_regularizer=embeddings_regularizer, 
                num_shards=1, 
                **kwargs)
import deepctr.inputs
deepctr.inputs.Embedding = HookEmbedding


# process data
hvd.init()
data = pandas.read_csv(args.data)
n = data.shape[0] // hvd.size() * hvd.size()
data = data.iloc[hvd.rank():n:hvd.size()]
inputs = dict()
feature_columns = list()
for name in data.columns:
    if name[0] == 'C':
        inputs[name] = (data[name] + int(name[1:]) * 1000000007) % (2**63) # hash encoding
        feature_columns.append(deepctr.feature_column.SparseFeat(name,
            vocabulary_size=-1, embedding_dim=9, dtype='int64'))
    elif name[0] == 'I':
        inputs[name] = data[name]
        feature_columns.append(deepctr.feature_column.DenseFeat(name, 1, dtype='float32'))


# compile distributed model
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
