import pandas
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from deepctr.models import DeepFM
from deepctr.feature_column import SparseFeat, DenseFeat,get_feature_names
import tensorflow as tf
import horovod.tensorflow.keras as hvd
import openembedding.tensorflow as ps 
import argparse
opt=['Adam', 'Adagrad','Ftrl','SGD']
parser = argparse.ArgumentParser()
parser.add_argument('--pipe', action='store_true')
parser.add_argument('--hash', action='store_true')
parser.add_argument('--dense', action='store_true')
parser.add_argument('--shard', default=-1, type=int)
parser.add_argument('--load')
parser.add_argument('--save')
args = parser.parse_args()

if args.dense:
    sparse_as_dense_size = 256
else:
    sparse_as_dense_size = 1

ps.flags.config = '{"server":{"server_concurrency":4}}'
hvd.init()
# 将embeddings_regularizer当作activity_regularizer，大规模embedding不适用embeddings_regularizer。
class PSEmbedding(ps.Embedding):
    def __init__(self, input_dim, output_dim, embeddings_initializer=None, embeddings_regularizer=None, **kwargs):
        sparse_as_dense = False
        if input_dim <= sparse_as_dense_size:
            sparse_as_dense = True
        super(PSEmbedding, self).__init__(input_dim, output_dim, 
                embeddings_initializer=embeddings_initializer,
                activity_regularizer=embeddings_regularizer,
                sparse_as_dense=sparse_as_dense,
                num_shards=args.shard,
                **kwargs)

if __name__ == "__main__":
    if args.hash:
        import deepctr
        deepctr.inputs.Embedding = PSEmbedding

    print(hvd.rank())
    data = pandas.read_csv('./criteo_sample.txt')

    sparse_features = ['C' + str(i) for i in range(1, 27)]
    dense_features = ['I' + str(i) for i in range(1, 14)]

    data[sparse_features] = data[sparse_features].fillna('-1', )
    data[dense_features] = data[dense_features].fillna(0, )
    target = ['label']

    # 1.do simple Transformation for dense features
    input_dims = dict()
    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    for i, feat in enumerate(sparse_features):
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        input_dims[feat] = data[feat].max() + 1
        if args.hash and i % 2 == 0:
            input_dims[feat] = -1 # 同时测试hash table
    # 2.set hashing space for each sparse field,and record dense feature field name

    fixlen_feature_columns = [SparseFeat(feat, vocabulary_size=input_dims[feat], embedding_dim=4)
                              for feat in sparse_features] + [DenseFeat(feat, 1, ) for feat in dense_features]

    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns, )

    # 3.generate input data for model
    train_batch_size = 4
    train, test = train_test_split(data, test_size=0.2, shuffle=False)
    train = train[0: int(train.shape[0] / train_batch_size) * train_batch_size]
    train_model_input = {name:train[name] for name in feature_names}
    test_model_input = {name:test[name] for name in feature_names}


    # 4.Define Model,train,predict and evaluate
    optimizer = tf.keras.optimizers.Adam()
    optimizer = ps.distributed_optimizer(optimizer) #
    optimizer = hvd.DistributedOptimizer(optimizer)

    model = DeepFM(linear_feature_columns,dnn_feature_columns, task='binary')
    model = ps.distributed_model(model, sparse_as_dense_size=sparse_as_dense_size, num_shards=args.shard)
    model.compile(optimizer, "binary_crossentropy", metrics=['AUC'], experimental_run_tf_function=False)
    model.summary()

    train_batch_input = dict()
    for name, value in train_model_input.items():
        train_batch_input[name] = tf.reshape(value, [-1, train_batch_size])
    train_batch_target = tf.reshape(train[target].values, [-1, train_batch_size])
    dataset = tf.data.Dataset.from_tensor_slices((train_batch_input, train_batch_target))
    if args.pipe:
        dataset = ps.pulling(dataset, model).prefetch(4)
    if args.load:
        model.load_weights(args.load)
    callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
                  hvd.callbacks.MetricAverageCallback() ]
    history = model.fit(dataset, epochs=30, verbose=2, callbacks=callbacks)
    pred_ans = model.predict(test_model_input, batch_size=256)

    if args.save:
        if hvd.rank() == 0 and not args.hash:
            ps.save_as_original_model(model, args.save + 'standalone', include_optimizer=False)
        
        if hvd.rank() == 0:
            model.save_weights(args.save)

    # 测试save后是否可以正常训练
    model.fit(train_model_input, train[target].values,
          batch_size=256, epochs=2, verbose=2, validation_split=0.2, callbacks=callbacks)
    print("test LogLoss", round(log_loss(test[target].values, pred_ans), 4))
    print("test AUC", round(roc_auc_score(test[target].values, pred_ans), 4))
