# -*- coding: utf-8 -*-
import pandas
import torch
import horovod.torch as hvd
from deepctr_torch.inputs import SparseFeat, DenseFeat
from deepctr_torch.models import WDL, DeepFM, xDeepFM

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--optimizer', default='Adagrad', choices=['Adagrad'])
parser.add_argument('--model', default="DeepFM", choices=["WDL", 'DeepFM', 'XDeepFM'])
parser.add_argument('--embedding_dim', default=64, type=int)
parser.add_argument('--batch_size', default=4096, type=int)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()
hvd.init()
if args.cpu:
    device = 'cpu'
else:
    #torch.cuda.set_device(hvd.local_rank())
    device = 'cuda:{}'.format(hvd.local_rank())

if __name__ == "__main__":
    data = pandas.read_csv(args.data)
    num_lines = data.shape[0]
    num_local_lines = int(num_lines / hvd.size()) // args.batch_size * args.batch_size
    local_start = hvd.local_rank() * num_local_lines
    local_end = local_start + num_local_lines
    print("num_lines:%d, num_local_lines:%d" % (num_lines, num_local_lines))
    print("local_start:%d, local_end:%d" % (local_start, local_end))
    
    target = ['label']
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    print(data.columns)
    
    feature_columns = []
    for name in sparse_features:
        feature_columns.append(SparseFeat(name, data[name].max() + 1, dtype='int64'))
    for name in dense_features:
        feature_columns.append(DenseFeat(name, 1, dtype='float32'))
    train = data.iloc[local_start:local_end]
    train_model_input = {name:train[name] for name in sparse_features + dense_features}

    if args.model == 'WDL':
        fc_sizes = (512, 256, 128, 32)
    elif args.model in {'DeepFM', 'xDeepFM'}:
        fc_sizes = (400, 400, 400)
    else:
        print("unknown model ", args.model)
    model = eval(args.model)(feature_columns, feature_columns, device=device,
          task='binary', dnn_hidden_units=fc_sizes, l2_reg_linear=0, l2_reg_embedding=0)

    optimizer = torch.optim.Adagrad(model.parameters())
    if hvd.size() > 1:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters(), op=hvd.Sum)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    model.compile(optimizer, "binary_crossentropy", metrics=["binary_crossentropy", "auc"])
    history = model.fit(train_model_input, train[target].values,
          batch_size=args.batch_size, epochs=args.epochs, verbose=2)
