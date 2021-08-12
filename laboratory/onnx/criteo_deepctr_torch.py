# -*- coding: utf-8 -*-
import pandas
import torch
# import horovod.torch as hvd
import time
import numpy as np
import sklearn
import deepctr_torch as deepctr

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data', required=True)
parser.add_argument('--optimizer', default='Adagrad', choices=['Adagrad'])
parser.add_argument('--model', default="DeepFM", choices=["WDL", 'DeepFM', 'XDeepFM'])
parser.add_argument('--embedding_dim', default=9, type=int)
parser.add_argument('--batch_size', default=4096, type=int)
parser.add_argument('--epochs', default=2, type=int)
parser.add_argument('--onnx', action='store_true')
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()
# hvd.init()
# if args.cpu:
#     device = 'cpu'
# else:
#    # torch.cuda.set_device(hvd.local_rank())
#    device = 'cuda:{}'.format(hvd.local_rank())

device = 'cuda'
def train_model(model, x, y, batch_size, epochs=1, optimizer=torch.optim.Adagrad):
    x = [np.expand_dims(tensor, 1) for tensor in x]
    x = torch.from_numpy(np.concatenate(x, axis=-1))
    y = torch.from_numpy(y)
    train_tensor_data = torch.utils.data.TensorDataset(x, y)
    train_loader = torch.utils.data.DataLoader(dataset=train_tensor_data, batch_size=batch_size)
    loss_func = torch.nn.functional.binary_cross_entropy
    for epoch in range(epochs):
        start_time = time.time()
        epoch_loss = 0.0
        epoch_auc = 0.0
        for x_train, y_train in train_loader:
            x_train = x_train.to(device).float()
            y_train = y_train.to(device).float()
            y_pred = model(x_train).to(device).squeeze()
            optimizer.zero_grad()
            loss = loss_func(y_pred, y_train.squeeze(), reduction='sum')
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            # train_result["AUC"].append(sklearn.metrics.roc_auc_score(
            #       y.cpu().data.numpy(), y_pred.cpu().data.numpy().astype("float64")))

        epoch_time = int(time.time() - start_time)
        print('Epoch {0}/{1}'.format(epoch + 1, epochs))
        eval_str = "{0}s - loss: {1: .4f}".format(epoch_time, epoch_loss)
        # eval_str += " - " + name + ": {0: .4f}".format(epoch_logs[name])
        print(eval_str)


if __name__ == "__main__":
    data = pandas.read_csv(args.data)
    num_lines = data.shape[0]
    num_local_lines = num_lines // args.batch_size * args.batch_size
    local_start = 0
    # num_local_lines = int(num_lines / hvd.size()) // args.batch_size * args.batch_size
    # local_start = hvd.local_rank() * num_local_lines
    local_end = local_start + num_local_lines
    print("num_lines:%d, num_local_lines:%d" % (num_lines, num_local_lines))
    print("local_start:%d, local_end:%d" % (local_start, local_end))
    
    target = ['label']
    dense_features = ['I' + str(i) for i in range(1, 14)]
    sparse_features = ['C' + str(i) for i in range(1, 27)]
    print(data.columns)
    
    feature_columns = []
    for name in sparse_features:
        feature_columns.append(deepctr.inputs.SparseFeat(name, data[name].max() + 1, dtype='int64'))
    for name in dense_features:
        feature_columns.append(deepctr.inputs.DenseFeat(name, 1, dtype='float32'))
    train = data.iloc[local_start:local_end]
    train_model_input = {name:train[name] for name in sparse_features + dense_features}

    if args.model == 'WDL':
        fc_sizes = (512, 256, 128, 32)
    elif args.model in {'DeepFM', 'xDeepFM'}:
        fc_sizes = (400, 400, 400)
    else:
        print("unknown model ", args.model)
    model = eval("deepctr.models." + args.model)(feature_columns, feature_columns, device=device,
          task='binary', dnn_hidden_units=fc_sizes, l2_reg_linear=0, l2_reg_embedding=0)
    x = [train_model_input[name] for name in model.feature_index]
    if args.onnx:
        from onnxruntime.training.ortmodule import ORTModule
        model = ORTModule(model)
    optimizer=torch.optim.Adagrad(model.parameters())
    train_model(model, x, train[target].values,
          batch_size=args.batch_size, epochs=args.epochs, optimizer=optimizer)
