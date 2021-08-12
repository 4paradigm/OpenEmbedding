import os
import sys
import grpc
import numpy
import pandas
import threading
import tensorflow as tf


from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc


class _ResultCounter(object):
    def __init__(self, num_tests, concurrency):
        self._num_tests = num_tests
        self._concurrency = concurrency
        self._error = 0
        self._done = 0
        self._active = 0
        self._condition = threading.Condition()

    def inc_error(self):
        with self._condition:
            self._error += 1

    def inc_done(self):
        with self._condition:
            self._done += 1
            self._condition.notify()

    def dec_active(self):
        with self._condition:
            self._active -= 1
            self._condition.notify()

    def get_error_rate(self):
        with self._condition:
            while self._done != self._num_tests:
                self._condition.wait()
            return self._error / float(self._num_tests)

    def throttle(self):
        with self._condition:
            while self._active == self._concurrency:
                self._condition.wait()
            self._active += 1


def _create_rpc_callback(label, result_counter):
    def _callback(result_future):
        exception = result_future.exception()
        if exception:
            result_counter.inc_error()
            print(exception)
        else:
            predict = numpy.array(result_future.result().outputs['prediction_layer'].float_val)
            print('label = ', label, ', predict = ', int(predict[0] + 0.5), ' ' ,predict[0])
            if label != int(predict[0] + 0.5):
                result_counter.inc_error()
            result_counter.inc_done()
            result_counter.dec_active()
    return _callback
  

import argparse
parser = argparse.ArgumentParser()
default_data = os.path.dirname(os.path.abspath(__file__)) + '/train100.csv'
parser.add_argument('--data', default=default_data)
parser.add_argument('--hash', action='store_true')
parser.add_argument('--grpc', default='127.0.0.1:8500')
parser.add_argument('--model', default='criteo')
args = parser.parse_args()


# process data
data = pandas.read_csv(args.data)
feature_names = list()
for name in data.columns:
    if name[0] == 'C':
        if args.hash:
            data[name] = (data[name] + int(name[1:]) * 1000000007) % (2**63)
        else:
            data[name] = data[name] % 65536
        feature_names.append(name)
    elif name[0] == 'I':
        feature_names.append(name)


# use TensorFlow Serving
channel = grpc.insecure_channel(args.grpc)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
result_counter = _ResultCounter(data.shape[0], 4)
for i in range(data.shape[0]):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = args.model
    for name in feature_names:
        dtype = tf.float32
        if name.startswith('C'):
            dtype = tf.int64
        request.inputs[name].CopyFrom(tf.make_tensor_proto(data[name][i], dtype=dtype, shape=[1, 1]))
    result_counter.throttle()
    result_future = stub.Predict.future(request, 5.0)  # 5 seconds
    result_future.add_done_callback(
        _create_rpc_callback(data['label'][i], result_counter))
print('error rate: ', result_counter.get_error_rate())
