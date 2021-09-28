import sys
import tensorflow as tf
import openembedding.tensorflow as embed


def run_tf_optimizer(optimizer, gradients):
    optimizer = optimizer.__class__.from_config(optimizer.get_config())
    var = tf.Variable(tf.ones(gradients[0].shape, gradients[0].dtype))
    for grad in gradients:
        optimizer.apply_gradients([(grad, var)])
    return var.read_value()


def run_my_optimizer(optimizer, gradients):
    var = embed.Embedding(gradients[0].shape[0], gradients[0].shape[1],
          tf.keras.initializers.Constant(1.0), dtype=gradients[0].dtype)
    indices = tf.range(var.input_dim)
    var.build(indices.shape)
    var.variable.set_server_optimizer(optimizer)
    for grad in gradients:
        fakegrad = var.variable.push_gradients(indices, grad)
        var.variable.update_weights(fakegrad)
    return var.variable.sparse_read(indices)

from tensorflow.keras.optimizers import *

gradients1d = [ tf.ones([1, 1], dtype=tf.float64) ]
gradients10d = [ tf.random.uniform([111, 11], -1, 1, dtype=tf.float64) for i in range(10) ]
gradients100d = [ tf.random.uniform([111, 11], -1, 1, dtype=tf.float64) for i in range(100) ]
gradients1 = [ tf.cast(tensor, dtype=tf.float32) for tensor in gradients1d ]
gradients10 = [ tf.cast(tensor, dtype=tf.float32) for tensor in gradients10d ]
gradients100 = [ tf.cast(tensor, dtype=tf.float32) for tensor in gradients100d ]
optimizers = [
    Adadelta(), Adadelta(0.1), Adadelta(0.1, rho=0.8),
    Adagrad(), Adagrad(0.1), Adagrad(0.1, 1000),
    Adam(), Adam(0.1), Adam(0.1, beta_1=0.8, beta_2=0.97),
    Adamax(), Adamax(0.1), Adamax(0.1, beta_1=0.8, beta_2=0.97),
    Ftrl(), Ftrl(0.1), 
    Ftrl(0.1, -0.5, 0.1, 0.01, 0.05, 'Ftrl', 0),
    Ftrl(0.1, -0.5, 0.1, 0.01, 0.05, 'Ftrl', 0, 0.05),
    Ftrl(0.1, -0.5, 0.1, 0.00, 0.05, 'Ftrl', 0),
    Ftrl(0.1, -0.5, 0.1, 0.00, 0.05, 'Ftrl', 0, 0.1),
    Ftrl(0.1, -0.5, 0.1, 0.01, 0.01, 'Ftrl', 0.05),
    Ftrl(0.1, -0.5, 0.1, 0.05, 0.00, 'Ftrl', 0),
    Ftrl(0.1, -0.5, 10, 0.00, 0.05, 'Ftrl', 0),
    Ftrl(0.1, -0.5, 10, 0.00, 0.05, 'Ftrl', 0, 0.5),
    Ftrl(0.1, -0.5, 10, 0.01, 0.01, 'Ftrl', 0.05),
    Ftrl(0.1, -0.5, 10, 0.05, 0.01, 'Ftrl', 0.05),
    RMSprop(), RMSprop(0.1, rho=0.8), RMSprop(0.1, momentum=0.5), RMSprop(rho=0.7, momentum=0.7),
    SGD(), SGD(0.1), SGD(momentum=0.5)
]


all_results = []
for gradients in [gradients1, gradients1d, gradients10, gradients10d, gradients100, gradients100d]:
    results = []
    for optimizer in optimizers:
        A = run_tf_optimizer(optimizer, gradients)
        B = run_my_optimizer(optimizer, gradients)
        row = A.shape[0] - 1
        col = A.shape[1] - 1
        error = tf.reduce_sum(tf.reduce_sum(tf.abs(A - B)))
        results.append((float(error), optimizer.get_config()['name'],
              float(A[0][0]), float(B[0][0]), float(A[row][col]), float(B[row][col])))
    all_results.append(sorted(results, key=lambda x: x[0]))

for results in all_results:
    for result in results:
        if result[0] > 10.0:
            print("error! ", result, file=sys.stderr)
            exit(1)
        print(result)
    print()
