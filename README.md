# OpenEmbedding

[![build status](https://github.com/4paradigm/openembedding/actions/workflows/build.yml/badge.svg)](https://github.com/4paradigm/openembedding/actions/workflows/build.yml)
[![docker pulls](https://img.shields.io/docker/pulls/4pdosc/openembedding.svg)](https://hub.docker.com/r/4pdosc/openembedding)
[![python version](https://img.shields.io/pypi/pyversions/openembedding.svg?style=plastic)](https://badge.fury.io/py/openembedding)
[![pypi package version](https://badge.fury.io/py/openembedding.svg)](https://badge.fury.io/py/openembedding)
[![downloads](https://pepy.tech/badge/openembedding)](https://pepy.tech/project/openembedding)

English version | [中文版](README_cn.md)


## About

OpenEmbedding is a distributed framework to accelerate TensorFlow training and support TensorFlow Serving. It uses the parameter server architecture to store the `Embedding` Layer. So that single machine memory is not the limit of model size. OpenEmbedding can cooperate with all-reduce framework to support both data parallel and model parallel. Compared with using all-reduce only, OpenEmbedding can achieve more than 500% acceleration in some conditions.

## Benchmark

![benchmark](documents/images/benchmark.png)

For models that contain sparse features, it is difficult to speed up using the all-reduce based framework Horovod, while using both OpenEmbedding and Horovod can get better acceleration effects. In the single 8 GPU scene, the speedup ratio is 3 to 8 times. Many models achieved 3 to 7 times the performance of Horovod.

- [Benchmark](documents/en/benchmark.md)

## Install & Quick Start

You can install and run OpenEmbedding by the following steps. The examples show the whole process of training [criteo](https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/) data with OpenEmbedding and predicting with Tensorflow Serving.

### Docker

NVIDIA docker is required to use GPU in image. The OpenEmbedding image can be obtained from [Docker Hub](https://hub.docker.com/r/4pdosc/openembedding/tags).

```bash
# The script "criteo_deepctr_stanalone.sh" will train and export the model to the path "tmp/criteo/1".
# It is okay to switch to:
#    "criteo_deepctr_horovod.sh" (multi-GPU training with Horovod),
#    "criteo_deepctr_mirrored.sh" (multi-GPU training with MirroredStrategy),
#    "criteo_deepctr_mpi.sh" (multi-GPU training with MultiWorkerMirroredStrategy and MPI).
docker run --rm --gpus all -v /tmp/criteo:/openembedding/tmp/criteo \
    4pdosc/openembedding:latest examples/run/criteo_deepctr_standalone.sh 

# Start TensorFlow Serving to load the trained model.
docker run --name serving-example -td -p 8500:8500 -p 8501:8501 \
        -v /tmp/criteo:/models/criteo -e MODEL_NAME=criteo tensorflow/serving:latest
# Wait the model server start.
sleep 5

# Send requests and get predict results.
docker run --rm --network host 4pdosc/openembedding:latest examples/run/criteo_deepctr_restful.sh

# Clear docker.
docker stop serving-example
docker rm serving-example
```

### Ubuntu

```bash
# Install the dependencies required by OpenEmbedding.
apt update && apt install -y gcc-7 g++-7 python3 libpython3-dev python3-pip
pip3 install --upgrade pip
pip3 install tensorflow==2.5.1
pip3 install openembedding

# Install the dependencies required by examples.
apt install -y git cmake mpich curl 
HOROVOD_WITHOUT_MPI=1 pip3 install horovod
pip3 install deepctr pandas scikit-learn mpi4py

# Download the examples.
git clone https://github.com/4paradigm/OpenEmbedding.git
cd OpenEmbedding

# The script "criteo_deepctr_stanalone.sh" will train and export the model to the path "tmp/criteo/1".
# It is okay to switch to:
#    "criteo_deepctr_horovod.sh" (multi-GPU training with Horovod),
#    "criteo_deepctr_mirrored.sh" (multi-GPU training with MirroredStrategy),
#    "criteo_deepctr_mpi.sh" (multi-GPU training with MultiWorkerMirroredStrategy and MPI).
examples/run/criteo_deepctr_standalone.sh 

# Start TensorFlow Serving to load the trained model.
docker run --name serving-example -td -p 8500:8500 -p 8501:8501 \
        -v `pwd`/tmp/criteo:/models/criteo -e MODEL_NAME=criteo tensorflow/serving:latest
# Wait the model server start.
sleep 5

# Send requests and get predict results.
examples/run/criteo_deepctr_restful.sh

# Clear docker.
docker stop serving-example
docker rm serving-example
```

### CentOS

```bash
# Install the dependencies required by OpenEmbedding.
yum install -y centos-release-scl
yum install -y python3 python3-devel devtoolset-7
scl enable devtoolset-7 bash
pip3 install --upgrade pip
pip3 install tensorflow==2.5.1
pip3 install openembedding

# Install the dependencies required by examples.
yum install -y git cmake mpich curl 
HOROVOD_WITHOUT_MPI=1 pip3 install horovod
pip3 install deepctr pandas scikit-learn mpi4py

# Download the examples.
git clone https://github.com/4paradigm/OpenEmbedding.git
cd OpenEmbedding

# The script "criteo_deepctr_stanalone.sh" will train and export the model to the path "tmp/criteo/1".
# It is okay to switch to:
#    "criteo_deepctr_horovod.sh" (multi-GPU training with Horovod),
#    "criteo_deepctr_mirrored.sh" (multi-GPU training with MirroredStrategy),
#    "criteo_deepctr_mpi.sh" (multi-GPU training with MultiWorkerMirroredStrategy and MPI).
examples/run/criteo_deepctr_standalone.sh 

# Start TensorFlow Serving to load the trained model.
docker run --name serving-example -td -p 8500:8500 -p 8501:8501 \
        -v `pwd`/tmp/criteo:/models/criteo -e MODEL_NAME=criteo tensorflow/serving:latest
# Wait the model server start.
sleep 5

# Send requests and get predict results.
examples/run/criteo_deepctr_restful.sh

# Clear docker.
docker stop serving-example
docker rm serving-example
```

### Note

The installation usually requires g++ 7 or higher, or a compiler compatible with `tf.version.COMPILER_VERSION`. The compiler can be specified by environment variable `CC` and `CXX`. Currently OpenEmbedding can only be installed on linux.
```bash
CC=gcc CXX=g++ pip3 install openembedding
```

If TensorFlow was updated, you need to reinstall OpenEmbedding.
```bash
pip3 uninstall openembedding && pip3 install --no-cache-dir openembedding
```

## User Guide

A sample program for common usage is as follows.

Create `Model` and `Optimizer`.
```python
import tensorflow as tf
import deepctr.models import WDL
optimizer = tf.keras.optimizers.Adam()
model = WDL(feature_columns, feature_columns, task='binary')
```

Transform to distributed `Model` and distributed `Optimizer`. The `Embedding` layer will be stored on the parameter server.
```python
import horovod as hvd
import openembedding as embed
hvd.init()

optimizer = embed.distributed_optimizer(optimizer)
optimizer = hvd.DistributedOptimizer(optimizer)

model = embed.distributed_model(model)
```
Here, `embed.distributed_optimizer` is used to convert the TensorFlow optimizer into an optimizer that supports the parameter server, so that the parameters on the parameter server can be updated. The function `embed.distributed_model` is to replace the `Embedding` layers in the model and override the methods to support saving and loading with parameter servers. Method `Embedding.call` will pull the parameters from the parameter server and the backpropagation function was registerd to push the gradients to the parameter server.

Data parallelism by Horovod.
```python
model.compile(optimizer, "binary_crossentropy", metrics=['AUC'],
              experimental_run_tf_function=False)
callbacks = [ hvd.callbacks.BroadcastGlobalVariablesCallback(0),
              hvd.callbacks.MetricAverageCallback() ]
model.fit(dataset, epochs=10, verbose=2, callbacks=callbacks)
```

Export as a stand-alone SavedModel so that can be loaded by TensorFlow Serving.
```python
if hvd.rank() == 0:
    # Must specify include_optimizer=False explicitly
    model.save_as_original_model('model_path', include_optimizer=False)
```

More examples as follows.
- [Replace `Embedding` layer](examples/criteo_deepctr_hook.py)
- [Transform network model](examples/criteo_deepctr_network.py)
- [Custom subclass model](examples/criteo_lr_subclass.py)
- [With MirroredStrategy](examples/criteo_deepctr_network_mirrored.py)
- [With MultiWorkerMirroredStrategy and MPI](examples/criteo_deepctr_network_mirrored.py)

## Build

### Docker Build

```
docker build -t 4pdosc/openembedding-base:0.1.0 -f docker/Dockerfile.base .
docker build -t 4pdosc/openembedding:0.0.0-build -f docker/Dockerfile.build .
```

### Native Build

The compiler needs to be compatible with `tf.version.COMPILER_VERSION` (>= 7), and install all [prpc](https://github.com/4paradigm/prpc) dependencies to `tools` or `/usr/local`, and then run `build.sh` to complete the compilation. The `build.sh` will automatically install prpc (pico-core) and parameter-server (pico-ps) to the `tools` directory.

```bash
git submodule update --init --checkout --recursive
pip3 install tensorflow
./build.sh clean && ./build.sh build
pip3 install ./build/openembedding-*.tar.gz
```

## Features

TensorFlow 2
- `dtype`: `float32`, `float64`.
- `tensorflow.keras.initializers`
  - `RandomNormal`, `RandomUniform`, `Constant`, `Zeros`, `Ones`.
  - The parameter `seed` is currently ignored.
- `tensorflow.keras.optimizers`
  - `Adadelta`, `Adagrad`, `Adam`, `Adamax`, `Ftrl`, `RMSprop`, `SGD`.
  - Not support `decay` and `LearningRateSchedule`.
  - Not support `Adam(amsgrad=True)`.
  - Not support `RMSProp(centered=True)`.
  - The parameter server uses a sparse update method, which may cause different training results for the `Optimizer` with momentum.
- `tensorflow.keras.layers.Embedding`
  - Support array for known `input_dim` and hash table for unknown `input_dim` (2**63 range).
  - Can still be stored on workers and use dense update method.
  - Should not use `embeddings_regularizer`, `embeddings_constraint`.
- `tensorflow.keras.Model`
  - Can be converted to distributed `Model` and automatically ignore or convert incompatible settings. (such as `embeddings_constraint`)
  - Distributed `save`, `save_weights`, `load_weights` and `ModelCheckpoint`.
  - Saving the distributed `Model` as a stand-alone SavedModel, which can be load by TensorFlow Serving.
  - Not support training multiple distributed `Model`s in one task.
- Can collaborate with Horovod. Training with `MirroredStrategy` or `MultiWorkerMirroredStrategy` is experimental.

## TODO

- Improve performance
- Support PyTorch training
- Support `tf.feature_column.embedding_column`
- Approximate `embedding_regularizer`, `LearningRateSchedule`, etc.
- Improve the support for `Initializer` and `Optimizer`
- Training multiple distributed `Model`s in one task 
- Support ONNX

## Designs

- [Training](documents/en/training.md)
- [Serving](documents/en/serving.md)


## Authors

- Yiming Liu (liuyiming@4paradigm.com)
- Yilin Wang (wangyilin@4paradigm.com)
- Guangchuan Shi (shiguangchuan@4paradigm.com)
- Zhao Zheng (zhengzhao@4paradigm.com)
