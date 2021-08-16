# OpenEmbedding

[![build status](https://github.com/4paradigm/openembedding/actions/workflows/build.yml/badge.svg)](https://github.com/4paradigm/openembedding/actions/workflows/build.yml)
[![docker pulls](https://img.shields.io/docker/pulls/4pdosc/openembedding.svg)](https://hub.docker.com/r/4pdosc/openembedding)
[![python version](https://img.shields.io/pypi/pyversions/openembedding.svg?style=plastic)](https://badge.fury.io/py/openembedding)
[![pypi package version](https://badge.fury.io/py/openembedding.svg)](https://badge.fury.io/py/openembedding)
[![downloads](https://pepy.tech/badge/openembedding)](https://pepy.tech/project/openembedding)

English version | [中文版](README_cn.md)

## Table of Contents

- [OpenEmbedding](#openembedding)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Benchmark](#benchmark)
  - [Features](#features)
  - [TODO](#todo)
  - [Install](#install)
  - [Docker](#docker)
  - [Quick Start](#quick-start)
  - [User Guide](#user-guide)
  - [Build](#build)
    - [Docker Build](#docker-build)
    - [Native Build](#native-build)
  - [Designs](#designs)

## About

OpenEmbedding is a distributed framework to accelerate TensorFlow training and support TensorFlow Serving. It uses the parameter server architecture to store the `Embedding` Layer. So that single machine memory is not the limit of model size. OpenEmbedding can cooperate with all-reduce framework to support both data parallel and model parallel. Compared with using all-reduce only, OpenEmbedding can achieve more than 500% acceleration in some conditions.

## Benchmark

![benchmark](documents/images/benchmark.png)

For models that contain sparse features, it is difficult to speed up using the all-reduce based framework Horovod, while using both OpenEmbedding and Horovod can get better acceleration effects. In the single 8 GPU scene, the speedup ratio is 3 to 8 times. Many models achieved 3 to 7 times the performance of Horovod.

- [Benchmark](documents/en/benchmark.md)

## Features

TensorFlow 2
- `dtype`: `float32`, `float64`
- `tensorflow.keras.initializers`
  - `RandomNormal`, `RandomUniform`, `Constant`, `Zeros`, `Ones`
  - The parameter `seed` is currently ignored
- `tensorflow.keras.optimizers`
  - `Adadelta`, `Adagrad`, `Adam`, `Adamax`, `Ftrl`, `RMSprop`, `SGD`
  - Not support `decay` and `LearningRateSchedule`
  - Not support `Adam(amsgrad=True)`
  - Not support `RMSProp(centered=True)`
  - The parameter server uses a sparse update method, which may cause different training results for the `Optimizer` with momentum
- `tensorflow.keras.layers.Embedding`
  - Support array for known `input_dim` and hash table for unknown `input_dim` (2**63 range)
  - Can still be stored on workers and use dense update method
  - Should not use `embeddings_regularizer`, `embeddings_constraint`
- tensorflow.keras.Model
  - Can be converted to distributed `Model` and automatically ignore or convert incompatible settings (such as `embeddings_constraint`)
  - Distributed `save`, `save_weights`, `load_weights` and `ModelCheckpoint`
  - Saving the distributed `Model` as a stand-alone SavedModel, which can be load by TensorFlow Serving
  - Not support training multiple distributed `Model`s in one task
- can collaborate with horovod, 'mirrored strategy' or 'multiworkermirroredstrategy'

## TODO

- Improve performance
- Support PyTorch training
- Support `tf.feature_column.embedding_column`
- Approximate `embedding_regularizer`, `LearningRateSchedule`, etc.
- Improve the support for `Initializer` and `Optimizer`
- Training multiple distributed `Model`s in one task 
- Support ONNX

## Install
The installation usually requires g++ 7 or higher, or a compiler compatible with `tf.version.COMPILER_VERSION`. The compiler can be specified by environment variable `CXX`. Currently OpenEmbedding can only be installed on linux.
```bash
pip3 install tensorflow horovod
pip3 install openembedding 
```

If TensorFlow was updated, you need to reinstall OpenEmbedding
```bash
pip3 uninstall openembedding && pip3 install --no-cache-dir openembedding
```

## Docker

Run the GPU image using NVIDIA docker. The image can be obtained through [Docker Hub](https://hub.docker.com/r/4pdosc/openembedding/tags).
```bash
docker run --gpus all -it 4pdosc/openembedding:latest /bin/bash
```

## Quick Start

The following examples can be run directly in the OpenEmbedding image.
```bash
# Stand-alone training
examples/runner/criteo_deepctr_standalone.sh

# Generate checkpoint and restore from it
examples/runner/criteo_deepctr_checkpoint.sh

# training on multi GPUs using  
examples/runner/criteo_deepctr_horovod.sh

# Use MirroredStrategy for single-machine multi-GPU training
examples/runner/criteo_deepctr_mirrored.sh

# Use MultiWorkerMirroredStrategy and MPI for multi-GPU training
examples/runner/criteo_deepctr_mpi.sh

# Download and preprocess the original criteo data format, and then train
examples/runner/criteo_preprocess.sh
```

The following example includes the entire process from distributed training to TensorFlow Serving.
```bash
examples/runner/criteo_deepctr_serving.sh
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
- [With TensorFlow MirroredStrategy](examples/criteo_deepctr_network_mirrored.py)
- [With TensorFlow MultiWorkerMirroredStrategy](examples/criteo_deepctr_network_mirrored.py)

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

## Designs

- [Training](documents/en/training.md)
- [Serving](documents/en/serving.md)
