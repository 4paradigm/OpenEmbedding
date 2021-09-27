import os
import uuid
import shutil
import atexit
import tensorflow as tf
from openembedding import *
_HASH_KEY_RANGE = 2**63

def _get_ext_suffix():
    import sysconfig
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    if ext_suffix:
        return ext_suffix
    ext_suffix = sysconfig.get_config_var('SO')
    if ext_suffix:
        return ext_suffix
    return '.so'
exb_ops = tf.load_op_library(os.path.dirname(__file__) + '/exb_ops' + _get_ext_suffix())


def _str_dict(config):
    return {str(key): str(val) for key, val in config.items()}


def _tensorflow_initializer_config(initializer, explicit=True):
    def is_type(initializer, name):
        camel = ''
        for str in name.split('_'):
            camel += str.capitalize()
        if isinstance(initializer, eval('tf.' + name + '_initializer')):
            return True
        if isinstance(initializer, eval('tf.keras.initializers.' + camel)):
            return True
        if isinstance(initializer, eval('tf.compat.v1.keras.initializers.' + camel)):
            return True
        return False

    config = None
    if is_type(initializer, 'random_normal'):
        category = 'normal'
        config = initializer.get_config()
        config['truncated'] = 0
    if is_type(initializer, 'random_uniform'):
        category = 'uniform'
        config = initializer.get_config()
    if is_type(initializer, 'constant'):
        category = 'constant'
        config = initializer.get_config()
    if is_type(initializer, 'zeros'):
        category = 'constant'
        config = {'value': 0.0}
    if is_type(initializer, 'ones'):
        category = 'constant'
        config = {'value': 1.0}
    if config is None:
        if explicit:
            raise ValueError('error initializer: ' + str(initializer))
        category = 'constant'
        config = {'value': 0.0}
    config.pop('seed', None)
    config.pop('dtype', None)
    config['category'] = category
    return _str_dict(config)


def _tensorflow_optimizer_config(optimizer, explicit=True):
    if not isinstance(optimizer, tf.keras.optimizers.Optimizer):
        raise ValueError('error optimizer: ' + str(optimizer))
    category = optimizer.__class__.__name__.lower()
    config = optimizer.get_config()
    if not explicit:
        config.pop('amsgrad', None)
        config.pop('centered', None)
        config.pop('decay', None)

    if category == 'adam' and config.pop('amsgrad', False):
        category += 'amsgrad'
        raise ValueError('hyper-ebmedding not support adam with amsgrad')
    if category == 'rmsprop' and config.pop('centered', False):
        category += 'centered'
        raise ValueError('hyper-ebmedding not support centered rmsprop')
    if float(config.pop('decay', 0.0)) != 0.0:
        raise ValueError('hyper-ebmedding not support learning rate decay')
    config.pop('name')
    config['category'] = category
    return _str_dict(config)


@tf.RegisterGradient("PullWeights")
def _PullWeightsGrad(op, grad):
    graph_var = op.inputs[0]
    indices = op.inputs[1]
    variable_intptr = op.get_attr('variable_intptr')
    fake_grad = exb_ops.push_gradients(graph_var, indices, grad, variable_intptr=variable_intptr)
    with tf.device('CPU:0'):
        version_fake_grad = tf.constant(0.0, dtype=graph_var.dtype)
    return [fake_grad, None, version_fake_grad]


@tf.custom_gradient
def sparse_read_as_dense(var, indices):
    def dense_gradient(grad):
        return [tf.math.unsorted_segment_sum(grad, indices, tf.shape(var)[0]), None]
    return tf.gather(var, indices), dense_gradient


class Context:
    def __init__(self):
        if flags.num_workers < 1:
            raise ValueError('error num_workers')
        if flags.wait_num_servers < -1:
            raise ValueError('error wait_num_servers')
        self._context = libexb.Context(flags.num_workers, flags.wait_num_servers,
              str(uuid.uuid1()), flags.config, flags.master_endpoint, flags.bind_ip)
        self._tracks = dict()
        self._model_version = None
        self.__num_workers = flags.num_workers
        def _atexit():
            for variable in self._tracks.values():
                if isinstance(variable, Variable):
                    variable._finalize()
            self._context.finalize()
        atexit.register(_atexit)

    def track_variable(self, graph_var, variable):
        if isinstance(graph_var, tf.distribute.DistributedValues):
            for value in graph_var._values:
                self._tracks[value.ref()] = variable
        else:
            self._tracks[graph_var.ref()] = variable

    def find_variable(self, apply_var):
        if apply_var.ref() in self._tracks:
            return self._tracks[apply_var.ref()]
        return None

    @property
    def model_version(self):
        if isinstance(self._model_version, tf.distribute.DistributedValues):
            for value in self._model_version._values:
                return float(value)
        else:
            return float(self._model_version)

    @property
    def num_workers(self):
        return self.__num_workers


_context = None
def _multi_worker_strategy():
    if tf.distribute.has_strategy():
        strategy = tf.distribute.get_strategy()
        if hasattr(tf.distribute, 'MultiWorkerMirroredStrategy'):
            if isinstance(strategy, tf.distribute.MultiWorkerMirroredStrategy):
                return strategy
        if isinstance(strategy, tf.distribute.MirroredStrategy):
            return None
        raise ValueError('unsupport strategy ' + str(strategy))
    return None


def _get_context():
    global _context
    strategy = _multi_worker_strategy()
    horovod = os.environ.get('HOROVOD_SIZE', '').isdigit()
    if _context is None:
        master = None
        if horovod:
            import horovod.tensorflow as hvd
            hvd.init()
            flags.num_workers = hvd.size()
            if not flags.master_endpoint:
                if hvd.rank() == 0:
                    master = Master()
                    flags.master_endpoint = master.endpoint
                flags.master_endpoint = hvd.broadcast_object(flags.master_endpoint)
        elif strategy is not None:
            flags.num_workers = strategy.extended._num_workers
            if not flags.master_endpoint:
                if strategy.extended.should_checkpoint:
                    master = Master()
                    flags.master_endpoint = master.endpoint
                tensor = list(flags.master_endpoint.encode())
                if len(tensor) >= 1024:
                    raise ValueError('too long endpoint')
                tensor += [257] * (1024 - len(tensor))
                distributed_values = strategy.experimental_distribute_values_from_function(
                    lambda _: tf.identity(tf.constant([tensor])))
                for tensor in list(strategy.gather(distributed_values, axis=0)):
                    tensor = list(tensor)
                    length = tensor.index(257)
                    if length > 0:
                        flags.master_endpoint = bytes(tensor[0:length]).decode()
        else:
            master = None
            if not flags.master_endpoint:
                master = Master()
                flags.master_endpoint = master.endpoint
        _context = Context()
        _context._master = master
    else:
        # Need both check.
        if horovod:
            import horovod.tensorflow as hvd
            if _context.num_workers != hvd.size():
                raise ValueError('num workers not equal to the initialize state')
        if strategy is not None:
            if _context.num_workers < strategy.extended._num_workers:
                raise ValueError('using multi worker strategy but openembedding not initialize correctly.'
                    'you should initialize openembedding or create first variable in strategy scope.')

    # 0.1 for using floor.
    # Delay tensorflow initialize.
    if _context._model_version is None:
        with tf.device('CPU:0'):
            _context._model_version = tf.Variable(0.1, dtype=tf.float64) 
            _context.track_variable(_context._model_version, _context._model_version)
    return _context


class Variable:
    def __init__(self, initializer=None, trainable=None, name=None,
          dtype=None, shape=None, num_shards=None, sparse_as_dense=False,
          graph_var=None):
        if not num_shards:
            num_shards = -1
        # TODO: Initializer support direct Tensor and auto set shape.
        if shape is not None:
            shape = list(shape)
        if shape is not None and shape[0] == -1:
            shape[0] = _HASH_KEY_RANGE
            sparse_as_dense = False

        if graph_var is not None:
            if name or trainable is not None:
                raise ValueError('parameter conflict with graph_var')
            if shape and shape[1:] != graph_var.shape[1:]:
                raise ValueError('graph_var shape not match')
        
        if sparse_as_dense:
            if graph_var is None:
                graph_var = tf.Variable(initializer(dtype=dtype, shape=shape),
                      trainable=trainable, name=name)
            self.__sparse_as_dense = sparse_as_dense
            self.__shape = graph_var.shape
            self.graph_var = graph_var
            return

        if shape[0] <= 0 or shape[0] > _HASH_KEY_RANGE:
            raise ValueError("error shape")

        if not isinstance(initializer, dict):
            initializer = _tensorflow_initializer_config(initializer)

        if dtype is None:
            dtype = tf.float32
        if not isinstance(dtype, str):
            dtype = dtype.name
        
        if graph_var is None:
            with tf.device('CPU:0'):
                graph_var = tf.Variable(tf.constant(0, dtype=dtype, shape=[1] + shape[1:]),
                      trainable=trainable, name=name)

        embedding_dim = 1
        for dim in shape[1:]:
            embedding_dim *= dim

        self.__sparse_as_dense = sparse_as_dense
        self.__shape = shape
        self.__initialized = True

        # TensorFlow:AutoGraph not support mangled names.
        self.graph_var = graph_var
        self.storage = _get_context()._context.create_storage(num_shards)
        self.variable = self.storage.create_variable(shape[0], embedding_dim, dtype)
        self.variable.set_initializer(initializer)
        self.model_uuid = _get_context()._context.model_uuid
        self.model_version = _get_context()._model_version
        self.optimizer_set = False
        _get_context().track_variable(graph_var, self)
    
    @property
    def name(self):
        return self.graph_var.name

    @property
    def shape(self):
        return self.__shape

    @property
    def sparse_as_dense(self):
        return self.__sparse_as_dense

    def prefetch(self, indices, steps=None):
        if steps is None:
            steps = -1
        if self.sparse_as_dense:
            raise ValueError("should not prefetch for sparse as dense.")
        
        if indices.dtype.name != 'int64':
            with tf.device('CPU:0'):
                indices = tf.cast(indices, tf.int64)
        return exb_ops.prefetch_pull_weights(self.graph_var, indices,
              variable_intptr=self.variable.intptr, steps=steps)

    def sparse_read(self, indices):
        if self.sparse_as_dense:
            return sparse_read_as_dense(self.graph_var, indices)
        
        if indices.dtype.name != 'int64':
            with tf.device('CPU:0'):
                indices = tf.cast(indices, tf.int64)
        if isinstance(indices, tf.RaggedTensor):
            return tf.ragged.map_flat_values(
                  exb_ops.pull_weights, self.graph_var, indices, self.model_version,
                  variable_intptr=self.variable.intptr,
                  storage_intptr=self.storage.intptr,
                  variable_id=self.variable.variable_id,
                  model_uuid=self.model_uuid)
        else:
            return exb_ops.pull_weights(self.graph_var, indices, self.model_version,
                  variable_intptr=self.variable.intptr,
                  storage_intptr=self.storage.intptr,
                  variable_id=self.variable.variable_id,
                  model_uuid=self.model_uuid)
        
    def set_server_optimizer(self, optimizer):
        if self.sparse_as_dense:
            raise ValueError("no need optimizer for sparse as dense.")

        if not isinstance(optimizer, dict):
            optimizer = _tensorflow_optimizer_config(optimizer)
        self.variable.set_optimizer(optimizer)
    
    def pull_weights(self, indices):
        return self.sparse_read(indices)

    def push_gradients(self, indices, gradients):
        if self.sparse_as_dense:
            raise ValueError("no need update weights for sparse as dense.")
        
        if indices.dtype.name != 'int64':
            with tf.device('CPU:0'):
                indices = tf.cast(indices, tf.int64)
        return exb_ops.push_gradients(self.graph_var, indices, gradients,
              variable_intptr=self.variable.intptr)

    def update_weights(self, fake_grad):
        if self.sparse_as_dense:
            raise ValueError("no need update weights for sparse as dense.")
        
        return exb_ops.update_weights(self.graph_var.handle, fake_grad,
              storage_intptr=self.storage.intptr)
    
    def _finalize(self):
        if self.__initialized:
            self.storage.finalize()
            self.__initialized = False


'''
Embedding layer, can replace tf.keras.layers.Embedding.

input_dim: Size of the vocabulary.
    -1: The input range is [0, 2**63).

    n: The input range is [0, n).

output_dim: Dimension of the dense embedding.

sparse_as_dense: Determine the storage location when input_dim != -1.
    False: The Embedding layer is stored on parameter servers. Should not set embeddings_regularizer and embeddings_constraint.

    True: Similar to tf.distributed.MirroredStrategy.

num_shards: The Embedding layer will be divided into shards.
    -1: One shard each server, same as num_shards = num_servers.

    n: The sum number of all shards on all servers.

explicit:
    False: The embeddings_regularizer is traited as activity_regularizer, and ignore other unsupported arguments。

    True: Will raise exception when unsupported arguments was used
'''
class Embedding(tf.keras.layers.Embedding):
    def __init__(self, input_dim, output_dim, embeddings_initializer='uniform',
          embeddings_regularizer=None, activity_regularizer=None, embeddings_constraint=None, *args,
          num_shards=None, sparse_as_dense=False, explicit=True, **kwargs):
        if input_dim is None:
            input_dim = -1
        if input_dim == -1:
            input_dim = _HASH_KEY_RANGE
            sparse_as_dense = False
        if not sparse_as_dense and explicit:
            if embeddings_regularizer:
                raise ValueError('not support embeddings_regularizer')
            if embeddings_constraint:
                raise ValueError('not support embeddings_constraint')
        if not activity_regularizer:
            activity_regularizer = embeddings_regularizer
        embeddings_regularizer = None
        embeddings_constraint = None
        if input_dim <= 0:
            raise ValueError('error input_dim')

        self.num_shards = num_shards # self.num_shards is private.
        self.sparse_as_dense = sparse_as_dense
        super(Embedding, self).__init__(input_dim, output_dim,
               embeddings_initializer=embeddings_initializer,
               embeddings_regularizer=embeddings_regularizer,
               activity_regularizer=activity_regularizer,
               embeddings_constraint=embeddings_constraint, *args, **kwargs)
        if not sparse_as_dense:
            self.server_initializer = _tensorflow_initializer_config(embeddings_initializer)
            # setattr hooked by tensorflow, so self.server_initializer is not dict but DictWrapper.

    def build(self, input_shape):
        if self.sparse_as_dense:
            self.embeddings = self.add_weight(name='embeddings',
                  shape=(self.input_dim, self.output_dim),
                  initializer=self.embeddings_initializer,
                  regularizer=self.embeddings_regularizer,
                  constraint=self.embeddings_constraint)
            self.variable = Variable(sparse_as_dense=True, graph_var=self.embeddings)
        else:
            with tf.device('CPU:0'):
                self.embeddings = self.add_weight(name='embeddings',
                      shape=(1, self.output_dim),
                      initializer=tf.constant_initializer(0))
            self.variable = Variable(
                  initializer=dict(self.server_initializer), # pybind11 need exactly dict type.
                  dtype=self.dtype,
                  shape=(self.input_dim, self.output_dim),
                  num_shards=self.num_shards,
                  graph_var=self.embeddings)
        self.model_version = _get_context()._model_version
        self.built = True

    def call(self, inputs):
        return self.variable.sparse_read(inputs)


_DistributedOptimizerClass = dict()
def _DistributedOptimizer(T):
    class _Optimizer(T):
        def __init__(self, *args, explicit=True, **kwargs):
            self._Class = _DistributedOptimizerClass[T]
            super(self._Class, self).__init__(*args, **kwargs)
            self.server_optimizer = _tensorflow_optimizer_config(self, explicit=explicit)

        def _resource_apply_dense(self, grad, var, apply_state=None):
            variable = _get_context().find_variable(var)
            if variable is _get_context()._model_version:
                with tf.device('CPU:0'):
                    return var.assign_add(1, read_value=False)
            elif variable:
                if not variable.optimizer_set:
                    variable.optimizer_set = True
                    variable.set_server_optimizer(self.server_optimizer)
                return variable.update_weights(grad)
            else:
                return super(self._Class, self)._resource_apply_dense(grad, var, apply_state)
    if T not in _DistributedOptimizerClass:
        _DistributedOptimizerClass[T] = type(T.__name__, (T,), dict(_Optimizer.__dict__))
    return _DistributedOptimizerClass[T]


Adadelta = _DistributedOptimizer(tf.keras.optimizers.Adadelta)
Adagrad = _DistributedOptimizer(tf.keras.optimizers.Adagrad)
Adam = _DistributedOptimizer(tf.keras.optimizers.Adam)
Adamax = _DistributedOptimizer(tf.keras.optimizers.Adamax)
Ftrl = _DistributedOptimizer(tf.keras.optimizers.Ftrl)
Nadam = _DistributedOptimizer(tf.keras.optimizers.Nadam)
RMSprop = _DistributedOptimizer(tf.keras.optimizers.RMSprop)
SGD = _DistributedOptimizer(tf.keras.optimizers.SGD)


def distributed_optimizer(optimizer, explicit=True):
    '''
    Return a distributed optimizer to train on parameter servers and workers.
    If horovod is used, this call must be before hvd.DistributedOptimizer.
    '''
    config = optimizer.get_config()
    config['explicit'] = explicit
    return _DistributedOptimizer(optimizer.__class__).from_config(config)


def save_server_model(model, filepath, include_optimizer=True):
    '''
    Save the parameters on servers.
    '''
    _get_context()._context.save_model(filepath,
          _get_context().model_version, include_optimizer)


def load_server_model(model, filepath):
    '''
    Load the parameters on servers. This function must be called synchronously by all workers.
    '''
    _get_context()._context.load_model(filepath)


def save_as_original_model(model, filepath, overwrite=True, include_optimizer=True, *args, **kwargs):
    '''
    Save the distributed model as a stand-alone TensorFlow SavedModel.

    filepath: Path to the SavedModel.

    include_optimizer: Must set to False explicitly.
    '''
    layers = dict()
    failed = list()
    if include_optimizer == True:
        raise ValueError('not support include optimizer')
    if not (isinstance(model, tf.keras.Sequential) or model._is_graph_network):
        raise ValueError('not support subclass model')

    def _clone_function(layer):
        if isinstance(layer, Embedding):
            if not layer.built:
                raise ValueError("not a built model")
            keras_layer = tf.keras.layers.Embedding(layer.input_dim, layer.output_dim,
                  embeddings_initializer=layer.embeddings_initializer,
                  activity_regularizer=layer.activity_regularizer,
                  mask_zero=layer.mask_zero,
                  input_length=layer.input_length,
                  name=layer.name)
            layers[keras_layer] = layer
            return keras_layer
        return layer
    
    for layer in model.layers:
        if isinstance(layer, Embedding) and layer.variable.shape[0] >= _HASH_KEY_RANGE:
            raise ValueError("can not convert sparse variable to keras.Embedding.")
    model = tf.keras.models.clone_model(model, clone_function=_clone_function)
    for keras_layer in model.layers:
        if keras_layer in layers:
            batch = 2**20 // keras_layer.output_dim + 1
            for i in range(0, keras_layer.input_dim, batch):
                indices = tf.range(i, min(keras_layer.input_dim, i + batch))
                values = layers[keras_layer].variable.sparse_read(indices)
                slices = tf.IndexedSlices(values, indices, keras_layer.embeddings.shape)
                keras_layer.embeddings.scatter_update(slices)
    model.save(filepath, overwrite=overwrite, include_optimizer=include_optimizer, *args, **kwargs)


_DistributedModelClass = dict()
def _DistributedModel(T):
    class _Model(T):
        def __init__(self, *args, **kwargs):
            self._Class = _DistributedModelClass[T]
            super(self._Class, self).__init__(*args, **kwargs)
        
        def save(self, filepath, overwrite=True, include_optimizer=True, *args, **kwargs):
            super(self._Class, self).save(filepath, overwrite=overwrite, include_optimizer=include_optimizer, *args, **kwargs)
            if os.path.exists(filepath + '/openembedding'):
                shutil.rmtree(filepath + '/openembedding')
            save_server_model(self, filepath + '/openembedding', include_optimizer=include_optimizer)
        
        def save_weights(self, filepath, *args, **kwargs):
            super(self._Class, self).save_weights(filepath, *args, **kwargs)
            if os.path.exists(filepath + '.openembedding/openembedding'):
                shutil.rmtree(filepath + '.openembedding/openembedding')
            save_server_model(self, filepath + '.openembedding/openembedding', include_optimizer=True)

        def load_weights(self, filepath, *args, **kwargs):
            super(self._Class, self).load_weights(filepath, *args, **kwargs)
            if os.path.exists(filepath + '.openembedding/openembedding'):
                load_server_model(self, filepath + '.openembedding/openembedding')
            elif os.path.exists(os.path.split(filepath)[0] + '/../openembedding'):
                load_server_model(self, os.path.split(filepath)[0] + '/../openembedding')
            else:
                raise IOError("embed not exist: " + filepath)
        
        def save_as_original_model(self, filepath, *args, **kwargs):
            save_as_original_model(self, filepath, *args, **kwargs)

    if T not in _DistributedModelClass:
        _DistributedModelClass[T] = type(T.__name__, (T,), dict(_Model.__dict__))
    return _DistributedModelClass[T]


'''
Some methods have been override to be compatible with distributed save and other methods.
You can also refer to distributed_model.
'''
Model = _DistributedModel(tf.keras.Model)

# TODO: Unsupported layer auto downgrade and warning.
def distributed_model(model, sparse_as_dense_size=64, num_shards=None, override_method=True, explicit=False):
    '''
    Return a keras.Model that supports distributed training with parameter servers.
    
    If the model is Sequential or Network, all keras.layer.Embedding will be replaced.
    Do not continue to use the input model after calling this function.

    model: The model to be converted.

    sparse_as_dense_size: Will set Embedding(sparse_as_dense=True) when input_dim <= sparse_as_dense_size.

    num_shard: Same as Embedding.

    override_method: Override save, save_weights, load_weights, save_as_original_model.
        Note that save_as_original_model is not available for subclass model.
        Usually one worker calls save_weights to save checkpoint,
        and all workers call load_weights synchronously to recovery from checkpoint.
        Compatible with tf.keras.models.ModelCheckpoint.
    
    explicit: Same as Embedding.
    '''
    if model._is_compiled:
        raise ValueError('model should not be compiled')

    def _clone_function(layer):
        if isinstance(layer, tf.keras.layers.Embedding) and not isinstance(layer, Embedding):
            sparse_as_dense = False
            if layer.input_dim <= sparse_as_dense_size:
                sparse_as_dense = True
            return Embedding(layer.input_dim, layer.output_dim,
                  embeddings_initializer=layer.embeddings_initializer,
                  embeddings_regularizer=layer.embeddings_regularizer,
                  activity_regularizer=layer.activity_regularizer,
                  embeddings_constraint=layer.embeddings_constraint,
                  mask_zero=layer.mask_zero,
                  input_length=layer.input_length,
                  name=layer.name,
                  num_shards=num_shards,
                  sparse_as_dense = sparse_as_dense,
                  explicit=explicit)
        return layer

    if isinstance(model, tf.keras.Sequential) or model._is_graph_network:
        model = tf.keras.models.clone_model(model, clone_function=_clone_function)
    elif explicit:
        raise ValueError('not support subclass model')
    if override_method:
        model._Class = _DistributedModel(model.__class__)
        model.__class__ = _DistributedModel(model.__class__)
    return model


def pulling(dataset, model, steps=None):
    '''
    EXPERIMENTAL! Return a tf.data.Dataset that will send pull requests.

    For each Embedding layer of the model
    if it has one and only one corresponding column in the dataset, 
    a pull request will be sent when the dataset is read.
    This is for performance optimization.

    dataset: an instance of tf.data.Dataset.

    model: an instance of Network.
        
    '''
    if not isinstance(dataset, tf.data.Dataset):
        raise ValueError('dataset must be tf.data.Dataset')
    if not model._is_graph_network:
        raise ValueError('model must be graph network')
    if not model._is_compiled:
        raise ValueError('model must be compiled')

    prefetch_dict = dict()
    relevant_nodes = list()
    for v in model._nodes_by_depth.values():
        relevant_nodes += v
    for layer in model.layers:
        if isinstance(layer, Embedding) and not layer.sparse_as_dense:
            for node in layer._inbound_nodes:
                if relevant_nodes and node not in relevant_nodes:
                    # node is not part of the current network
                    continue

                for inbound_layer, node_index, tensor_index, _ in node.iterate_inbound():
                    if isinstance(inbound_layer, tf.keras.layers.InputLayer):
                        prefetch_dict.setdefault(layer.variable, list())
                        prefetch_dict[layer.variable].append(inbound_layer.name)

    def mapper(*args):
        results = list(args)
        results[0] = dict(args[0].items())
        # Prefetching multiple indices in one variable are not supported.
        for variable, names in prefetch_dict.items():
            if len(names) == 1: 
                print("prefetch " + names[0] + " for " + variable.name)
                results[0][names[0]] = variable.prefetch(results[0][names[0]], steps=steps)
        return results
    return dataset.map(mapper) # Prefetching order must be same as the batch order.




# pmem experimental
def should_persist_server_model(model):
    return _get_context()._context.should_persist_model()

def persist_server_model(model, filepath, persist_pending_window):
    _get_context()._context.persist_model(filepath,
          _get_context().model_version, persist_pending_window)

def restore_server_model(model, filepath):
    _get_context()._context.restore_model(filepath)
