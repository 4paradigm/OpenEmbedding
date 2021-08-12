import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.layers import Embedding as KerasEmbedding
import horovod.tensorflow.keras as hvd
import openembedding.tensorflow as embed


class Embedding(embed.Embedding):
    def __init__(self, *args, **kwargs):
        explicit = kwargs.pop('explicit', False)
        super().__init__(*args, explicit=explicit, **kwargs)


class Model(embed.Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compile(self, optimizer, *args, **kwargs):
        kwargs.pop('experimental_run_tf_function', None)
        optimizer = embed.distributed_optimizer(optimizer, explicit=False)
        optimizer = hvd.DistributedOptimizer(optimizer, op=hvd.Sum)
        return super().compile(optimizer, *args, experimental_run_tf_function=False, **kwargs)
    
    def save(self, *args, **kwargs):
        if hvd.rank() == 0:
            keras.layers.Embedding = KerasEmbedding
            tf.keras.layers.Embedding = KerasEmbedding
            keras.Model = KerasModel
            tf.keras.Model = KerasModel
            keras.models.Model = KerasModel
            tf.keras.models.Model = KerasModel
            super().save_as_original_model(*args, **kwargs)
            keras.layers.Embedding = Embedding
            tf.keras.layers.Embedding = Embedding
            keras.Model = Model
            tf.keras.Model = Model
            keras.models.Model = Model
            tf.keras.models.Model = Model


    def save_weights(self, *args, **kwargs):
        if hvd.rank() == 0:
            super().save_weights(*args, **kwargs)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1, callbacks=None, *args, **kwargs):
        if isinstance(x, dict):
            x1 = dict()
            n = len(y) // hvd.size() * hvd.size()
            for key, value in x.items():
                x1[key] = value[hvd.rank():n:hvd.size()]
            y1 = y[hvd.rank():n:hvd.size()]
        else:
            raise ValueError('only support dict input')
        if not callbacks:
            callbacks = []
        callbacks = callbacks + [
              hvd.callbacks.BroadcastGlobalVariablesCallback(0),
              hvd.callbacks.MetricAverageCallback() ]
        return super().fit(x1, y1, batch_size, epochs, verbose, callbacks=callbacks, *args, **kwargs)


keras.layers.Embedding = Embedding
tf.keras.layers.Embedding = Embedding
keras.Model = Model
tf.keras.Model = Model
keras.models.Model = Model
tf.keras.models.Model = Model

hvd.init()
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank() % len(gpus)], 'GPU')
