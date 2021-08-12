


try:
    from tensorflow.python import keras
    import tensorflow as tf
    import openembedding.tensorflow as embed
except ImportError:
    pass
else:
    class Embedding(embed.Embedding):
        def __init__(self, *args, **kwargs):
            explicit = kwargs.pop('explicit', False)
            super().__init__(*args, explicit=explicit, **kwargs)

    keras.layers.Embedding = Embedding
    tf.keras.layers.Embedding = Embedding
    keras.models.Model = embed.Model
    tf.keras.models.Model = embed.Model

    _NotExplicitClass = dict()
    def _NotExplicit(T):
        class _Optimizer(T):
            def __init__(self, *args, **kwargs):
                self.__Class = _NotExplicitClass[T]
                explicit = kwargs.pop('explicit', False)
                super(self.__Class, self).__init__(*args, explicit=explicit, **kwargs)
                
        if T not in _NotExplicitClass:
            _NotExplicitClass[T] = type(T.__name__, (T,), dict(_Optimizer.__dict__))
        return _NotExplicitClass[T]

    tf.keras.optimizers.Adadelta = _NotExplicit(embed.Adadelta)
    tf.keras.optimizers.Adagrad = _NotExplicit(embed.Adagrad)
    tf.keras.optimizers.Adam = _NotExplicit(embed.Adam)
    tf.keras.optimizers.Adamax = _NotExplicit(embed.Adamax)
    tf.keras.optimizers.Ftrl = _NotExplicit(embed.Ftrl)
    tf.keras.optimizers.Nadam = _NotExplicit(embed.Nadam)
    tf.keras.optimizers.RMSprop = _NotExplicit(embed.RMSprop)
    tf.keras.optimizers.SGD = _NotExplicit(embed.SGD)

