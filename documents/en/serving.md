# Serving

## Stand-alone Model

![standalone](../images/standalone.drawio.png)

You can save the distributed model as a stand-alone SavedModel by `save_as_orignal_model`. SavedModel contains forward calculation graph and all parameters including `Embedding`, which can be loaded directly by TensorFlow Serving. This SavedModel cannot be used for training because it does not store `Optimizer` states.

## Distributed Model

![serving](../images/serving.drawio.png)

The distributed model needs to be loaded with TensorFlow Serving including OpenEmbedding Operator. The startup process is as follows:
1. Start the parameter server cluster, including ZooKeeper Master, Server, and Controller.
2. Load the EmbeddingModel to the parameter server cluster through the Controller.
3. Start TensorFlow Serving and load the SavedModel and connect to the ZooKeeper Master of the parameter server.

A UUID is stored in SavedModel to maintain the correspondence with EmbeddingModel. If the corresponding EmbeddingModel is not found on the parameter server, Tensorflow Serving will return "not found model" without causing other exceptions.