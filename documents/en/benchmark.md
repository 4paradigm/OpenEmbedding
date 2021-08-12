# Benchmark

## Multi GPUs

Compare the acceleration effects of only Horovod and OpenEmbedding & Horovod on TensorFlow.

| Option | Setting |
| - | - |
| CPU | 2 * CPU Xeon(R) Gold 5218 CPU @ 2.30GHz |
| GPU | 8 * Tesla T4 |
| Data | Criteo |
| Data Format | TFRecord |
| Model | WDL, DeepFM, XDeepFM |
| Embedding Dimension | 9, 64 |
| Optimizer | Adagrad |
| Batch Size per GPU | 4096 |

![benchmark](../images/benchmark.png)

With the increase in the number of GPUs, it is difficult to speed up using the all-reduce based framework Horovod. For WDL 64, which accounts for a larger proportion of the sparse part, the performance of DeepFM 64 will decrease instead. For XDeepFM 9, Horovod can still get better acceleration due to the large amount of model calculations and the relatively small proportion of the sparse part. However, when the number of GPUs increases, the gap with OpenEmbedding & Horovod becomes larger and larger. Since XDeepFM 64 has a huge amount of calculation and takes too long, there is no test here.

## Remote Parameter Server

> In the previous section, OpenEmbedding & Horovod actually used the Cache Local setting in this section.

| Case | Setting |
| - | - |
| Local | Local server |
| Cache Local | Local server, high-frequency `Embedding` parameters updated by dense method and synchronized by Horovod ring all-reduce operator |
| Remote 100G | Remote server，connect with worker through 100G bit/s network |
| Cache Remote 100G | Remote server，connect with worker through 100G bit/s network，`Embedding` same as Cache Local |

![avatar](../images/benchmark-server.png)

As shown in the figure, in a 100G network, the communication between server and worker will not affect the performance significantly. In addition, the `Cache` test cases can usually get about 10% speedup.

## Big Data

OpenEmbedding has the ability to handle large-scale data. For the sparse features in large-scale data, it is sometimes difficult to de-duplicate and relabel. In OpenEmbedding, it can be hashed to the non-negative integer range of int64, and parameter servers will use hash table to store the parameters.

The performance test results of the 1TB Criteo data set are as follows.
| | |
| - | - |
| Model | DeepFM 9 |
| Optimizer | Adagrad |
| Setting | Remote |
| Data | Criteo1T |
| Data Format | TSV |
| Instance per Epoch | 3.3 G |
| Training speed | 692 kips |
| Time per Epoch | 4763 s |
| Checkpoint Time | 869 s |
| Server Memory | 1 * 175 GB |
| Worker Memory | 8 * 1.6 GB |
| Checkpoint Size | 78 GB |
| SavedModel Size | 45 GB |

# Run Steps

## Multi GPUs

1. Copy test/benchmark example/criteo_preprocess.py
2. Download and decompress Criteo dat and get `train.txt` about 11 GB
3. Preprocess `python3 criteo_preprocess.py train.txt train.csv`
4. Transform data to TFRecord format `mkdir -p tfrecord && python3 criteo_tfrecord.py train.csv 0 1`
5. Run the brenchmark case Horovod `horovodrun -np 2 python3 criteo_deepctr.py --data tfrecord`
6. Run the brenchmark case OpenEmbedding & Horovod `horovodrun -np 2 python3 criteo_deepctr.py --data`

## Remote Parameter Server

For the ip of the two machines are ip1 and ip2 respectively
1. Run servers `python3 server.py ip2:34567`
2. Run workers `python3 criteo_deepctr.py --data tfrecord --server --cache --master_endpoint ip2:34567 --bind_ip ip1`

## Big Data

1. Download and decompress Criteo 1TB data to `criteo1T` folder and the pattern of file path should be criteo1T/day_*
2. Run servers `python3 server.py ip1:34567`
3. Run workers `horovodrun -np 8 python3 criteo_deepctr.py --data criteo1T --server --master_endpoint ip1:34567`

You can use `--checkpoint`, `--save` and other parameters to specify the model save path. Note that all paths including `--data` should be shared. Distributed file systems can be mounted between different machines to share the path.
