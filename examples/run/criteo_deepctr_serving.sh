#!/bin/bash
set -e
horovodrun -np 2 python3 examples/criteo_deepctr_network.py --export tmp/serving
docker run --name serving-test -td -p 8500:8500 -p 8501:8501 \
        -v `pwd`/tmp/serving:/models/criteo/1 -e MODEL_NAME=criteo tensorflow/serving:latest
sleep 5 # wait model server start
python3 examples/tensorflow_serving_client.py
python3 examples/tensorflow_serving_restful.py
python3 examples/tensorflow_serving_restful.py --rows 1
docker stop serving-test
docker rm serving-test
