#!/bin/bash
set -e
python3 examples/tensorflow_serving_restful.py
python3 examples/tensorflow_serving_restful.py --rows 1
