#!/bin/bash
set -e
horovodrun --gloo -np 2 python3 examples/criteo_deepctr_network.py