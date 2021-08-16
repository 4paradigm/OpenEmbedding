#!/bin/bash
set -e
horovodrun -np 2 python3 examples/criteo_deepctr_network.py