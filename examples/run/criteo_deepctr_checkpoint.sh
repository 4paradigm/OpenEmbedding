#!/bin/bash
set -e
horovodrun -np 2 python3 examples/criteo_deepctr_network.py --checkpoint tmp/epoch
horovodrun -np 2 python3 examples/criteo_deepctr_network.py --load tmp/epoch4/variables/variables
