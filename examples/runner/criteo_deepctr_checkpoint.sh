#!/bin/bash
set -e
python3 examples/criteo_deepctr_network.py --checkpoint tmp/epoch
python3 examples/criteo_deepctr_network.py --load tmp/epoch4/variables/variables
