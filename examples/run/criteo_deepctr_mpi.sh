#!/bin/bash
set -e
mpirun -np 2 python3 examples/criteo_deepctr_network_mpi.py