#!/bin/bash
set -e
wget https://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz -O tmp/dac_sample.tar.gz
tar -xzf tmp/dac_sample.tar.gz -C tmp
python3 examples/criteo_preprocess.py tmp/dac_sample.txt tmp/dac_sample.csv
python3 criteo_lr_subclass.py --data tmp/dac_sample.csv --batch_size 256
