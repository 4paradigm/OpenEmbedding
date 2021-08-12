wget https://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz
tar -xzf dac_sample.tar.gz
python3 criteo_preprocess.py dac_sample.txt dac_sample.csv
