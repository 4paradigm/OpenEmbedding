set -e
site=`python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`
cp openembedding_hook_tensorflow.py ${site}/
cp mlcompile /usr/local/bin/
cp mlrun /usr/local/bin/
