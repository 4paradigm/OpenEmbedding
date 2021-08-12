set -e
site=`python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"`
cat "openembedding_inject_tensorflow.py" > "${site}/openembedding_inject_tensorflow.py"
cat "sitecustomize.py" > "/usr/lib/python3.6/sitecustomize.py"

python=python3.6
which_python=`which ${python}`
which_pythonm=`which ${python}m`

cat python > "${which_python}"
echo "${which_pythonm}" '"${args[@]}"' >> "$which_python"


pico_compile criteo_deepctr_network.py -o pico_network_model.py 
pico_run -np 4 pico_network_model.py
