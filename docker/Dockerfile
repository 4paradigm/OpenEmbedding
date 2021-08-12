FROM tensorflow/tensorflow:2.5.1-gpu
RUN apt-get update && apt-get install -y gcc-7 g++-7 cmake \
    openssh-client openmpi-bin libopenmpi-dev vim wget curl
RUN pip3 install mpi4py horovod
RUN pip3 install pandas scikit-learn deepctr
ADD . /openembedding
RUN pip3 install /openembedding/output/dist/openembedding-*.tar.gz
WORKDIR /openembedding
