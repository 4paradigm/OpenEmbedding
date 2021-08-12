FROM tensorflow/tensorflow:2.2.0-gpu
RUN apt-get update && apt-get install -y gcc-7 g++-7 cmake
RUN pip install horovod
ADD . /openembedding
WORKDIR /openembedding/laboratory/inject

RUN bash inject.sh
WORKDIR /root
