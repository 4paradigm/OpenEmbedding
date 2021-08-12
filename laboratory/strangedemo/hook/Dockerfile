FROM tensorflow/tensorflow:2.2.0-gpu
RUN apt-get update && apt-get install -y gcc-7 g++-7 cmake
RUN apt-get install -y vim wget
RUN pip3 install horovod pandas scikit-learn deepctr
RUN pip3 install jupyter jupyterlab

ADD openembedding-0.1.0.tar.gz /openembedding/openembedding-0.1.0.tar.gz
RUN pip3 install /openembedding/openembedding-0.1.0.tar.gz
ADD laboratory/strangedemo/hook /openembedding/hook
WORKDIR /openembedding/hook
RUN bash install.sh
WORKDIR /root
RUN rm -rf /openembedding