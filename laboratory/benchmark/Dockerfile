RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
    python3.7 get-pip.py && \
    rm get-pip.py

RUN apt-get update && apt-get install -y python3.7-dev

RUN pip3.7 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir \
    future \
    grpcio \
    h5py \
    mock \
    numpy \
    requests \
    pandas \
    sklearn \
    deepctr \
    tensorflow==2.2

RUN apt-get update && apt-get install -y cmake build-essential devscripts debhelper fakeroot
RUN wget https://github.com/NVIDIA/nccl/archive/v2.8.3-1.tar.gz && tar -xzf v2.8.3-1.tar.gz && \
    cd nccl-2.8.3-1 && make -j src.build && make pkg.debian.build
RUN apt-get -y install ./nccl-2.8.3-1/build/pkg/deb/libnccl2_2.8.3-1+cuda10.1_amd64.deb ./nccl-2.8.3-1/build/pkg/deb/libnccl-dev_2.8.3-1+cuda10.1_amd64.deb
RUN HOROVOD_GPU_OPERATIONS=NCCL pip3.7 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir horovod

WORKDIR /root
RUN apt-get -y install libnuma-dev librdmacm-dev libibverbs-dev

RUN wget https://download.open-mpi.org/release/open-mpi/v4.0/openmpi-4.0.5.tar.gz && \
    tar -xzf openmpi-4.0.5.tar.gz && cd openmpi-4.0.5 && \
    ./configure --prefix=/usr/local/openmpi CFLAGS="-fPIC" CXXFlAGS="-fPIC" --enable-static && \
    make -j && make install

RUN apt-get update && apt-get install -y gawk vim libssl-dev tsocks privoxy ssh patchelf

RUN rm /usr/bin/python && rm /usr/bin/python3 && rm /usr/local/bin/pip && rm /usr/local/bin/pip3 && \
    ln -s /usr/bin/python3.7 /usr/bin/python && \
    ln -s /usr/bin/python3.7 /usr/bin/python3 && \
    ln -s /usr/local/bin/pip3.7 /usr/local/bin/pip && \
    ln -s /usr/local/bin/pip3.7 /usr/local/bin/pip3

RUN pip3.7 install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3.7 uninstall -y horovod && HOROVOD_GPU_OPERATIONS=NCCL pip3.7 install -i https://mirrors.aliyun.com/pypi/simple/ --no-cache-dir --upgrade horovod

ENV THRID_PARTY /usr/local
