# Training
As described in [here](documents/en/training.md), there are two types of nodes (worker and server) in OpenEmbedding training cluster. We consider to deploy servers and workers in different physical machines, and we use Centos 7.8 as an example.

## 1 Persistent Memory (PMem) Server Setup
### 1.1 Upgrade Kernel
The default kernel version of Centos 7.8 is 3.10.0. To satisfy the requirement of Intel PMDK, we have to upgrade the kernel version.
```bash
# install elrepo source
yum -y install https://www.elrepo.org/elrepo-release-7.el7.elrepo.noarch.rpm
# install the latest kernel-lt and kernel-lt-devel
yum -y --enablerepo=elrepo-kernel install kernel-lt kernel-lt-devel
# check kernel list
grub2-mkconfig -o /boot/grub2/grub.cfg
awk -F\' '$1=="menuentry " {print $2}' /etc/grub2.cfg
# change kernel start sequence
grub2-set-default 0
grub2-mkconfig -o /boot/grub2/grub.cfg
reboot
```

### 1.2 Setup PMem
```bash
yum -y install ipmctl ndctl tbb.x86_64 numactl
ndctl create-namespace -f -e namespace0.0 --mode=fsdax
mkfs.ext4 /dev/pmem0
mkdir /mnt/pmem0 
mount -o dax /dev/pmem0 /mnt/pmem0 
```

### 1.3 Install OpenEmbedding and the Dependencies
```bash
# Install the dependencies required by OpenEmbedding.
yum install -y python3 python3-devel
yum -y install centos-release-scl-rh
yum -y install devtoolset-8-gcc-c++
scl enable devtoolset-8 -- bash
python3 -m pip install --upgrade pip
pip3 install tensorflow==2.2.0 pybind11 psutil

# Install OpenEmbedding.
pip3 install pmem-openembedding-0.0.0.tar.gz
```

### 1.4 Start Server Process
```bash
# Download the OpenEmbedding.
git clone https://github.com/4paradigm/OpenEmbedding.git
cd OpenEmbedding

#start the main PMem-based parameter server
python3 ./test/benchmark/server.py --bind_ip server_ip:server_port --pmem /mnt/pmem0
```

## 2 GPU Worker Setup
### 2.1 Install GPU Driver, CUDA and CUDNN 
CUDA version 10.1
Driver version 418.39

### 2.2 Install OpenEmbedding and the Dependencies
```bash
# Install the dependencies required by OpenEmbedding.
yum install -y python3 python3-devel ndctl centos-release-scl-rh devtoolset-8-gcc-c++
scl enable devtoolset-8 -- bash
python3 -m pip install --upgrade pip
pip3 install tensorflow==2.2.0 pybind11 psutil

# Install the dependencies required by examples.
yum install -y git cmake mpich curl python36-cffi.x86_64 
pip3 install horovod  # Enable NCCL, Keras, Tensorflow support
#pip3 install deepctr pandas scikit-learn mpi4py
pip3 install deepctr pandas scikit-learn

# Install OpenEmbedding.
pip3 install pmem-openembedding-0.0.0.tar.gz
```

###2.3 Setup & Start Workers
```bash
# Download the OpenEmbedding.
git clone https://github.com/4paradigm/OpenEmbedding.git
cd OpenEmbedding

# Start Training Criteo 
horovodrun -np 1 python3 ./test/benchmark/criteo_deepctr.py --data criteo_kaggle_train.csv --server --batch_size 4096 --master_endpoint server_ip:server_port
```
