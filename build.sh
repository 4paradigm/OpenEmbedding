#!/bin/bash
set -e
CURFILE=`readlink -m $0`
CURDIR=`dirname ${CURFILE}`
PROJECT_ROOT=`pwd`
echo ${PROJECT_ROOT}
if [ "${PROJECT_ROOT}" != "${CURDIR}" ]; then
    echo "PROJECT_ROOT != CURDIR" 1>&2
    exit 1
fi

if [ "${THIRD_PARTY}" == "" ]; then
    THIRD_PARTY=${PROJECT_ROOT}/tools
fi

if [ "$J" == "" ]; then
    export J=`nproc | awk '{print int(($0 + 1)/ 2)}'` # make cocurrent thread number
fi

function build() {
    DEFINES=" -DTHIRD_PARTY=${THIRD_PARTY} -DCMAKE_INSTALL_PREFIX=${THIRD_PARTY}"

    if [ "${USE_RDMA}" == "0" ]; then
        DEFINES="${DEFINES} -DUSE_RDMA=OFF"
    elif [ "${USE_RDMA}" == "1" ]; then
        DEFINES="${DEFINES} -DUSE_RDMA=ON"
    fi

    mkdir -p ${PROJECT_ROOT}/pico-ps/pico-core/build
    pushd ${PROJECT_ROOT}/pico-ps/pico-core/build
    cmake ../ ${DEFINES} -DSKIP_BUILD_TEST=ON
    make -j$J
    make install
    popd

    mkdir -p ${PROJECT_ROOT}/pico-ps/build
    pushd ${PROJECT_ROOT}/pico-ps/build
    cmake ../ ${DEFINES} -DSKIP_BUILD_TEST=ON
    make -j$J
    make install
    popd

    DEFINES="${DEFINES} -DOPENEMBEDDING_VERSION=${VERSION}"
    if [ "${SKIP_CHECK_WHEEL_SETUP}" == "0" ]; then
        DEFINES="${DEFINES} -DSKIP_CHECK_WHEEL_SETUP=OFF"
    elif [ "${SKIP_CHECK_WHEEL_SETUP}" == "1" ]; then
        DEFINES="${DEFINES} -DSKIP_CHECK_WHEEL_SETUP=ON"
    fi
    mkdir -p ${PROJECT_ROOT}/build
    pushd ${PROJECT_ROOT}/build
    cmake ../ ${DEFINES} -DPYTHON=${PYTHON}
    make -j$J
    popd
}

function clean() {
    rm -r -f ${PROJECT_ROOT}/build
    rm -r -f ${PROJECT_ROOT}/pico-ps/build
    rm -r -f ${PROJECT_ROOT}/pico-ps/pico-core/build
}

function publish_check() {
    git diff
    local git_diff=`git diff` 
    if [ "$git_diff" == "" ]; then
        echo "git nodiff"
    else
        echo -e "please commit your change before publish"
        return 1
    fi
}

function exec_test() {
    echo RUN TEST $@
    if timeout --foreground 600 $@ 2>tmp/last_test.err; then
        echo SUCESS! $@
    else
        cat tmp/last_test.err
        echo FAILED! $@
        exit 1
    fi
}

function unit_test() {
    # rm -rf ${PROJECT_ROOT}/.ut 
    # 测试优化器正确性
    mkdir -p tmp
    exec_test python3 test/optimizer_test.py

    # 获取 dac_sample 测试数据
    if [ ! -f tmp/dac_sample.tar.gz ]; then
        wget -O tmp/dac_sample.tar.gz https://labs.criteo.com/wp-content/uploads/2015/04/dac_sample.tar.gz
    fi
    tar -xzf tmp/dac_sample.tar.gz -C tmp
    exec_test python3 examples/criteo_preprocess.py tmp/dac_sample.txt tmp/dac_sample.csv
    exec_test python3 examples/criteo_deepctr_network.py --data tmp/dac_sample.csv --batch_size 4096
    exec_test horovodrun -np 2 python3 examples/criteo_deepctr_network.py --data tmp/dac_sample.csv --batch_size 4096

    nproc=`nproc`
    if [ "$nproc" -ge "8" ]; then
        nproc=8
    fi
    echo $nproc
    
    # 测试各种训练模式
    for np in 1 $nproc; do
        exec_test horovodrun -np $np python3 test/benchmark/criteo_deepctr.py \
            --data tmp/dac_sample.csv --server --epochs 3
        exec_test horovodrun -np $np python3 test/benchmark/criteo_deepctr.py \
            --data tmp/dac_sample.csv --server --cache --epochs 3
        exec_test horovodrun -np $np python3 test/benchmark/criteo_deepctr.py \
            --data tmp/dac_sample.csv --server --cache --prefetch --epochs 3
    done

    # 测试只有一个 batch
    exec_test python3 test/benchmark/criteo_deepctr.py \
        --data examples/train100.csv --batch_size 100 --server --cache --prefetch
    exec_test horovodrun -np 2 python3 test/benchmark/criteo_deepctr.py \
        --data examples/train100.csv --batch_size 50 --server --cache --prefetch
    exec_test horovodrun -np $nproc python3 test/benchmark/criteo_deepctr.py \
        --data examples/train100.csv --batch_size 10 --server --cache --prefetch

    # 测试 example
    exec_test python3 examples/criteo_lr_subclass.py
    exec_test python3 examples/criteo_deepctr_hook.py
    exec_test horovodrun -np 2 python3 examples/criteo_deepctr_hook.py --checkpoint tmp/epoch
    exec_test horovodrun -np $nproc python3 examples/criteo_deepctr_hook.py --load tmp/epoch4/variables/variables

    exec_test python3 examples/criteo_deepctr_network_mirrored.py
    exec_test mpirun -np 2 --allow-run-as-root \
        python3 examples/criteo_deepctr_network_mirrored.py --checkpoint tmp/epoch
    exec_test mpirun -np $nproc --allow-run-as-root \
        python3 examples/criteo_deepctr_network_mirrored.py --load tmp/epoch4/variables/variables

    # 测试 all reduce 队列
    exec_test python3 examples/criteo_deepctr_network.py --data examples/wide100.csv
    exec_test horovodrun -np $nproc python3 examples/criteo_deepctr_network.py --data examples/wide100.csv

    # 测试 example 分布式 train -> 单机 serving 全流程
    exec_test python3 examples/criteo_deepctr_network.py
    exec_test horovodrun -np 2 python3 examples/criteo_deepctr_network.py --checkpoint tmp/epoch
    exec_test horovodrun -np $nproc python3 examples/criteo_deepctr_network.py \
        --load tmp/epoch4/variables/variables --export tmp/serving

    docker run --name serving-test -td -p 8500:8500 -p 8501:8501 \
        -v `pwd`/tmp/serving:/models/criteo/1 -e MODEL_NAME=criteo tensorflow/serving:latest
    sleep 20 # 需要等待 serving 启动
    exec_test python3 examples/tensorflow_serving_client.py
    exec_test python3 examples/tensorflow_serving_restful.py
    docker stop serving-test
    docker rm serving-test
}

case "$1" in
    test)
        unit_test
    ;;
    clean)
        clean
    ;;
    build|"")
        build
    ;;
    *)
        echo "unkown cmd"
        exit 1
    ;;
esac
