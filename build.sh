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
    mkdir -p tmp
    exec_test python3 test/optimizer_test.py

    # test examples
    exec_test examples/run/criteo_deepctr_standalone.sh
    exec_test examples/run/criteo_deepctr_horovod.sh
    exec_test examples/run/criteo_deepctr_checkpoint.sh
    exec_test examples/run/criteo_deepctr_serving.sh
    exec_test examples/run/criteo_deepctr_mirrored.sh
    # exec_test examples/run/criteo_deepctr_mpi.sh --allow-run-as-root
    exec_test examples/run/criteo_preprocess.sh

    nproc=`nproc`
    if [ "$nproc" -ge "8" ]; then
        nproc=8
    fi
    echo $nproc
    
    # test train modes
    for np in 1 $nproc; do
        exec_test horovodrun -np $np python3 test/benchmark/criteo_deepctr.py \
            --data tmp/dac_sample.csv --server --epochs 3
        exec_test horovodrun -np $np python3 test/benchmark/criteo_deepctr.py \
            --data tmp/dac_sample.csv --server --cache --epochs 3
        exec_test horovodrun -np $np python3 test/benchmark/criteo_deepctr.py \
            --data tmp/dac_sample.csv --server --cache --prefetch --epochs 3
    done

    # test only one batch
    exec_test python3 test/benchmark/criteo_deepctr.py \
        --data examples/train100.csv --batch_size 100 --server --cache --prefetch
    exec_test horovodrun -np 2 python3 test/benchmark/criteo_deepctr.py \
        --data examples/train100.csv --batch_size 50 --server --cache --prefetch
    exec_test horovodrun -np $nproc python3 test/benchmark/criteo_deepctr.py \
        --data examples/train100.csv --batch_size 10 --server --cache --prefetch

    # test load dump
    exec_test horovodrun -np 2 python3 examples/criteo_deepctr_hook.py --checkpoint tmp/epoch
    exec_test horovodrun -np $nproc python3 examples/criteo_deepctr_hook.py \
        --load tmp/epoch4/variables/variables
    
    exec_test mpirun -np 2 --allow-run-as-root \
        python3 examples/criteo_deepctr_network_mpi.py --checkpoint tmp/epoch
    exec_test mpirun -np $nproc --allow-run-as-root \
        python3 examples/criteo_deepctr_network_mpi.py \
        --load tmp/epoch4/variables/variables --checkpoint tmp/epoch
    exec_test python3 examples/criteo_deepctr_network_mirrored.py \
        --load tmp/epoch4/variables/variables

    # test with many all reduce
    exec_test python3 examples/criteo_deepctr_network.py --data examples/wide100.csv
    exec_test horovodrun -np $nproc python3 examples/criteo_deepctr_network.py --data examples/wide100.csv
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
