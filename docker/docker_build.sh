#!/bin/bash
set -e
if [ "${VERSION}" == "" ]; then
    VERSION=0.0.0
fi

function build() {    
    IMAGE=4pdosc/openembedding:${VERSION}-build
    docker build -t ${IMAGE} -f docker/Dockerfile.build --build-arg VERSION=${VERSION} .
    docker run --name dockerbuild -itd ${IMAGE} /bin/bash
    rm -rf output
    mkdir -p output/dist
    docker cp dockerbuild:/openembedding/build/pypi/dist/openembedding-${VERSION}.tar.gz output/dist
    docker stop dockerbuild
    docker rm dockerbuild
    docker rmi ${IMAGE}
}

function image() {
    IMAGE=4pdosc/openembedding:${VERSION}
    docker build -t ${IMAGE} -f docker/Dockerfile .
}

function image_test() {
    mkdir -p tmp
    IMAGE=4pdosc/openembedding:${VERSION}
    echo '{'                            > tmp/daemon.json
    echo '    "storage-driver": "vfs"' >> tmp/daemon.json
    echo '}'                           >> tmp/daemon.json

    echo 'set -e' > tmp/test.sh
    echo 'curl -fsSL https://get.docker.com | sh' >> tmp/test.sh
    echo 'mkdir -p /etc/docker' >> tmp/test.sh
    echo 'cp tmp/daemon.json /etc/docker' >> tmp/test.sh
    echo 'service docker start' >> tmp/test.sh
    echo 'pip3 install tensorflow-serving-api' >> tmp/test.sh
    echo './build.sh test' >> tmp/test.sh

    docker run --privileged --name image_test -v `pwd`/tmp:/openembedding/tmp ${IMAGE} bash tmp/test.sh
    docker rm image_test
}

case "$1" in
    build|"")
        build
    ;;
    image)
        image
    ;;
    test)
        image_test
    ;;
    *)
        echo "unkown cmd"
        exit 1
    ;;
esac
