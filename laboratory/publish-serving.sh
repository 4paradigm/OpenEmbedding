echo $1
target=$1/tensorflow_serving/custom_ops
if [ "X$1" != "X" ]; then
    mkdir -p "$target"
    mkdir -p "$target/openembedding"
    mkdir -p "$target/openembedding/core"
    mkdir -p "$target/openembedding/tensorflow"
    cp "./build/openembedding/core/libcexb_pack.so" "$target/openembedding/core/libcexb_pack.so"
    cp "./openembedding/core/c_api.h" "$target/openembedding/core/c_api.h"
    cp "./openembedding/tensorflow/exb_ops.cpp" "$target/openembedding/tensorflow/exb_ops.cpp"
    cp "./openembedding/tensorflow/exb_ops.cpp" "$target/openembedding/tensorflow/exb_ops.h"
fi
