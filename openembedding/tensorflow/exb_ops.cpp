#include "tensorflow/stream_executor/stream.h"
#include "tensorflow/stream_executor/stream_executor.h"
#include "tensorflow/core/common_runtime/dma_helper.h"

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/register_types.h"

#include "Prefetch.h"

namespace paradigm4 {
namespace exb {

using namespace tensorflow;
using namespace tensorflow::shape_inference;
using namespace stream_executor;

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template<class T> struct OnGPU: std::false_type {};
template<> struct OnGPU<GPUDevice>: std::true_type {};

struct ExecutorHostMemory {
    StreamExecutor* executor = nullptr;
    void* base = nullptr;
    size_t size = 0;
};

class TemporaryHostMemoryPool {
public:
    ExecutorHostMemory allocate(StreamExecutor* executor, size_t size) {
        ExecutorHostMemory host;
        if (!executor) {
            return host;
        }
        host.executor = executor;
        {
            exb_lock_guard guard(_mutex);
            size_t num = _pool.size();
            std::vector<ExecutorHostMemory>& hosts = _pool[executor];
            if (!hosts.empty()) {
                host = hosts.back();
                hosts.pop_back();
            } 
            if (_pool.size() > num) {
                std::cerr << "new executor: " << _pool.size() << std::endl;
            }
        }
        if (host.size < size) {
            if (host.base) {
                executor->HostMemoryDeallocate(host.base);
            }
            host.base = executor->HostMemoryAllocate(host.size * 2 + size);
        }
        return host;
    }

    void deallocate(ExecutorHostMemory host) {
        if (host.executor && host.base) {
            exb_lock_guard guard(_mutex);
            _pool[host.executor].emplace_back(host);
        }
    };

private:
    exb_mutex _mutex;
    std::unordered_map<StreamExecutor*, std::vector<ExecutorHostMemory> > _pool;
};

VersionTable global_train_version;
VersionTable global_prefetch_version;
PrefetchTable global_prefetch_table;
TemporaryHostMemoryPool global_temporary_host_memory;

class TemporaryHostMemory {
public:
    TemporaryHostMemory() {}
    TemporaryHostMemory(StreamExecutor* executor, size_t size) {
        _host = global_temporary_host_memory.allocate(executor, size);
    }
    ~TemporaryHostMemory() {
        global_temporary_host_memory.deallocate(_host);
    }

    TemporaryHostMemory(const TemporaryHostMemory& other)=delete;
    TemporaryHostMemory& operator=(const TemporaryHostMemory& other)=delete;

    TemporaryHostMemory(TemporaryHostMemory&& other) {
        _host = other._host;
        other._host = ExecutorHostMemory();
    }

    TemporaryHostMemory& operator=(TemporaryHostMemory&& other) {
        global_temporary_host_memory.deallocate(_host);
        _host = other._host;
        other._host = ExecutorHostMemory();
        return *this;
    }

    void* base() {
        return _host.base;
    }
private:
    ExecutorHostMemory _host;
};

REGISTER_OP("PrefetchPullWeights")
    .Input("params: dtype") // this is for the fake var shape = [1] + embedding_shape, dtype is the same as the embeddings' type)
    .Input("indices: Tindices")
    .Output("output: Tindices")
    .Attr("variable_intptr: int")
    .Attr("steps: int")
    .Attr("dtype: type")
    .Attr("Tindices: {int64}")
    .SetShapeFn([](InferenceContext* context) {
        ShapeHandle unused;
        TF_RETURN_IF_ERROR(context->WithRankAtLeast(context->input(0), 1, &unused));
        context->set_output(0, context->input(1));
        return Status::OK();
    });

template <typename T, typename Index>
class PrefetchPullWeightsOp : public OpKernel {
public:
    explicit PrefetchPullWeightsOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("variable_intptr", &variable_intptr_));
        OP_REQUIRES_OK(context, context->GetAttr("steps", &steps_));
        OP_REQUIRES(context, variable_intptr_ != 0, errors::InvalidArgument("null variable_intptr"));
        OP_REQUIRES(context, steps_ >= -1, errors::InvalidArgument("error prefetch steps"));
        if (steps_ == -1) {
            steps_ = INT64_MAX;
        }
    }

    ~PrefetchPullWeightsOp() {}

    void Compute(OpKernelContext* context) override {
        static_assert(sizeof(Index) == sizeof(uint64_t), "");
        OP_REQUIRES(context, !running_ && steps_ >= 0,
            errors::InvalidArgument("prefetch pull not support running concurrency."));
        if (steps_ == 0) {
            return;
        }
        running_ = true;
        const Tensor& params = context->input(0);
        const Tensor& indices = context->input(1);
        context->set_output(0, indices);
        
        OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(params.shape()),
            errors::InvalidArgument("params must be at least 1 dimensional"));
            
        TensorShape result_shape;
        for (int i = 0; i < indices.dims(); ++i) {
            result_shape.AddDim(indices.dim_size(i));
        }
        for (int i = 1; i < params.dims(); ++i) {
            result_shape.AddDim(params.dim_size(i));
        }

        PrefetchKey key;
        key.variable = reinterpret_cast<exb_variable*>(variable_intptr_);
        key.indices = reinterpret_cast<const uint64_t*>(indices.flat<Index>().data());
        key.n = indices.NumElements();
        key.version = global_prefetch_version.pull_version(variable_intptr_);
        global_prefetch_version.update_version(variable_intptr_);
        steps_ -= 1;

        PrefetchValue value;
        value.check = key.hash();
        value.waiter = exb_pull_weights(key.variable, key.indices, key.n, key.version);
        global_prefetch_table.push(key, std::move(value));
        running_ = false;
    }

private:
    int64 variable_intptr_ = 0;
    int64 steps_ = 0;
    bool running_ = false;
};

#define REGISTER_PREFETCH_PULL_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("PrefetchPullWeights")                       \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("indices")                  \
                              .HostMemory("output")                   \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          PrefetchPullWeightsOp<type, index_type>)

#define REGISTER_PREFETCH_PULL_ALL_INDICES(dev, type) \
  REGISTER_PREFETCH_PULL_FULL(dev, type, int64)

#define REGISTER_PREFETCH_PULL_CPU(type) \
  REGISTER_PREFETCH_PULL_ALL_INDICES(CPU, type) \
  REGISTER_PREFETCH_PULL_ALL_INDICES(GPU, type)

// Registration of the CPU implementations.
TF_CALL_NUMBER_TYPES(REGISTER_PREFETCH_PULL_CPU);
TF_CALL_QUANTIZED_TYPES(REGISTER_PREFETCH_PULL_CPU);

#undef REGISTER_PREFETCH_PULL_CPU
#undef REGISTER_PREFETCH_PULL_ALL_INDICES
#undef REGISTER_PREFETCH_PULL_FULL


REGISTER_OP("PullWeights")
    .Input("params: dtype") // this is for the fake var shape = [1] + embedding_shape, dtype is the same as the embeddings' type)
    .Input("indices: Tindices")
    .Input("model_version: float64") //for serving
    .Output("output: dtype")
    .Attr("variable_intptr: int")
    .Attr("storage_intptr: int")
    .Attr("variable_id: int") //for serving
    .Attr("model_uuid: string") //for serving
    .Attr("dtype: type")
    .Attr("Tindices: {int64}")
    .SetShapeFn([](InferenceContext* context) {
        //check shape 
        ShapeHandle unused;
        TF_RETURN_IF_ERROR(context->WithRankAtLeast(context->input(0), 1, &unused));
        ShapeHandle embedding_shape;
        TF_RETURN_IF_ERROR(context->Subshape(context->input(0), 1, &embedding_shape));
        ShapeHandle out;
        TF_RETURN_IF_ERROR(context->Concatenate(context->input(1), embedding_shape, &out));
        context->set_output(0, out);
        return Status::OK();
    });

template <typename T, typename Index>    
class PullWeightsOp : public OpKernel {
public:
    explicit PullWeightsOp(OpKernelConstruction* context): OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("model_uuid", &model_uuid_));
        OP_REQUIRES_OK(context, context->GetAttr("variable_id", &variable_id_));
        OP_REQUIRES_OK(context, context->GetAttr("variable_intptr", &variable_intptr_));
        OP_REQUIRES_OK(context, context->GetAttr("storage_intptr", &storage_intptr_));
        OP_REQUIRES(context, variable_intptr_ != 0, errors::InvalidArgument("null variable_intptr"));
        OP_REQUIRES(context, storage_intptr_ != 0, errors::InvalidArgument("null storage_intptr_"));
    }

    ~PullWeightsOp() {
        if (exb_serving() && variable_) {
            exb_release_model_variable(variable_);
        }
    }

    void Compute(OpKernelContext* context) override {
        static_assert(sizeof(Index) == sizeof(uint64_t), "");
        
        const Tensor& params = context->input(0);
        const Tensor& indices = context->input(1);
        const Tensor& model_version = context->input(2);
        OP_REQUIRES(context, TensorShapeUtils::IsScalar(model_version.shape()),
              errors::InvalidArgument("model version must be scalar"));

        OP_REQUIRES(context, TensorShapeUtils::IsVectorOrHigher(params.shape()),
              errors::InvalidArgument("params must be at least 1 dimensional"));

        if (variable_.load(std::memory_order_acquire) == nullptr) {
            std::lock_guard<std::mutex> lock(_mutex);
            if (variable_ == nullptr) {
                if (exb_serving()) {
                    // serving
                    int64_t ver = std::floor(model_version.flat<double>()(0)); 
                    std::string model_sign = model_uuid_ + "-" + std::to_string(ver);
                    variable_ = exb_get_model_variable(exb_serving(), model_sign.c_str(), variable_id_);
                    OP_REQUIRES(context, variable_ != 0,
                          errors::InvalidArgument("not found model ", model_sign));
                } else {
                    // train
                    variable_ = reinterpret_cast<exb_variable*>(variable_intptr_);
                }
            }
        }

        /* The result shape is indices.shape + params.shape[1:].
         * should know each dimentions of params 
         * 
         * eg: we create 1000*5*5 embeddings. This means there are 1000 rows stored in parameter server, and the output of one embedding is a 5*5 matrix.  
         *     if indices is a 3*3 matrix, so the output tensor of this ops is 3*3*5*5.
         * */
        TensorShape result_shape;
        for (int i = 0; i < indices.dims(); ++i) {
            result_shape.AddDim(indices.dim_size(i));
        }
        for (int i = 1; i < params.dims(); ++i) {
            result_shape.AddDim(params.dim_size(i));
        }

        Tensor* out = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, result_shape, &out));

        PrefetchKey key;
        key.variable = variable_;
        key.indices = reinterpret_cast<const uint64_t*>(indices.flat<Index>().data());
        key.n = indices.NumElements();
        key.version = global_train_version.pull_version(storage_intptr_);

        PrefetchValue value;
        exb_pull_waiter* waiter = nullptr;
        if (global_prefetch_table.pop(key, value)) {
            OP_REQUIRES(context, value.check == key.hash(),
                  errors::InvalidArgument("prefetch not match, maybe prefetch multi times or concurrently"));
            waiter = value.waiter;
        } else {
            waiter = exb_pull_weights(key.variable, key.indices, key.n, key.version);
        }

        void* data = out->flat<T>().data();
        OP_REQUIRES(context, exb_pull_wait(waiter, key.indices, key.n, data),
              errors::InvalidArgument("pull failed: ", exb_last_error()));
    }

private:
    //tensorflow::string name_;
    std::atomic<exb_variable*> variable_ = {nullptr};
    int64 variable_intptr_ = 0;
    int64 storage_intptr_ = 0;
    int32 variable_id_ = 0;
    std::string model_uuid_;
    std::mutex _mutex;
};

#define REGISTER_PULL_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("PullWeights")                       \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("indices")               \
                              .HostMemory("model_version")           \
                              .HostMemory("output")                  \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          PullWeightsOp<type, index_type>)

#define REGISTER_PULL_ALL_INDICES(type) \
  REGISTER_PULL_FULL(CPU, type, int64) \
  REGISTER_PULL_FULL(GPU, type, int64) 

// Registration of the CPU implementations.
TF_CALL_NUMBER_TYPES(REGISTER_PULL_ALL_INDICES);
TF_CALL_QUANTIZED_TYPES(REGISTER_PULL_ALL_INDICES);

#undef REGISTER_PULL_CPU
#undef REGISTER_PULL_ALL_INDICES
#undef REGISTER_PULL_FULL


REGISTER_OP("PushGradients")
    .Input("params: dtype") // fake var
    .Input("indices: Tindices")
    .Input("grads: dtype")
    .Output("output: dtype") // fake out
    .Attr("variable_intptr: int")
    .Attr("dtype: type")
    .Attr("Tindices: {int64}")
    .SetShapeFn([](InferenceContext* context) {
        context->set_output(0, context->input(0));
        return Status::OK();
    });

template <typename T, typename Index>
class PushGradientsOp : public OpKernel {
public:
    explicit PushGradientsOp(OpKernelConstruction* context) : OpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("variable_intptr", &variable_intptr_));
        OP_REQUIRES(context, variable_intptr_ != 0, errors::InvalidArgument("null variable_intptr"));
    }

    void Compute(OpKernelContext* context) override {
        static_assert(sizeof(Index) == sizeof(uint64_t), "");
        
        const Tensor& params = context->input(0);
        const Tensor& indices = context->input(1);
        const Tensor& grads = context->input(2);

        const int64 N = indices.NumElements();
    
        Tensor* out = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, params.shape(), &out));

        /// TODO: check datatype
        const void* data = grads.flat<T>().data();
        exb_waiter* waiter = exb_push_gradients(reinterpret_cast<exb_variable*>(variable_intptr_),
            reinterpret_cast<const uint64_t*>(indices.flat<Index>().data()), N, data);
        OP_REQUIRES(context, exb_wait(waiter),
                errors::InvalidArgument("push failed: ", exb_last_error()));
    }

private:
    int64 variable_intptr_ = 0;
};


#define REGISTER_PUSH_FULL(dev, type, index_type)                    \
  REGISTER_KERNEL_BUILDER(Name("PushGradients")                       \
                              .Device(DEVICE_##dev)                    \
                              .HostMemory("grads")                    \
                              .HostMemory("indices")                  \
                              .TypeConstraint<type>("dtype")           \
                              .TypeConstraint<index_type>("Tindices"), \
                          PushGradientsOp<type, index_type>)

#define REGISTER_PUSH_ALL_INDICES(type) \
  REGISTER_PUSH_FULL(CPU, type, int64) \
  REGISTER_PUSH_FULL(GPU, type, int64) 

// Registration of the CPU implementations.
TF_CALL_NUMBER_TYPES(REGISTER_PUSH_ALL_INDICES);
TF_CALL_QUANTIZED_TYPES(REGISTER_PUSH_ALL_INDICES);

#undef REGISTER_PUSH_CPU
#undef REGISTER_PUSH_ALL_INDICES
#undef REGISTER_PUSH_FULL

REGISTER_OP("UpdateWeights")
    .Input("params: resource") // fake var
    .Input("grads: dtype")
    .Attr("storage_intptr: int")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* context) {
        return Status::OK();
    });


template <typename T>
class UpdateWeightsOp : public AsyncOpKernel {
public:
    explicit UpdateWeightsOp(OpKernelConstruction* context) : AsyncOpKernel(context) {
        OP_REQUIRES_OK(context, context->GetAttr("storage_intptr", &storage_intptr_));
        OP_REQUIRES(context, storage_intptr_ != 0, errors::InvalidArgument("null storage_intptr_"));
    }

    void ComputeAsync(OpKernelContext* context, DoneCallback done) override {
        // 依赖all reduce fake gradients同步
        global_train_version.update_version(storage_intptr_);
        exb_waiter* waiter = exb_update_weights(reinterpret_cast<exb_storage*>(storage_intptr_));
        ThreadPool::singleton().submit([context, waiter, done]() {
            OP_REQUIRES_ASYNC(context, exb_wait(waiter),
                  errors::InvalidArgument("update failed: ", exb_last_error()), done);
            done();
        });
    }

private:
    int64 storage_intptr_ = 0;
};

#define REGISTER_UPDATE_FULL(dev, type)                    \
  REGISTER_KERNEL_BUILDER(Name("UpdateWeights")                       \
                              .Device(DEVICE_##dev)                    \
                              .TypeConstraint<type>("dtype"),           \
                          UpdateWeightsOp<type>)

#define REGISTER_UPDATE(type) \
  REGISTER_UPDATE_FULL(CPU, type) \
  REGISTER_UPDATE_FULL(GPU, type)

// Registration of the CPU implementations.
TF_CALL_NUMBER_TYPES(REGISTER_UPDATE);
TF_CALL_QUANTIZED_TYPES(REGISTER_UPDATE);


#undef REGISTER_UPDATE_CPU
#undef REGISTER_UPDATE_ALL_INDICES
#undef REGISTER_UPDATE_FULL

REGISTER_OP("StreamSend")
    .Input("input: dtype")
    .Output("output: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* context) {
        context->set_output(0, context->input(0));
        return Status::OK();
    });

REGISTER_OP("StreamRecv")
    .Input("input: dtype")
    .Output("output: dtype")
    .Attr("dtype: type")
    .SetShapeFn([](InferenceContext* context) {
        context->set_output(0, context->input(0));
        return Status::OK();
    });

template <typename T>
class StreamSendOp : public OpKernel {
public:
    explicit StreamSendOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        Tensor* out = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &out));
        Stream* stream = context->op_device_context()->stream();
        size_t bytes = input.flat<T>().size() * sizeof(T);
        DeviceMemoryBase device_base(out->flat<T>().data());
        stream->ThenMemcpy(&device_base, input.flat<T>().data(), bytes);
        Tensor hold = *out;
        stream->ThenDoHostCallback([input, hold](){});
    }

private:
    int64 storage_intptr_ = 0;
};

template <typename T>
class StreamRecvOp : public OpKernel {
public:
    explicit StreamRecvOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext* context) override {
        const Tensor& input = context->input(0);
        Tensor* out = nullptr;
        OP_REQUIRES_OK(context, context->allocate_output(0, input.shape(), &out));
        Stream* stream = context->op_device_context()->stream();
        size_t bytes = input.flat<T>().size() * sizeof(T);
        DeviceMemoryBase device_base((void*)input.flat<T>().data());
        stream->ThenMemcpy(out->flat<T>().data(), device_base, bytes);
        Tensor hold = *out;
        stream->ThenDoHostCallback([input, hold](){});
    }

private:
    int64 storage_intptr_ = 0;
};

#define REGISTER_SEND_FULL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("StreamSend")                       \
                              .Device(DEVICE_GPU)                    \
                              .HostMemory("input")                  \
                              .TypeConstraint<type>("dtype"),           \
                          StreamSendOp<type>)

// Registration of the CPU implementations.
TF_CALL_NUMBER_TYPES(REGISTER_SEND_FULL);
TF_CALL_QUANTIZED_TYPES(REGISTER_SEND_FULL);

#define REGISTER_RECV_FULL(type)                    \
  REGISTER_KERNEL_BUILDER(Name("StreamRecv")                       \
                              .Device(DEVICE_GPU)                    \
                              .HostMemory("output")                  \
                              .TypeConstraint<type>("dtype"),           \
                          StreamRecvOp<type>)

// Registration of the CPU implementations.
TF_CALL_NUMBER_TYPES(REGISTER_RECV_FULL);
TF_CALL_QUANTIZED_TYPES(REGISTER_RECV_FULL);


}
}