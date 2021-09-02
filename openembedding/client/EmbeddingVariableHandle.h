#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_VARIABLE_HANDLE_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_VARIABLE_HANDLE_H

#include <pico-ps/handler/UDFHandler.h>
#include <pico-ps/handler/PushHandler.h>
#include <pico-ps/handler/LoadHandler.h>
#include <pico-ps/handler/DumpHandler.h>

#include "Meta.h"
#include "ObjectPool.h"

#include "EmbeddingPullOperator.h"
#include "EmbeddingPushOperator.h"
#include "EmbeddingLoadOperator.h"
#include "EmbeddingDumpOperator.h"
#include "EmbeddingStoreOperator.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

struct HandlerWaiter {
public:
    template<class F>
    HandlerWaiter(F&& waiter): _waiter(std::forward<F>(waiter)) {
        SCHECK(_waiter);
    }

    ~HandlerWaiter() {
        SCHECK(_wait_called);
    }
    
    HandlerWaiter(const HandlerWaiter&) = delete;
    HandlerWaiter& operator=(const HandlerWaiter&) = delete;

    HandlerWaiter(HandlerWaiter&& other) {
        _wait_called = other._wait_called;
        _waiter = other._waiter;
        other._wait_called = true;
    }

    ps::Status wait(void* result = nullptr) {
        _wait_called = true;
        return _waiter(result);
    }

private:
    bool _wait_called = false;
    std::function<ps::Status(void*)> _waiter;
};

// not handler, just a handle of storage handler.
class EmbeddingVariableHandle {
public:
    // n * embedding_dim;
    const EmbeddingVariableMeta& meta()const {
        return _meta;
    }

    uint32_t variable_id()const {
        return _variable_id;
    }

    void init_config(const core::Configure& config)const;

    // predictor controller
    ps::Status clear_weights();

    // predictor controller
    HandlerWaiter pull_weights(const uint64_t* indices, size_t n, int64_t batch_id)const;

    HandlerWaiter push_gradients(const uint64_t* indices, size_t n, const char* gradients)const;

    int _timeout = -1;
    bool _read_only = false;
    uint32_t _variable_id = 0;
    EmbeddingVariableMeta _meta;

    ObjectPool<std::unique_ptr<ps::UDFHandler>>* _read_only_pull_handler = nullptr;
    ObjectPool<std::unique_ptr<ps::UDFHandler>>* _pull_handler = nullptr;
    ObjectPool<std::unique_ptr<ps::UDFHandler>>* _push_handler = nullptr;
    ObjectPool<std::unique_ptr<ps::PushHandler>>* _init_handler = nullptr;
};

class EmbeddingStorageHandler {
public:
    EmbeddingStorageHandler() {}
    EmbeddingStorageHandler(const EmbeddingStorageHandler&) = delete;
    EmbeddingStorageHandler& operator=(const EmbeddingStorageHandler&) = delete;
    
    EmbeddingStorageHandler(EmbeddingStorageHandler&&) = default;
    EmbeddingStorageHandler& operator=(EmbeddingStorageHandler&&) = default;

    EmbeddingVariableHandle variable(uint32_t variable_id, EmbeddingVariableMeta meta);

    HandlerWaiter update_weights();

    // predictor controller
    ps::Status load_storage(const URIConfig& uri, size_t server_concurency = 4);

    // predictor controller
    ps::Status dump_storage(const URIConfig& uri, size_t file_number);

    int _timeout = -1;
    ObjectPool<std::unique_ptr<ps::UDFHandler>> _read_only_pull_handler;
    ObjectPool<std::unique_ptr<ps::UDFHandler>> _pull_handler;
    ObjectPool<std::unique_ptr<ps::UDFHandler>> _push_handler;
    ObjectPool<std::unique_ptr<ps::UDFHandler>> _store_handler;
    ObjectPool<std::unique_ptr<ps::PushHandler>> _init_handler;
    
    std::unique_ptr<ps::LoadHandler> _load_handler;
    std::unique_ptr<ps::DumpHandler> _dump_handler;
};


}
}
}

#endif
