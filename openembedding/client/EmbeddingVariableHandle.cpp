#include "EmbeddingVariableHandle.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class Handler>
class HandlerPointer {
public:
    HandlerPointer(ObjectPool<std::unique_ptr<Handler>>* pool)
        : _pool(pool), _handler(_pool->acquire()) {}

    ~HandlerPointer() {
        SCHECK(_handler == nullptr);
    }

    Handler* operator->()const {
        return _handler.get();
    }

    explicit operator bool()const {
        return _handler.operator bool();
    }

    HandlerWaiter done_waiter() {
        if (_handler) {
            ObjectPool<std::unique_ptr<Handler>>* pool = _pool;
            Handler* handler = _handler.release();
            return [pool, handler](void*) {
                ps::Status status = handler->wait();
                pool->release(std::unique_ptr<Handler>(handler));
                if (!status.ok()) {
                    SLOG(WARNING) << status.ToString();
                }
                return status;
            };
        }    
        SLOG(WARNING) << "no handler";
        return [](void*) { return ps::Status::Error("no handler"); };
    }
private:
    ObjectPool<std::unique_ptr<Handler>>* _pool = nullptr;
    std::unique_ptr<Handler> _handler;
};

template<class Handler>
class HandlerWaiterDone {
public:
    HandlerWaiterDone(HandlerPointer<Handler>&& pointer): _pointer(std::move(pointer)) {}
    
    ps::Status operator()(void*) {
        ps::Status status = _pointer->wait();
        if (!status.ok()) {
            SLOG(WARNING) << status.ToString();
        }
        _pointer.release();
        return status;
    }
private:
    HandlerPointer<Handler> _pointer;
};

template<class Handler>
HandlerWaiterDone<Handler> handler_waiter_done(HandlerPointer<Handler>&& pointer) {
    return HandlerWaiterDone<Handler>(std::move(pointer));
}


HandlerWaiter EmbeddingVariableHandle::init_config(const core::Configure& config)const {
    SCHECK(!_read_only);
    std::unique_ptr<EmbeddingInitItems> items = std::make_unique<EmbeddingInitItems>();
    items->variable_id = _variable_id;
    items->meta = _meta;
    items->variable_config = config.dump();
    std::string meta_str;
    _meta.to_json_node().save(meta_str);
    SLOG(INFO) << "variable " << meta_str << " init config:\n" << config.dump();

    HandlerPointer<ps::PushHandler> handler(_init_handler);
    if (handler) {
        handler->async_push(std::move(items), _timeout);
    }
    return handler.done_waiter();
}

// predictor controller
HandlerWaiter EmbeddingVariableHandle::clear_weights() {
    if (_read_only) {
        SLOG(WARNING) << "the variable is read only";
        return [](void*){ return ps::Status::Error("the variable is read only"); };
    }
    std::unique_ptr<EmbeddingInitItems> items = std::make_unique<EmbeddingInitItems>();
    items->variable_id = _variable_id;
    items->meta = _meta;
    items->clear_weights = true;
    SLOG(INFO) << "variable " << _variable_id << " clear_weights";

    HandlerPointer<ps::PushHandler> handler(_init_handler);
    if (handler) {
        handler->async_push(std::move(items), _timeout);
    }
    return handler.done_waiter();
}

// predictor controller
HandlerWaiter EmbeddingVariableHandle::pull_weights(const uint64_t* indices, size_t n, int64_t batch_id)const {
    VTIMER(1, embedding_variable, pull_weights, ms);
    core::vector<EmbeddingPullItems> items(1);
    items[0].variable_id = _variable_id;
    items[0].meta = _meta;
    items[0].indices = indices;
    items[0].n = n;
    items[0].batch_id = batch_id;

    ObjectPool<std::unique_ptr<ps::UDFHandler>>* handler_pool = _pull_handler;
    if (_read_only) {
        handler_pool = _read_only_pull_handler;
    }
    ps::UDFHandler* pull_handler = handler_pool->acquire().release();
    if (!pull_handler) {
        SLOG(WARNING) << "no pull_handler";
        return [](void*) { return ps::Status::Error("no pull_handler"); };
    }
    pull_handler->call(&items, _timeout);

    return [this, handler_pool, pull_handler](void* result) {
        core::vector<EmbeddingPullResults> block_items(1);
        block_items[0] = *static_cast<EmbeddingPullResults*>(result);
        pull_handler->set_wait_result(&block_items);
        ps::Status status = pull_handler->wait();
        if (block_items[0].should_persist) {
            _should_persist->store(true, std::memory_order_relaxed);
        }
        handler_pool->release(std::unique_ptr<ps::UDFHandler>(pull_handler));
        return status;
    };
}

HandlerWaiter EmbeddingVariableHandle::push_gradients(const uint64_t* indices, size_t n, const char* gradients)const {
    VTIMER(1, embedding_variable, push_gradients, ms);
    SCHECK(!_read_only);
    
    core::vector<EmbeddingPushItems> items(1);
    items[0].variable_id = _variable_id;
    items[0].meta = _meta;
    items[0].indices = indices;
    items[0].n = n;
    items[0].gradients = gradients;

    HandlerPointer<ps::UDFHandler> handler(_push_handler);
    if (handler) {
        handler->call(&items, _timeout);
    }
    return handler.done_waiter();
}


EmbeddingVariableHandle EmbeddingStorageHandler::variable(uint32_t variable_id, EmbeddingVariableMeta meta) {
    EmbeddingVariableHandle variable;
    variable._timeout = _timeout;
    variable._read_only = false;
    variable._variable_id = variable_id;
    variable._meta = std::move(meta);
    variable._read_only_pull_handler = &_read_only_pull_handler;
    variable._pull_handler = &_pull_handler;
    variable._push_handler = &_push_handler;
    variable._init_handler = &_init_handler;
    return variable;
}

HandlerWaiter EmbeddingStorageHandler::update_weights() {
    HandlerPointer<ps::UDFHandler> handler(&_store_handler);
    if (handler) {
        handler->call(&_timeout, _timeout);
    }
    return handler.done_waiter();
}

// predictor controller
HandlerWaiter EmbeddingStorageHandler::load_storage(const URIConfig& uri, size_t server_concurency) {
    std::string hadoop_bin;
    uri.config().get_val(core::URI_HADOOP_BIN, hadoop_bin);
    
    // Hack load handler for shared file system with out hdfs.
    core::URIConfig hdfs("hdfs://");
    hdfs.config() = uri.config();
    // The type of the path not changed after set name, and it is still treated as a shared path.
    // After the client lists the files and sends them to the server,
    // the server will re-parse out the correct path type.
    // Finally, the server will load each file in the correct way.
    hdfs.set_name(uri.name());

    bool restore_model = false;
    uri.config().get_val("restore_model", restore_model);
    HandlerPointer<ps::LoadHandler> handler(&_load_handler);
    if (handler) {
        if (restore_model) {
            handler->restore(hdfs, false, hadoop_bin, server_concurency, _timeout);
        } else {
            handler->load(hdfs, hadoop_bin, server_concurency, _timeout);
        }
    }
    return handler.done_waiter();
}

// predictor controller
HandlerWaiter EmbeddingStorageHandler::dump_storage(const URIConfig& uri, size_t file_number) {
    std::string hadoop_bin;
    uri.config().get_val(core::URI_HADOOP_BIN, hadoop_bin);
    HandlerPointer<ps::DumpHandler> handler(&_dump_handler);
    if (handler) {
        handler->dump(ps::DumpArgs(uri, file_number, hadoop_bin));
    }
    return handler.done_waiter();
}



}
}
}
