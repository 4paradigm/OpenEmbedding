#include "EmbeddingVariableHandle.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

void EmbeddingVariableHandle::init_config(const core::Configure& config)const {
    SCHECK(!_read_only);
    std::unique_ptr<EmbeddingInitItems> items = std::make_unique<EmbeddingInitItems>();
    items->variable_id = _variable_id;
    items->meta = _meta;
    items->variable_config = config.dump();
    std::string meta_str;
    _meta.to_json_node().save(meta_str);
    SLOG(INFO) << "variable " << meta_str << " init config:\n" << config.dump();

    std::unique_ptr<ps::PushHandler> init_handler = _init_handler->acquire();
    SCHECK(init_handler) << "no init handler";
    init_handler->async_push(std::move(items), _timeout);
    ps::Status status = init_handler->wait();
    _init_handler->release(std::move(init_handler));
    SCHECK(status.ok()) << status.ToString();
}

// predictor controller
ps::Status EmbeddingVariableHandle::clear_weights() {
    if (_read_only) {
        RETURN_WARNING_STATUS(ps::Status::Error("the variable is read only"));    
    }
    std::unique_ptr<EmbeddingInitItems> items = std::make_unique<EmbeddingInitItems>();
    items->variable_id = _variable_id;
    items->meta = _meta;
    items->clear_weights = true;
    SLOG(INFO) << "variable " << _variable_id << " clear_weights";

    std::unique_ptr<ps::PushHandler> init_handler = _init_handler->acquire();
    if (!init_handler) {
        RETURN_WARNING_STATUS(ps::Status::Error("no init handler"));
    }
    init_handler->async_push(std::move(items), _timeout);
    ps::Status status = init_handler->wait();
    _init_handler->release(std::move(init_handler));
    return status;
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

    ps::UDFHandler* push_handler = _push_handler->acquire().release();
    SCHECK(push_handler) << "no push_handler";
    push_handler->call(&items, _timeout);
    return [this, push_handler](void*) {
        ps::Status status = push_handler->wait();
        SCHECK(status.ok()) << status.ToString();
        _push_handler->release(std::unique_ptr<ps::UDFHandler>(push_handler));
        return status;
    };
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
    ps::UDFHandler* store_handler = _store_handler.acquire().release();
    store_handler->call(&_timeout, _timeout);
    return [this, store_handler](void*) {
        ps::Status status = store_handler->wait();
        SCHECK(status.ok()) << status.ToString();
        _store_handler.release(std::unique_ptr<ps::UDFHandler>(store_handler));
        return status;
    };
}

// predictor controller
ps::Status EmbeddingStorageHandler::load_storage(const URIConfig& uri, size_t server_concurency) {
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

    if (!_load_handler) {
        RETURN_WARNING_STATUS(ps::Status::Error("no load_handler"));
    }
    _load_handler->load(hdfs, hadoop_bin, server_concurency, _timeout);
    return _load_handler->wait();
}

// predictor controller
ps::Status EmbeddingStorageHandler::dump_storage(const URIConfig& uri, size_t file_number) {
    std::string hadoop_bin;
    uri.config().get_val(core::URI_HADOOP_BIN, hadoop_bin);

    if (!_dump_handler) {
        RETURN_WARNING_STATUS(ps::Status::Error("no dump_handler"));
    }
    _dump_handler->dump(ps::DumpArgs(uri, file_number, hadoop_bin));
    return _dump_handler->wait();
}



}
}
}
