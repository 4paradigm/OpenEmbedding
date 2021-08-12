#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_WORKER_CONTEXT_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_WORKER_CONTEXT_H

#include <pico-ps/service/Server.h>
#include "Connection.h"
#include "Communication.h"
#include "EmbeddingVariableHandle.h"
#include "Model.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class WorkerContext {
public:
    WorkerContext(RpcConnection* connection,
          int32_t worker_num, int32_t wait_server_num = -1);

    ~WorkerContext();

    int32_t create_storage(int32_t shard_num = -1);

    void delete_storage(int32_t storage_id);

    EmbeddingVariableHandle create_variable(int32_t storage_id, const EmbeddingVariableMeta& meta);

    HandlerWaiter update_weights(int32_t storage_id);

    int32_t worker_rank()const {
        return _comm->comm_rank();
    }

    int32_t worker_num()const {
        return _comm->comm_size();
    }

    Connection* connection()const {
        return _conn;
    }

    void load_model(const core::URIConfig& uri)const;

    void dump_model(const core::URIConfig& uri, const std::string& model_sign);

    void barrier(const std::string& key) {
        _comm->barrier(key);
    }

    template<class T>
    void boardcast(const std::string& key, T& value) {
        _comm->boardcast(key, value, 0);
    }

    void report_accumulator();

private:
    core::RWSpinLock _lock;
    Connection* _conn;
    std::unique_ptr<Communication> _comm;
    std::unique_ptr<ps::Server> _server;

    std::unique_ptr<Model> _model;

    ServerConfig _server_config;

    bool _reporter = false;
    size_t _report_monitor = 0;
};

}
}
}

#endif
