#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_MODEL_CONTROLLER_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_MODEL_CONTROLLER_H

#include "Model.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

// for predictor
class ModelManager {
public:
    ModelManager(Connection* connection): _conn(connection) {}

    /// TODO: cache pull_model_meta with timeout
    // predictor pull对timeout的需求可能与其他handler不同
    ps::Status find_model_variable(const std::string& model_sign, uint32_t variable_id,
          std::shared_ptr<Model>& out, EmbeddingVariableHandle& handle, int timeout = -1);

private:
    core::RWSpinLock _lock;
    Connection* _conn = nullptr;
    std::unordered_map<std::string, std::shared_ptr<Model>> _models; 
};


// for controller, all heavy methods are async
class ModelController {
public:
    ModelController(RpcConnection* connection): _conn(connection),
           _threads(_conn->env_config().server.server_concurrency) {}

    ps::Status create_model(const core::URIConfig& model_uri,
          std::string& model_sign, core::PicoJsonNode& result, int32_t replica_num, int32_t shard_num);

    ps::Status delete_model(const std::string& model_sign);

    ps::Status show_model(const std::string& model_sign, core::PicoJsonNode& result);

    ps::Status show_models(core::PicoJsonNode& result);

    ps::Status show_node(int32_t node_id, core::PicoJsonNode& result);

    ps::Status show_nodes(core::PicoJsonNode& result);

    ps::Status shutdown_node(int32_t node_id);
private:
    RpcConnection* _conn = nullptr;
    ThreadGroup _threads;
};

}
}
}

#endif
