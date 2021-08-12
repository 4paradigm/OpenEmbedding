#ifndef PARADIGM4_HYPEREMBEDDING_CONNECTION_H
#define PARADIGM4_HYPEREMBEDDING_CONNECTION_H

#include <pico-core/PicoJsonNode.h>
#include <pico-core/RpcServer.h>
#include <pico-core/MasterClient.h>
#include <pico-ps/service/Server.h>
#include <pico-ps/service/Client.h>
#include <pico-ps/common/defs.h>
#include <pico-ps/controller/Controller.h>

#include "Meta.h"
#include "EmbeddingVariableHandle.h"
#include "EnvConfig.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class Connection {
public:
    virtual ~Connection() {};
    virtual comm_rank_t global_rank() = 0;
    virtual ps::Context* server_context() = 0;
    virtual std::vector<int32_t> running_servers() = 0;
    virtual ps::Status close_server(int32_t server_id) = 0;
    virtual void close_servers() = 0;
    virtual ps::Status create_storage(const std::map<int32_t, std::vector<int32_t>>& node_shards, int32_t& storage_id) = 0;
    virtual ps::Status delete_storage(int32_t storage_id) = 0;
    virtual ps::Status set_storage_restore_uri(int32_t storage_id, const core::URIConfig& uri) = 0;

    virtual ps::Status create_storage_handler(int32_t storage_id, std::unique_ptr<EmbeddingStorageHandler>&) = 0;
    virtual uint32_t generate_id(const std::string&) = 0;
    virtual ps::Status pull_model_meta(const std::string& model_sign, ModelMeta& model_meta) = 0;
    

    virtual const EnvConfig& env_config()const = 0;

    void set_default_hadoop_bin(core::URIConfig& uri);

protected:
    ps::Status create_operator(int32_t storage_id, const std::string& key,
          int32_t& handler_id, std::shared_ptr<ps::Operator>& op);
};


class RpcConnection: public Connection {
public:
    RpcConnection(const EnvConfig& env);

    ~RpcConnection() override;

    comm_rank_t global_rank() override {
        return _rpc->global_rank();
    }

    std::unique_ptr<ps::Server> create_server();

    std::unique_ptr<ps::Controller> create_controller();

    ps::Context* server_context() override {
        return _client->context().get();
    }

    std::vector<int32_t> running_servers()override;

    ps::Status close_server(int32_t server_id)override;

    void close_servers() override;

    ps::Status create_storage(const std::map<int32_t, std::vector<int32_t>>& node_shards, int32_t& storage_id)override;

    ps::Status delete_storage(int32_t storage_id)override;

    ps::Status create_storage_handler(int32_t storage_id, std::unique_ptr<EmbeddingStorageHandler>& storage)override;

    ps::Status set_storage_restore_uri(int32_t storage_id, const core::URIConfig& uri);

    uint32_t generate_id(const std::string& name);

    ps::Status pull_model_meta(const std::string& model_sign, ModelMeta& model_meta)override;

    ps::Status push_model_meta(const ModelMeta& model_meta);

    ps::Status update_model_meta(const ModelMeta& model_meta);
    
    ps::Status delete_model_meta(const std::string& model_sign);

    std::vector<std::string> list_model();

    bool try_lock_model(const std::string& model_sign);

    void unlock_model(const std::string& model_sign);

    const EnvConfig& env_config()const override {
        return _env;
    }

    core::RpcService* rpc()const {
        return _rpc.get();
    }

    core::MasterClient* master_client()const {
        return _master_client.get();
    }

private:
    template<class T>
    ps::Status create_handler(int32_t storage_id, const std::string& key, std::unique_ptr<T>& handler) {
        int32_t handler_id = -1;
        std::shared_ptr<ps::Operator> op;
        CHECK_STATUS_RETURN(create_operator(storage_id, key, handler_id, op));
        handler = std::make_unique<T>(storage_id, handler_id, op, _client.get());
        return ps::Status();
    }

    template<class T>
    void create_handler_pool(int32_t storage_id, const std::string& key,
          ObjectPool<std::unique_ptr<T>>& handler_pool) {
        handler_pool = [this, storage_id, key]() {
            std::unique_ptr<T> handler;
            ps::Status status = create_handler(storage_id, key, handler);
            if (!status.ok()) {
                SLOG(WARNING) << key << " " << status.ToString();
            }
            return handler;
        };
    }

    std::string _model_path = "_hyper-embedding-model_";
    std::string _model_lock_path = "_hyper-embedding-model-lock_";
    std::unique_ptr<core::RpcService> _rpc;
    std::unique_ptr<core::MasterClient> _master_client;
    std::unique_ptr<core::RpcClient> _rpc_client;
    std::unique_ptr<ps::Client> _client;
    EnvConfig _env;
};

}
}
}

#endif
