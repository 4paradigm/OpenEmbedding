#include "Connection.h"
#include "EmbeddingDumpOperator.h"
#include "EmbeddingInitOperator.h"
#include "EmbeddingLoadOperator.h"
#include "EmbeddingPullOperator.h"
#include "EmbeddingPushOperator.h"
#include "EmbeddingRestoreOperator.h"
#include "EmbeddingStorage.h"
#include "EmbeddingStoreOperator.h"
#include "PersistManager.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

using ps::Operator;
REGISTER_OPERATOR(embedding, EmbeddingDumpOperator);
REGISTER_OPERATOR(embedding, EmbeddingInitOperator);
REGISTER_OPERATOR(embedding, EmbeddingLoadOperator);
REGISTER_OPERATOR(embedding, EmbeddingPullOperator);
REGISTER_OPERATOR(embedding, EmbeddingPushOperator);
REGISTER_OPERATOR(embedding, EmbeddingRestoreOperator);
REGISTER_OPERATOR(embedding, EmbeddingStorageOperator);
REGISTER_OPERATOR(embedding, EmbeddingStoreOperator);

void Connection::set_default_hadoop_bin(core::URIConfig& uri) {
    std::string hadoop_bin;
    uri.config().get_val(core::URI_HADOOP_BIN, hadoop_bin);
    if (uri.storage_type() == core::FileSystemType::HDFS && hadoop_bin.empty()) {
        uri.config().set_val(core::URI_HADOOP_BIN, "hdfs dfs");
    }
}

ps::Status Connection::create_operator(int32_t storage_id, const std::string& key,
        int32_t& handler_id, std::shared_ptr<ps::Operator>& op) {
    op = nullptr;
    handler_id = -1;
    ps::Status status;
    ps::TableDescriptorReader reader;
    CHECK_STATUS_RETURN(server_context()->GetTableDescriptorReader(storage_id, reader));
    core::Configure op_config;
    if (!reader.table().key_to_hdl.count(key)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidID("handler key not found"));
    }
    handler_id = reader.table().key_to_hdl.at(key);
    ps::OperatorDescriptor opd = reader.table().op_descs.at(handler_id);
    op_config.load(opd.config_str);
    op = ps::OperatorFactory::singleton().create(opd.lib_name, opd.op_name, op_config);
    if (op == nullptr) {
        RETURN_WARNING_STATUS(ps::Status::InvalidID("operator not found"));
    }
    return ps::Status();
}



RpcConnection::RpcConnection(const EnvConfig& env) {
    // RDMA will also use tcp_config.
    core::TcpSocket::set_tcp_config(_env.rpc.tcp);
    _env = env;
    if (_env.server.server_concurrency == -1) {
        _env.server.server_concurrency = std::thread::hardware_concurrency();
    }

    SLOG(INFO) << "server concurrency: " << _env.server.server_concurrency;
    SCHECK(!_env.master.endpoint.empty());
    if (_env.master.type == "tcp") {
        _master_client = std::make_unique<core::TcpMasterClient>(
                _env.master.endpoint, _env.master.root_path);
    } else if (_env.master.type == "zk") {
        _master_client = std::make_unique<core::ZkMasterClient>(
                _env.master.endpoint, _env.master.root_path,
                _env.master.recv_timeout, _env.master.recv_timeout);
    } else {
        SLOG(FATAL) << "unknown master type.";
    }
    _master_client->initialize();
    _rpc = std::make_unique<core::RpcService>();
    _rpc->initialize(_master_client.get(), _env.rpc);
    _rpc_client = _rpc->create_client(PSERVER_C2S_RPC_NAME, 0);
    _client = std::make_unique<ps::Client>();
    _client->initialize(_master_client.get(), _rpc_client.get());
    _master_client->tree_node_add(_model_path);
    _master_client->tree_node_add(_model_lock_path);

    VariableAsyncTaskThreadPool::singleton().initialize(_env.server.server_concurrency);
    if (!_env.server.pmem_pool_root_path.empty()) {
        SLOG(INFO) << "using pmem with dram cache size: " << _env.server.cache_size << "MB";
        PersistManager::singleton().initialize(
              _env.server.pmem_pool_root_path + "/rank" + std::to_string(_rpc->global_rank()));
        PersistManager::singleton().set_cache_size(_env.server.cache_size << 20);   
    }
}

RpcConnection::~RpcConnection() {
    _client->finalize();
    _client.reset();
    _rpc_client.reset();
    _rpc->finalize();
    _master_client->finalize();
    _master_client.reset();
    VariableAsyncTaskThreadPool::singleton().finalize();
}

std::unique_ptr<ps::Server> RpcConnection::create_server() {
    ps::ServerConfig config;
    config.server_c2s_thread_num = _env.server.server_concurrency;
    config.server_s2s_thread_num = _env.server.server_concurrency;
    config.server_load_thread_num = _env.server.server_concurrency;

    return std::make_unique<ps::Server>(config, _master_client.get(), _rpc.get());
}

std::unique_ptr<ps::Controller> RpcConnection::create_controller() {
    return std::make_unique<ps::Controller>(
            _master_client.get(), _client.get(), _env.server.recv_timeout);
}

std::vector<int32_t> RpcConnection::running_servers() {
    std::vector<int32_t> servers;
    try {
        std::unique_ptr<ps::Controller> controller = create_controller();
        controller->load_nodes();
        std::vector<ps::Node*> nodes = controller->get_running_nodes();
        for (ps::Node* node: nodes) {
            servers.push_back(node->node_id);
        }
    } catch (ps::CtlExpection& exception) {
        SLOG(FATAL) << exception.what();
    }
    return servers;
}

ps::Status RpcConnection::close_server(int32_t server_id) {
    SLOG(INFO) << "closing " << server_id;
    return _client->close_pserver(server_id, _env.server.recv_timeout);
}

void RpcConnection::close_servers() {
    std::vector<int> server_ids;
    _client->get_pserver_list(server_ids);
    for (int server_id: server_ids) {
        SCHECK(_client->close_pserver(server_id).ok());
    }
}

ps::Status RpcConnection::create_storage(const std::map<int32_t, std::vector<int32_t>>& node_shards, int32_t& storage_id) {
    core::Configure config;
    for (auto& pair: node_shards) {
        YAML::Node node;
        for (int32_t shard_id: pair.second) {
            node["shard_list"].push_back(shard_id);
        }
        node["g_rank"] = pair.first;
        config.node()["nodes"].push_back(node);
    }
    core::Configure op_config;
    op_config.node()["update_early_return"] = _env.server.update_early_return;
    op_config.node()["compress_algorithm"] = _env.server.message_compress; 
    config.node()["op_config"] = op_config.node();

    int timeout = _env.server.recv_timeout;
    CHECK_STATUS_RETURN(_client->create_storage(
            "embedding", "EmbeddingStorageOperator", config, storage_id, timeout));
    int32_t handler_id;
    op_config.node()["read_only"] = true;
    CHECK_STATUS_RETURN(_client->register_handler("read_only_pull", "embedding",
            "EmbeddingPullOperator", op_config, storage_id, handler_id, timeout));
    op_config.node()["read_only"] = false;
    CHECK_STATUS_RETURN(_client->register_handler("pull", "embedding",
            "EmbeddingPullOperator", op_config, storage_id, handler_id, timeout));
    CHECK_STATUS_RETURN(_client->register_handler("push", "embedding",
            "EmbeddingPushOperator", op_config, storage_id, handler_id, timeout));
    CHECK_STATUS_RETURN(_client->register_handler("store", "embedding",
            "EmbeddingStoreOperator", op_config, storage_id, handler_id, timeout));
    CHECK_STATUS_RETURN(_client->register_handler("init", "embedding",
            "EmbeddingInitOperator", op_config, storage_id, handler_id, timeout));
    CHECK_STATUS_RETURN(_client->register_handler("dump", "embedding",
            "EmbeddingDumpOperator", op_config, storage_id, handler_id, timeout));
    CHECK_STATUS_RETURN(_client->register_handler("load", "embedding",
            "EmbeddingLoadOperator", op_config, storage_id, handler_id, timeout));
    return ps::Status();
}

ps::Status RpcConnection::delete_storage(int32_t storage_id) {
    return _client->delete_storage(storage_id);
}

ps::Status RpcConnection::create_storage_handler(int32_t storage_id, std::unique_ptr<EmbeddingStorageHandler>& storage) {
    _client->initialize_storage(storage_id);
    storage = std::make_unique<EmbeddingStorageHandler>();
    storage->_timeout = _env.server.recv_timeout;
    create_handler_pool(storage_id, "read_only_pull", storage->_read_only_pull_handler);
    create_handler_pool(storage_id, "pull", storage->_pull_handler);
    create_handler_pool(storage_id, "push", storage->_push_handler);
    create_handler_pool(storage_id, "store", storage->_store_handler);
    create_handler_pool(storage_id, "init", storage->_init_handler);
    create_handler(storage_id, "dump", storage->_dump_handler);
    create_handler(storage_id, "load", storage->_load_handler);
    return ps::Status();
}

/// TODO: hold error
ps::Status RpcConnection::set_storage_restore_uri(int32_t storage_id, const core::URIConfig& uri) {
    _client->set_table_uri(storage_id, uri.uri());
    return ps::Status();
}

uint32_t RpcConnection::generate_id(const std::string& name) {
    return _master_client->generate_id("openembedding-" + name);
}

ps::Status RpcConnection::pull_model_meta(const std::string& model_sign, ModelMeta& model_meta) {
    if (model_sign.empty()) {
        return ps::Status::InvalidID("empty model sign");
    }
    std::string path = _model_path + '/' + model_sign;
    std::string str;
    if (!_master_client->tree_node_get(path, str)) {
        // Not print WARNNING.
        return ps::Status::InvalidID("model sign not exist: " + model_sign);
    }
    core::PicoJsonNode json;
    if (!json.load(str)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidConfig("invalid model meta: " + str));
    }
    if (!model_meta.from_json_node(json)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidConfig("invalid model meta: " + str));
    }
    return ps::Status();
}

ps::Status RpcConnection::push_model_meta(const ModelMeta& model_meta) {
    std::string path = _model_path + '/' + model_meta.model_sign;
    core::PicoJsonNode json = model_meta.to_json_node();
    if (!_master_client->tree_node_add(path, json.dump())) {
        RETURN_WARNING_STATUS(ps::Status::InvalidID("model sign already exist: " + model_meta.model_sign));
    }
    return ps::Status();
}

ps::Status RpcConnection::update_model_meta(const ModelMeta& model_meta) {
    std::string path = _model_path + '/' + model_meta.model_sign;
    core::PicoJsonNode json = model_meta.to_json_node();
    if (!_master_client->tree_node_set(path, json.dump())) {
        RETURN_WARNING_STATUS(ps::Status::InvalidID("model sign not exist: " + model_meta.model_sign));
    }
    return ps::Status();
}

ps::Status RpcConnection::delete_model_meta(const std::string& model_sign) {
    if (model_sign.empty()) {
        return ps::Status::InvalidID("empty model sign");
    }
    std::string path = _model_path + '/' + model_sign;
    if (!_master_client->tree_node_del(path)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidID("model sign not exist: " + model_sign));
    }
    return ps::Status();
}

std::vector<std::string> RpcConnection::list_model() {
    std::vector<std::string> children;
    _master_client->tree_node_sub(_model_path, children);
    return children;
}

bool RpcConnection::try_lock_model(const std::string& model_sign) {
    std::string path = _model_lock_path + '/' + model_sign;
    return _master_client->tree_node_add(path, "", true);
}

void RpcConnection::unlock_model(const std::string& model_sign) {
    std::string path = _model_lock_path + '/' + model_sign;
    _master_client->tree_node_del(path);
}



}
}
}
