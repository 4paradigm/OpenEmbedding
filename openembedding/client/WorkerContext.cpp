#include "WorkerContext.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

WorkerContext::WorkerContext(RpcConnection* connection,
        int32_t worker_num, int32_t wait_server_num) {
    _conn = connection;
    _comm = std::make_unique<Communication>(connection->rpc(), worker_num);
    LogReporter::set_id("WORKER", _comm->comm_rank());
    if (wait_server_num == -1) {
        ps::ServerConfig server_config;
        _server = connection->create_server();
        _server->initialize();
    }

    _comm->barrier("WorkerContext");
    connection->rpc()->update_ctx();
    _comm->barrier("WorkerContext");
    _model = std::make_unique<Model>(_conn);
    
    
    if (_conn->env_config().server.report_interval > 0) {
        static bool first = true;
        if (!first) {
            SLOG(WARNING) << "only one learner context can use reporter!";
            SLOG(WARNING) << "maybe hang if create learner context parallelly！";
        } else {
            _reporter = true;
            paradigm4::pico::core::pico_is_evaluate_performance() = true;
            if (_comm->comm_rank() == 0) {
                AccumulatorServer::singleton().initialize(connection->rpc());
                _report_monitor = pico_monitor().submit("accumulator_reporter",
                    0,
                    static_cast<uint64_t>(_conn->env_config().server.report_interval * 1000),
                    [&] { report_accumulator(); });
            }
            AccumulatorClient::singleton().initialize(connection->rpc());
        }
    }
}

WorkerContext::~WorkerContext() {
    if (_reporter) {
        AccumulatorClient::singleton().finalize();
        _comm->barrier("~WorkerContext");
        if (_comm->comm_rank() == 0) {
            AccumulatorServer::singleton().finalize();
            pico_monitor().destroy_with_additional_run(_report_monitor).wait();
        }
    }

    _model.reset();
    _comm->barrier("~WorkerContext");
    if (_comm->comm_rank() == 0) {
        _conn->close_servers();
    }
    _comm->barrier("~WorkerContext");
    if (_server) {
        _server->finalize();
    }
    _comm->barrier("~WorkerContext");
}

int32_t WorkerContext::create_storage(int32_t shard_num) {
    static std::atomic<size_t> ino = {0};
    std::map<int32_t, std::vector<int32_t>> node_shards;
    std::vector<int32_t> servers = _conn->running_servers();
    if (shard_num == -1) {
        shard_num = servers.size();
    }
    size_t start = ino.fetch_add(1);
    for (int32_t shard_id = 0; shard_id < shard_num; ++shard_id) {
        node_shards[(shard_id + start) % servers.size()].push_back(shard_id);
    }
    int32_t storage_id = _comm->sync_bcast("create_storage", [this, node_shards]() {
        int32_t storage_id;
        SCHECK(_conn->create_storage(node_shards, storage_id).ok());
        return storage_id;
    });
    core::lock_guard<core::RWSpinLock> lk(_lock);
    _model->add_storage(storage_id, std::to_string(storage_id));
    return storage_id;
}

void WorkerContext::delete_storage(int32_t storage_id) {
    std::string name = "delete_storage" + std::to_string(storage_id);
    _comm->sync_bcast(name, [this, storage_id]() {
        SCHECK(_conn->delete_storage(storage_id).ok());
        return true;
    });
}

EmbeddingVariableHandle WorkerContext::create_variable(int32_t storage_id, const EmbeddingVariableMeta& meta) {
    ModelVariableMeta variable;
    variable.meta = meta;
    variable.storage_name = std::to_string(storage_id);
    EmbeddingVariableHandle handle;
    uint32_t variable_id = _model->model_meta().variables.size();
    {
        core::lock_guard<core::RWSpinLock> lk(_lock);
        SCHECK(_model->add_variable(variable).ok());
        SCHECK(_model->access_variable(variable_id, handle).ok());
    }
    std::string name = "create_variable" + std::to_string(variable_id);
    _comm->sync_bcast(name, [this, &handle]() {
        handle.init_config(core::Configure());
        return true;
    });
    return handle;
}

HandlerWaiter WorkerContext::update_weights(int32_t storage_id) {
    if (storage_id % _comm->comm_size() == _comm->comm_rank()) {
        core::shared_lock_guard<core::RWSpinLock> lk(_lock);
        EmbeddingStorageHandler* storage = nullptr;
        SCHECK(_model->access_storage(storage_id, storage).ok());
        return storage->update_weights();
    }
    return [](void*){ return ps::Status(); };
}

void WorkerContext::load_model(const core::URIConfig& uri)const {
    ModelOfflineMeta model_meta;
    _model->read_meta_file(uri, model_meta);
    if (_comm->load_model_sign(model_meta.model_sign)) {
        SCHECK(_model->load_model(uri).ok());
    }
}

void WorkerContext::dump_model(const core::URIConfig& uri, const std::string& model_sign) {
    SCHECK(_model->dump_model(uri, model_sign).ok());
}

void WorkerContext::report_accumulator() {
    auto output_info = AccumulatorServer::singleton().generate_output_info();
    if (output_info.size() == 0) {
    SLOG(INFO) << "===== No Accumulator =====";
        return;
    }

    SLOG(INFO) << "======== ACCUMULATOR INFO[" << "] ======";
    size_t max_name_len = 0;
    for (auto item : output_info) {
        max_name_len = std::max(item.first.length(), max_name_len);
    }
    std::string whitespace;
    whitespace.reserve(max_name_len + 2);
    for (auto item : output_info) {
        int whitespace_num = max_name_len - item.first.length() + 2;
        whitespace.clear();
        for (int i = 0; i < whitespace_num; i++) {
            whitespace.push_back(' ');
        }
        SLOG(INFO) << whitespace << item.first << " : " << item.second;
    }
    SLOG(INFO) << "==================================";   
}

}
}
}
