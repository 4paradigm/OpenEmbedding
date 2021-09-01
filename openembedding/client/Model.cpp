#include "Model.h"

#include <pico-ps/service/Server.h>
#include "EmbeddingShardFile.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

void Model::set_model_status(ps::ModelStatus model_status) {
    _model_meta.model_status = model_status;
}

ps::Status Model::test_status(const ps::Status& status) {
    if (status.ok()) {
        _model_meta.model_error = "";
    } else {
        _model_meta.model_error = status.ToString();
    }
    return status;
}

ps::Status Model::update_model_meta(const ModelMeta& model_meta) {
    for (auto& pair: model_meta.storages) {
        int32_t storage_id = pair.second;
        if (!_storages.count(storage_id)) {
            std::unique_ptr<EmbeddingStorageHandler> handler;
            CHECK_STATUS_RETURN(_conn->create_storage_handler(storage_id, handler));
            SCHECK(_storages.emplace(storage_id, std::move(handler)).second);
        }
    }
    _model_meta = model_meta;
    return ps::Status();
};

ps::Status Model::add_storage(int32_t storage_id, std::string storage_name) {
    if (_model_meta.storages.count(storage_name)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidID("storage name already exists: " + storage_name));
    }
    _model_meta.storages.emplace(storage_name, storage_id);
    std::unique_ptr<EmbeddingStorageHandler> handler;
    CHECK_STATUS_RETURN(_conn->create_storage_handler(storage_id, handler));
    SCHECK(_storages.emplace(storage_id, std::move(handler)).second);
    return ps::Status();
}

ps::Status Model::add_variable(const ModelVariableMeta& variable) {
    if (!_model_meta.storages.count(variable.storage_name)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidID("storage not created: " + variable.storage_name));
    }
    _model_meta.variables.push_back(variable);
    return ps::Status();
}

ps::Status Model::access_storage(int32_t storage_id, EmbeddingStorageHandler*& storage)const {
    if (!_storages.count(storage_id)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidID("storage id not found: " + storage_id));
    }
    storage = _storages.at(storage_id).get();
    return ps::Status();
}

ps::Status Model::access_variable(uint32_t variable_id, EmbeddingVariableHandle& handle)const {
    if (variable_id >= _model_meta.variables.size()) {
        return ps::Status::InvalidID("variable id not found: " + variable_id);
    }
    ModelVariableMeta variable = _model_meta.variables[variable_id];
    if (!_model_meta.storages.count(variable.storage_name)) {
        return ps::Status::InvalidID("storage not created: " + variable.storage_name);
    }
    int32_t storage_id = _model_meta.storages.at(variable.storage_name);
    handle = _storages.at(storage_id)->variable(variable_id, variable.meta);
    return ps::Status();
}

ps::Status Model::dump_model(core::URIConfig uri, std::string model_sign)const {
    _conn->set_default_hadoop_bin(uri);
    FileWriter meta_file;
    core::FileSystem::create_output_dir(uri);
    SCHECK(meta_file.open(uri + "/model_meta"));
    ModelOfflineMeta model_meta;
    model_meta.model_sign = model_sign;
    model_meta.variables = _model_meta.variables;
    core::PicoJsonNode json = model_meta.to_json_node();
    std::string str = json.dump(4);
    meta_file.write(str.c_str(), str.length());
    for (auto& pair: _storages) {
        std::string path = "/" + std::to_string(pair.first);
        /// TODO: config file num
        CHECK_STATUS_RETURN(pair.second->dump_storage(uri + path, 4));
    }
    return ps::Status();
}

ps::Status Model::load_model(core::URIConfig uri) {
    _conn->set_default_hadoop_bin(uri);
    ModelOfflineMeta model_meta;
    CHECK_STATUS_RETURN(read_meta_file(uri, model_meta));
    if (model_meta.variables != _model_meta.variables) {
        RETURN_WARNING_STATUS(ps::Status::InvalidConfig("model meta not match"));
    }
    for (size_t variable_id = 0; variable_id < _model_meta.variables.size(); ++variable_id) {
        EmbeddingVariableHandle handle;
        CHECK_STATUS_RETURN(access_variable(variable_id, handle));
        CHECK_STATUS_RETURN(handle.clear_weights());
    }

    for (auto& pair: _model_meta.storages) {
        CHECK_STATUS_RETURN(_storages.at(pair.second)->load_storage(uri + "/" + pair.first));
        _conn->set_storage_restore_uri(pair.second, uri + "/" + pair.first);
    }
    return ps::Status();
}

ps::Status Model::load_model() {
    return load_model(core::URIConfig(_model_meta.model_uri));
}

ps::Status Model::create_model(core::URIConfig uri) {
    if (!_model_meta.model_sign.empty()) {
        return ps::Status::Error("model has created: " + _model_meta.model_sign);
    }
    _conn->set_default_hadoop_bin(uri);
    ModelOfflineMeta model_meta;
    CHECK_STATUS_RETURN(read_meta_file(uri, model_meta));
    _model_meta.model_sign = model_meta.model_sign;
    _model_meta.variables = model_meta.variables;
    _model_meta.model_uri = uri.uri();
    return ps::Status();
}

ps::Status Model::create_model_storages(int32_t replica_num, int32_t shard_num) {
    static std::atomic<size_t> ino = {0};
    
    std::map<int32_t, std::vector<int32_t>> node_shards;
    std::vector<int32_t> servers = _conn->running_servers();
    if (replica_num <= 0) {
        RETURN_WARNING_STATUS(ps::Status::InvalidConfig("invalid replica num"));
    }
    if (static_cast<int>(servers.size()) < (replica_num)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidConfig("not enough server for replica"));
    }
    if (shard_num == -1) {
        shard_num = servers.size();
    }
    if (shard_num <= 0) {
        RETURN_WARNING_STATUS(ps::Status::InvalidConfig("error shard num"));
    }
    size_t start = ino.fetch_add(shard_num * replica_num);
    for (int32_t shard_id = 0; shard_id < shard_num; ++shard_id) {
        for (int32_t i = 0; i < replica_num; ++i) {
            int32_t j = (shard_id * replica_num + i + start) % servers.size();
            node_shards[servers[j]].push_back(shard_id);
        }
    }
    
    for (auto& variable: _model_meta.variables) {
        if (!_model_meta.storages.count(variable.storage_name)) {
            int32_t storage_id;
            CHECK_STATUS_RETURN(_conn->create_storage(node_shards, storage_id));
            add_storage(storage_id, variable.storage_name);
        }
    }
    return ps::Status();
}

void Model::delete_model_storages() {
    for (auto& pair: _storages) {
        _conn->delete_storage(pair.first);
    }
    _model_meta.storages.clear();
    _storages.clear();
}

ps::Status Model::read_meta_file(const core::URIConfig& uri, ModelOfflineMeta& model_meta) {
    std::string str;
    core::PicoJsonNode json;
    FileReader reader;
    core::URIConfig mode_meta_uri = uri + "/model_meta";
    if (!reader.open(mode_meta_uri)) {
        RETURN_WARNING_STATUS(ps::Status::Error("open model file meta failed: " + mode_meta_uri.uri()));
    }
    char ch;
    while (reader.read(&ch, 1)) {
        str += ch;
    }
    if (!json.load(str)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidConfig("invalid model file meta: " + str));
    }
    std::string version;
    if (!json.at("version").try_as(version)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidConfig("model file meta miss version field"));
    }
    if (version != ModelOfflineMeta::version()) {
        RETURN_WARNING_STATUS(ps::Status::InvalidConfig("unsupport version: " + version));
    }
    if (!model_meta.from_json_node(json)) {
        RETURN_WARNING_STATUS(ps::Status::InvalidConfig("invalid model file meta: " + str));
    }
    return ps::Status();
}

}
}
}
