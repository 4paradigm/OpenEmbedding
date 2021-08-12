#include "ModelController.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class ModelUnlockGuard {
public:
    ModelUnlockGuard(RpcConnection* conn, std::string model_sign): conn(conn), model_sign(model_sign) {};
    ~ModelUnlockGuard() {
        if (conn) {
            conn->unlock_model(model_sign);
        }
    }

    void async() {
        conn = nullptr;
    }

    RpcConnection* conn;
    std::string model_sign;
};

ps::Status ModelManager::find_model_variable(const std::string& model_sign, uint32_t variable_id,
        std::shared_ptr<Model>& out, EmbeddingVariableHandle& handle, int timeout) {
    out = nullptr;
    ModelMeta model_meta;
    CHECK_STATUS_RETURN(_conn->pull_model_meta(model_sign, model_meta));
    core::lock_guard<core::RWSpinLock> lk(_lock);
    if (!_models.count(model_sign)) {
        _models[model_sign] = std::make_shared<Model>(_conn);
    }
    out = _models[model_sign];
    CHECK_STATUS_RETURN(out->update_model_meta(model_meta));
    if (out->model_meta().model_status == ps::ModelStatus::CREATING) {
        RETURN_WARNING_STATUS(ps::Status::Error("model is in CREATING, pull is not safe"));
    }
    CHECK_STATUS_RETURN(out->access_variable(variable_id, handle));
    if (timeout != -1) {
        handle._timeout = timeout;
    }
    handle._read_only = true;
    return ps::Status();
}

// for controller, all heavy methods are async
ps::Status ModelController::create_model(const core::URIConfig& model_uri,
      std::string& model_sign, core::PicoJsonNode& result, int32_t replica_num, int32_t shard_num) {
    std::shared_ptr<Model> model = std::make_shared<Model>(_conn);
    CHECK_STATUS_RETURN(model->create_model(model_uri));
    model_sign = model->model_meta().model_sign;
    if (!_conn->try_lock_model(model_sign)) {
        return ps::Status::InvalidID("model is in processing: " + model_sign);
    }
    ModelUnlockGuard guard(_conn, model_sign);

    // 处理中断的CREATING
    ModelMeta model_meta;
    ps::Status status = _conn->pull_model_meta(model_sign, model_meta);
    if (!status.ok() && !status.IsInvalidID()) {
        return status;
    }
    if (status.ok()) {
        if (model_meta.model_status != ps::ModelStatus::CREATING) {
            return ps::Status::InvalidID("model sign already exist: " + model_sign);
        }
        model_meta.model_uri = model_uri.uri();
        CHECK_STATUS_RETURN(model->update_model_meta(model_meta));
        CHECK_STATUS_RETURN(model->create_model_storages(replica_num, shard_num));
    } else {
        model->set_model_status(ps::ModelStatus::CREATING);
        CHECK_STATUS_RETURN(model->create_model_storages(replica_num, shard_num));
        CHECK_STATUS_RETURN(_conn->push_model_meta(model->model_meta()));
    }
    result = model->model_meta().to_json_node();
    _threads.async_exec([this, model, model_sign](int) {
        ModelUnlockGuard guard(_conn, model_sign);
        if (model->test_status(model->load_model()).ok()) {
            model->set_model_status(ps::ModelStatus::NORMAL);
        }
        _conn->update_model_meta(model->model_meta());
    });
    guard.async();
    return ps::Status();
}

ps::Status ModelController::delete_model(const std::string& model_sign) {
    if (!_conn->try_lock_model(model_sign)) {
        return ps::Status::InvalidID("model is in processing: " + model_sign);
    }
    ModelUnlockGuard guard(_conn, model_sign);
    
    ModelMeta model_meta;
    CHECK_STATUS_RETURN(_conn->pull_model_meta(model_sign, model_meta));
    if (model_meta.model_status != ps::ModelStatus::NORMAL &&
        model_meta.model_status != ps::ModelStatus::DELETING) {
        std::string act = "model " + ModelMeta::to_string(model_meta.model_status);
        return ps::Status::InvalidID(act + " is suspended, need complete: " + model_sign);
    }

    std::shared_ptr<Model> model = std::make_shared<Model>(_conn);
    CHECK_STATUS_RETURN(model->update_model_meta(model_meta));
    model->set_model_status(ps::ModelStatus::DELETING);
    CHECK_STATUS_RETURN(_conn->update_model_meta(model->model_meta()));

    _threads.async_exec([this, model, model_sign](int) {
        ModelUnlockGuard guard(_conn, model_sign);
        model->delete_model_storages();
        _conn->delete_model_meta(model_sign);
    });
    guard.async();
    return ps::Status();
}

ps::Status ModelController::show_model(const std::string& model_sign, core::PicoJsonNode& result) {
    ModelMeta model_meta;
    CHECK_STATUS_RETURN(_conn->pull_model_meta(model_sign, model_meta));
    result = core::PicoJsonNode::object();
    result.add(model_meta.model_sign, model_meta.to_json_node());
    return ps::Status();
}

ps::Status ModelController::show_models(core::PicoJsonNode& result) {
    result = core::PicoJsonNode::object();
    std::vector<std::string> model_signs = _conn->list_model();
    for (std::string model_sign: model_signs) {
        result.add(model_sign, core::PicoJsonNode::object());
    }
    return ps::Status();
}

ps::Status ModelController::show_node(int32_t node_id, core::PicoJsonNode& result) {
    auto controller = _conn->create_controller();
    std::string str = controller->show_node(node_id);
    if (str.empty()) {
        std::string error = controller->get_last_error();
        if (error == "node not exist") {
            return ps::Status::InvalidID(error + ": " + std::to_string(node_id));
        }
        if (!error.empty()) {
            return ps::Status::Error(error + ": " + std::to_string(node_id));
        }
    }
    result = core::PicoJsonNode::object();
    result.add(std::to_string(node_id), core::PicoJsonNode::object());
    return ps::Status();
}

ps::Status ModelController::show_nodes(core::PicoJsonNode& result) {
    std::vector<int32_t> nodes = _conn->running_servers();
    result = core::PicoJsonNode::object();
    for (int32_t node_id: nodes) {
        result.add(std::to_string(node_id), core::PicoJsonNode::object());
    }
    return ps::Status();
}

ps::Status ModelController::shutdown_node(int32_t node_id) {
    auto controller = _conn->create_controller();
    if (!controller->shutdown_node(node_id)) {
        return ps::Status::Error(controller->get_last_error());
    }
    return ps::Status();
}  

}
}
}
