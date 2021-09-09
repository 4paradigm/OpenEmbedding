#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   

#include <stdint.h>
#include "c_api.h"

namespace paradigm4 {
namespace exb {

class Variable {
public:
    Variable(exb_storage* storage,
          uint64_t vocabulary_size, size_t embedding_dim, std::string datatype) {
        _handle = exb_create_variable(storage, vocabulary_size, embedding_dim, datatype.c_str());
    }

    intptr_t intptr() {
        return reinterpret_cast<intptr_t>(_handle);
    }

    uint32_t variable_id() {
        return exb_variable_id(_handle);
    }

    void set_initializer(std::map<std::string, std::string> config) {
        exb_initializer* initializer = exb_create_initializer(config["category"].c_str());
        for (auto& key_val: config) {
            if (key_val.first != "category") {
                exb_set_initializer_property(initializer, key_val.first.c_str(), key_val.second.c_str());
            }
        }
        exb_set_initializer(_handle, initializer);
    }

    void set_optimizer(std::map<std::string, std::string> config) {
        exb_optimizer* optimizer = exb_create_optimizer(config["category"].c_str());
        for (auto& key_val: config) {
            if (key_val.first != "category") {
                exb_set_optimizer_property(optimizer, key_val.first.c_str(), key_val.second.c_str());
            }
        }
        exb_set_optimizer(_handle, optimizer);
    }

private:
    exb_variable* _handle = nullptr;
};

class Storage {
public:
    Storage(const Storage&) = delete;
    Storage& operator= (const Storage&) = delete;

    Storage(exb_context* context, int32_t shard_num) {
        _handle = exb_create_storage(context, shard_num);
    }

    ~Storage() {
        if (_handle) {
            exb_fatal("Storage not finalize");
        }
    }

    void finalize() {
        exb_delete_storage(_handle);
        _handle = nullptr;
    }

    Variable create_variable(uint64_t vocabulary_size, size_t embedding_dim, std::string datatype) {
        return Variable(_handle, vocabulary_size, embedding_dim, datatype);
    }

    intptr_t intptr() {
        return reinterpret_cast<intptr_t>(_handle);
    }

    int32_t storage_id() {
        return exb_storage_id(_handle); 
    }

private:
    exb_storage* _handle = nullptr;
};

class Context {
public:
    Context(int32_t worker_num, int32_t wait_server_num, std::string model_uuid,
          std::string yaml_config, std::string master_endpoint, std::string rpc_bind_ip) {
        _connection = exb_connect(yaml_config.c_str(), master_endpoint.c_str(), rpc_bind_ip.c_str());
        _handle = exb_context_initialize(_connection, worker_num, wait_server_num);
        exb_string str;
        if (model_uuid.size() >= 128) {
            exb_fatal("uuid too long.");
        }
        memset(str.data, 0, sizeof(str));
        memcpy(str.data, model_uuid.data(), model_uuid.size());
        exb_barrier(_handle, "uuid", &str);
        _model_uuid = str.data;
    }

    ~Context() {
        if (_handle) {
            exb_fatal("Context not finalize");
        } 
    }

    void finalize() {
        exb_context_finalize(_handle);
        exb_disconnect(_connection);
        _handle = nullptr;
        _connection = nullptr;
    }

    intptr_t intptr() {
        return reinterpret_cast<intptr_t>(_handle);
    }

    int worker_rank() {
        return exb_worker_rank(_handle);
    }
    
    std::string model_uuid() {
        return _model_uuid;
    }

    std::unique_ptr<Storage> create_storage(int32_t shard_num) {
        return std::make_unique<Storage>(_handle, shard_num);
    }

    void save_model(std::string path, double model_version, bool include_optimizer) {
        int64_t ver = floor(model_version);
        std::string model_sign = _model_uuid + '-' + std::to_string(ver);
        if (include_optimizer) {
            exb_dump_model_include_optimizer(_handle, path.c_str(), model_sign.c_str());
        } else {
            exb_dump_model(_handle, path.c_str(), model_sign.c_str());
        }
    }

    void load_model(std::string path) {
        exb_load_model(_handle, path.c_str());
    }

private:
    std::string _model_uuid;
    exb_connection* _connection = nullptr;
    exb_context* _handle = nullptr;
};

class Master {
public:
    Master(std::string bind_ip) {
        _handle = exb_master_start(bind_ip.c_str());
    };

    ~Master() {
        if (_handle) {
            exb_fatal("Master not finalize");
        }
    }

    void finalize() {
        exb_master_join(_handle);
        _handle = nullptr;
    }

    std::string endpoint() {
        exb_string str;
        exb_master_endpoint(_handle, &str);
        return str.data;
    }
private:
    exb_master* _handle;
};

class Server {
public:
    Server(std::string yaml_config, std::string master_endpoint, std::string rpc_bind_ip) {
        _connection = exb_connect(yaml_config.c_str(), master_endpoint.c_str(), rpc_bind_ip.c_str());
        _handle = exb_server_start(_connection);
    }

    ~Server() {
        if (_handle) {
            exb_fatal("Server not join");
        }
    }

    void exit() {
        exb_server_exit(_handle);
    }

    void join() {
        exb_server_join(_handle);
        exb_disconnect(_connection);
        _handle = nullptr;
        _connection = nullptr;
    }
    exb_connection* _connection = nullptr;
    exb_server* _handle = nullptr;
};

std::string version() {
    return exb_version();
}

size_t checkpoint_batch_id() {
    return exb_checkpoint_batch_id();
}

}
}

using namespace paradigm4::exb;
PYBIND11_MODULE(libexb, m) {
    // exb_serving(); // import registered optimizer
    auto gil_scoped_release = pybind11::call_guard<pybind11::gil_scoped_release>();
    pybind11::class_<Master>(m, "Master")
        .def(pybind11::init<std::string>(), gil_scoped_release)
        .def("finalize", &Master::finalize, gil_scoped_release)
        .def_property_readonly("endpoint", &Master::endpoint);

    pybind11::class_<Context>(m, "Context")
        .def(pybind11::init<int32_t, int32_t, std::string, std::string, std::string, std::string>(), gil_scoped_release)
        .def("finalize", &Context::finalize, gil_scoped_release)
        .def("create_storage", &Context::create_storage, gil_scoped_release)
        .def("save_model", &Context::save_model, gil_scoped_release)
        .def("load_model", &Context::load_model, gil_scoped_release)
        .def_property_readonly("intptr", &Context::intptr)
        .def_property_readonly("worker_rank", &Context::worker_rank)
        .def_property_readonly("model_uuid", &Context::model_uuid);
        

    pybind11::class_<Storage>(m, "Storage")
        .def("finalize", &Storage::finalize, gil_scoped_release)
        .def("create_variable", &Storage::create_variable, gil_scoped_release)
        .def_property_readonly("intptr", &Storage::intptr)
        .def_property_readonly("storage_id", &Storage::storage_id);

    pybind11::class_<Variable>(m, "Variable")
        .def("set_initializer", &Variable::set_initializer)
        .def("set_optimizer", &Variable::set_optimizer)
        .def_property_readonly("intptr", &Variable::intptr)
        .def_property_readonly("variable_id", &Variable::variable_id);
    
    pybind11::class_<Server>(m, "Server")
        .def(pybind11::init<std::string, std::string, std::string>(), gil_scoped_release)
        .def("exit", &Server::exit, gil_scoped_release)
        .def("join", &Server::join, gil_scoped_release);

    m.def("version", &version);
    m.def("checkpoint_batch_id", &exb_checkpoint_batch_id);
}
