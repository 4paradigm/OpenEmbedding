#include "c_api.h"
#include "Factory.h"
#include "WorkerContext.h"
#include "ModelController.h"
#include "EmbeddingRestoreOperator.h"

using namespace paradigm4::pico;
using namespace paradigm4::pico::embedding;

extern "C" {

exb_connection* exb_conn = nullptr;

struct exb_connection* exb_serving() {
    return exb_conn;
}

struct exb_connection {
    std::unique_ptr<RpcConnection> entity;
    std::unique_ptr<ModelManager> manager;
};

struct exb_master {
    std::unique_ptr<Master> entity;
};

struct exb_server {
    std::unique_ptr<ps::Server> entity;
};

struct exb_context {
    exb_connection* connection = nullptr;
    std::unique_ptr<WorkerContext> entity;
};

struct exb_storage {
    int32_t storage_id = 0;
    WorkerContext* context = nullptr;
    std::vector<exb_variable*> variables;
};

struct exb_variable {
    ModelManager* manager = nullptr;
    std::shared_ptr<Model> model;
    std::string model_sign;
    EmbeddingVariableHandle handle;
};

struct exb_optimizer {
    std::string category;
    core::Configure config;
};

struct exb_initializer {
    std::string category;
    core::Configure config;
};

struct exb_configure {
    core::Configure config;
};

struct exb_channel {
    core::RpcChannel<void*> entity;
};

std::string& exb_thread_local_error_string() {
    thread_local std::string str;
    return str;
}

int& exb_thread_local_wait_time() {
    thread_local int time;
    return time;
}


// thread local
const char* exb_last_error() {
    return exb_thread_local_error_string().c_str();
}

int exb_last_wait_time_ms() {
    return exb_thread_local_wait_time();
}


struct exb_connection* exb_connect(const char* yaml_config, const char* master_endpoint, const char* rpc_bind_ip) {
    EnvConfig env;
    core::Configure configure;
    configure.load(yaml_config);
    env.load_yaml(configure, master_endpoint, rpc_bind_ip);
    exb_connection* connection = new exb_connection;
    connection->entity = std::make_unique<RpcConnection>(env);
    connection->manager = std::make_unique<ModelManager>(connection->entity.get());
    return connection;
}

int exb_running_server_count(struct exb_connection* connection) {
    return connection->entity->running_servers().size();
}

void exb_disconnect(struct exb_connection* connection) {
    connection->manager.reset();
    connection->entity.reset();
    delete connection;
}

struct exb_master* exb_master_start(const char* bind_ip) {
    exb_master* master = new exb_master;
    master->entity = std::make_unique<core::Master>(bind_ip);
    master->entity->initialize();
    return master;
}

void exb_master_endpoint(struct exb_master* master, exb_string* value) {
    memset(value, 0, sizeof(*value));
    SCHECK(master->entity->endpoint().size() < 127);
    strcpy(value->data, master->entity->endpoint().c_str());
}

void exb_master_join(struct exb_master* master) {
    master->entity->exit();
    master->entity->finalize();
    delete master;
}

struct exb_server* exb_server_start(struct exb_connection* connection) {
    exb_server* server = new exb_server;
    server->entity = connection->entity->create_server();
    server->entity->initialize();
    return server;
}

void exb_server_exit(struct exb_server* server) {
    server->entity->exit();
}

void exb_server_join(struct exb_server* server) {
    server->entity->finalize();
    server->entity.reset();
    delete server;
}

struct exb_context* exb_context_initialize(exb_connection* connection,
      int32_t worker_num, int32_t wait_server_num) {
    exb_context* context = new exb_context;
    context->connection = connection;
    context->entity = std::make_unique<WorkerContext>(
          connection->entity.get(), worker_num, wait_server_num);
    return context;
}

void exb_context_finalize(struct exb_context* context) {
    delete context;
}


int exb_worker_rank(struct exb_context* context) {
    return context->entity->worker_rank();
}

struct exb_storage* exb_create_storage(struct exb_context* context, int32_t shard_num) {
    exb_storage* storage = new exb_storage;
    storage->context = context->entity.get();
    storage->storage_id = context->entity->create_storage(shard_num);
    return storage; 
}

void exb_delete_storage(struct exb_storage* storage) {
    storage->context->delete_storage(storage->storage_id);
    // In order to maintain the version, the memory need to be hold.
    // So cannot delete it here, temporarily treated as a global variable.
    /// TODO: Delete at the right time.
    // for (exb_variable* variable: storage->variables) {
    //     delete variable;
    // }
    // delete storage;
}

struct exb_variable* exb_create_variable(struct exb_storage* storage,
      uint64_t vocabulary_size, uint64_t embedding_dim, const char* dtype) {
    EmbeddingVariableMeta meta;
    meta.datatype = DataType(dtype);
    meta.embedding_dim = embedding_dim;
    meta.vocabulary_size = vocabulary_size;
    exb_variable* variable = new exb_variable;
    variable->handle = storage->context->create_variable(storage->storage_id, meta);
    storage->variables.push_back(variable);
    return variable;
}

int32_t exb_storage_id(struct exb_storage* storage) {
    return storage->storage_id;
}

uint32_t exb_variable_id(struct exb_variable* variable) {
    return variable->handle.variable_id();
}

void exb_set_optimizer(struct exb_variable* variable, struct exb_optimizer* optimizer_ptr) {
    core::Configure config;
    std::string optimizer = optimizer_ptr->category;
    SAVE_CONFIG(config, optimizer);
    config.node()[optimizer] = optimizer_ptr->config.node();
    variable->handle.init_config(config);
    delete optimizer_ptr;
}

void exb_set_initializer(struct exb_variable* variable, struct exb_initializer* initializer_ptr) {
    core::Configure config;
    std::string initializer = initializer_ptr->category;
    SAVE_CONFIG(config, initializer);
    config.node()[initializer] = initializer_ptr->config.node();
    variable->handle.init_config(config);
    delete initializer_ptr;
}

size_t exb_unique_indices(const uint64_t* indices, size_t n, size_t* unique) {
    EasyHashMap<uint64_t, size_t> mp(-1, n);
    for (size_t i = 0; i < n; ++i) {
        if (mp.count(indices[i])) {
            unique[i] = mp.at(indices[i]);
        } else {
            unique[i] = mp.size();
            mp.force_emplace(indices[i], unique[i]);
        }
    }
    return mp.size();
}

struct exb_pull_waiter* exb_pull_weights(const struct exb_variable* variable, const uint64_t* indices, size_t n, uint64_t version) {
    core::unique_ptr<HandlerWaiter> waiter = core::make_unique<HandlerWaiter>(
            variable->handle.pull_weights(indices, n, version));
    return reinterpret_cast<exb_pull_waiter*>(waiter.release());
}

struct exb_waiter* exb_push_gradients(struct exb_variable* variable, const uint64_t* indices, size_t n, const void* gradients) {
    core::unique_ptr<HandlerWaiter> waiter = core::make_unique<HandlerWaiter>(
            variable->handle.push_gradients(indices, n, reinterpret_cast<const char*>(gradients)));
    return reinterpret_cast<exb_waiter*>(waiter.release());
}

struct exb_waiter* exb_update_weights(struct exb_storage* storage) {
    core::unique_ptr<HandlerWaiter> waiter = core::make_unique<HandlerWaiter>(
            storage->context->update_weights(storage->storage_id));
    return reinterpret_cast<exb_waiter*>(waiter.release());
}

bool exb_pull_wait(struct exb_pull_waiter* waiter, const uint64_t* indices, size_t n, void* weights) {
    core::unique_ptr<HandlerWaiter> wait(reinterpret_cast<HandlerWaiter*>(waiter));
    EmbeddingPullResults items = {indices, n, reinterpret_cast<char*>(weights)};
    ps::Status status = wait->wait(&items);
    if (!status.ok()) {
        exb_thread_local_error_string() = status.ToString();
        return false;
    }
    return true;
}


bool exb_wait(struct exb_waiter* waiter) {
    core::unique_ptr<HandlerWaiter> wait(reinterpret_cast<HandlerWaiter*>(waiter));
    ps::Status status = wait->wait();
    if (!status.ok()) {
        exb_thread_local_error_string() = status.ToString();
        return false;
    }
    return true;
}

struct exb_optimizer* exb_create_optimizer(const char* category) {
    exb_optimizer* optimizer = new exb_optimizer;
    optimizer->category = category;
    return optimizer;
}

void exb_set_optimizer_property(struct exb_optimizer* optimizer, const char* key, const char* value) {
    optimizer->config.node()[key] = std::string(value);
}



struct exb_initializer* exb_create_initializer(const char* category) {
    exb_initializer* initializer = new exb_initializer();
    initializer->category = category;
    return initializer;
}

void exb_set_initializer_property(struct exb_initializer* initializer, const char* key, const char* value) {
    initializer->config.node()[key] = std::string(value);
}

void exb_dump_model_include_optimizer(struct exb_context* context, const char* path, const char* model_sign) {
    core::URIConfig uri(path);
    uri.config().set_val("include_optimizer", true);
    context->entity->dump_model(uri, model_sign);
}

void exb_dump_model(struct exb_context* context, const char* path, const char* model_sign) {
    core::URIConfig uri(path);
    uri.config().set_val("include_optimizer", false);
    context->entity->dump_model(uri, model_sign);
}

void exb_load_model(struct exb_context* context, const char* path) {
    core::URIConfig uri(path);
    context->entity->load_model(uri);
}

// HA is not required for standalone.
void exb_create_model(struct exb_connection* connection, const char* path, int32_t replica_num, int32_t shard_num) {
    core::URIConfig uri(path); 
    Model model(connection->entity.get());
    SCHECK(model.create_model(uri).ok());
    SCHECK(model.create_model_storages(replica_num, shard_num).ok());
    ps::Status status = model.load_model(uri);
    SCHECK(status.ok()) << status.ToString();
    model.set_model_status(ps::ModelStatus::NORMAL);
    SCHECK(connection->entity->push_model_meta(model.model_meta()).ok());
}

// HA
struct exb_variable* exb_get_model_variable(struct exb_connection* connection, const char* model_sign, int32_t variable_id, int pull_timeout) {
    exb_variable* variable = new exb_variable;
    variable->model_sign = model_sign;
    variable->manager = connection->manager.get();
    ps::Status status = connection->manager->find_model_variable(
        model_sign, variable_id, variable->model, variable->handle, pull_timeout);
    if (!status.ok()) {
        exb_thread_local_error_string() = status.ToString();
        delete variable;
        return nullptr;
    }
    return variable;
}

void exb_release_model_variable(struct exb_variable* variable) {
    if (variable->manager) {
        delete variable;
    } else {
        SLOG(WARNING) << "not model variable";
    }
}


void exb_barrier(struct exb_context* context, const char* name, exb_string* value) {
    if (value) {
        std::vector<char> x(value->data, value->data + sizeof(*value));
        context->entity->boardcast(name, x);
        memcpy(value, x.data(), sizeof(*value));
    } else {
        context->entity->barrier(name);
    }
}


struct exb_channel* exb_channel_create() {
    return new exb_channel;
}

void exb_channel_delete(struct exb_channel* channel) {
    delete channel;
}

void exb_channel_close(struct exb_channel* channel) {
    channel->entity.terminate();
}

void exb_channel_write(struct exb_channel* channel, void* p) {
    channel->entity.send(std::move(p));
}

bool exb_channel_read(struct exb_channel* channel, void** p) {
    return channel->entity.recv(*p, -1);
}

void exb_mutex_lock(struct exb_mutex* mutex) {
    reinterpret_cast<core::RWSpinLock*>(mutex)->lock();
};
void exb_mutex_unlock(struct exb_mutex* mutex) {
    reinterpret_cast<core::RWSpinLock*>(mutex)->unlock();
}
void exb_mutex_lock_shared(struct exb_mutex* mutex) {
    reinterpret_cast<core::RWSpinLock*>(mutex)->lock_shared();
}
void exb_mutex_unlock_shared(struct exb_mutex* mutex) {
    reinterpret_cast<core::RWSpinLock*>(mutex)->unlock_shared();
}
void exb_mutex_upgrade(struct exb_mutex* mutex) {
    reinterpret_cast<core::RWSpinLock*>(mutex)->upgrade();
}
void exb_mutex_downgrade(struct exb_mutex* mutex) {
    reinterpret_cast<core::RWSpinLock*>(mutex)->downgrade();
}

void* exb_malloc(size_t size) {
    return pico_malloc(size);
}

void exb_free(void* p) {
    return pico_free(p);
}

void exb_info(const char* message) {
    SLOG(INFO) << message;
}
void exb_warning(const char* message) {
    SLOG(WARNING) << message;
}
void exb_fatal(const char* message) {
    SLOG(FATAL) << message;
}

}

#ifndef OPENEMBEDDING_VERSION
#define OPENEMBEDDING_VERSION "unknown"
static_assert(false, "unknown OPENEMBEDDING_VERSION")
#endif

const char* exb_version() {
    return OPENEMBEDDING_VERSION;
}
