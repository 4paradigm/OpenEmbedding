
#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_EXB_CAPI_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_EXB_CAPI_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

struct exb_connection;

struct exb_master;
struct exb_server;
struct exb_configure;
struct exb_context;
struct exb_storage;
struct exb_variable;
struct exb_optimizer;
struct exb_initializer;
struct exb_pull_waiter;
struct exb_waiter;
struct exb_channel;
struct exb_mutex {
    int64_t data[16];
};
struct exb_string {
    char data[128];
};

struct exb_connection* exb_serving();
// TCP configuration should be consistent for all connections.
// There may be unknown problems with multiple connections at the same time.
// wait_server_num = -1 means to start ps in each worker process.
struct exb_connection* exb_connect(const char* yaml_config,
      const char* master_endpoint, const char* rpc_bind_ip = "");

// thread local
const char* exb_last_error();

int exb_last_wait_time_ms();

int exb_running_server_count(struct exb_connection*);

void exb_disconnect(struct exb_connection*);

struct exb_master* exb_master_start(const char* bind_ip = "");

void exb_master_endpoint(struct exb_master*, exb_string* value);

void exb_master_join(struct exb_master*); // destroy

struct exb_server* exb_server_start(struct exb_connection*);

void exb_server_exit(struct exb_server*);

void exb_server_join(struct exb_server*); // destroy

struct exb_context* exb_context_initialize(struct exb_connection*,
      int32_t worker_num, int32_t wait_server_num = -1);

void exb_context_finalize(struct exb_context*);

int exb_worker_rank(struct exb_context*);

struct exb_storage* exb_create_storage(struct exb_context*, int32_t shard_num = -1);

void exb_delete_storage(struct exb_storage*);

struct exb_variable* exb_create_variable(struct exb_storage*,
      uint64_t vocabulary_size, size_t embedding_dim, const char* dtype = "float32");

int32_t exb_storage_id(struct exb_storage*);

uint32_t exb_variable_id(struct exb_variable*);

void exb_set_initializer(struct exb_variable*, struct exb_initializer*);

void exb_set_optimizer(struct exb_variable*, struct exb_optimizer*);

size_t exb_unique_indices(const uint64_t* indices, size_t n, size_t* unique);

struct exb_pull_waiter* exb_pull_weights(const struct exb_variable*,
      const uint64_t* indices, size_t n, int64_t batch_id);

struct exb_waiter* exb_push_gradients(struct exb_variable*,
      const uint64_t* indices, size_t n, const void* gradients);

struct exb_waiter* exb_update_weights(struct exb_storage*);

bool exb_pull_wait(struct exb_pull_waiter*, const uint64_t* indices, size_t n, void* weights);

bool exb_wait(struct exb_waiter*);

struct exb_optimizer* exb_create_optimizer(const char* category);

void exb_set_optimizer_property(struct exb_optimizer*, const char* key, const char* value);

struct exb_initializer* exb_create_initializer(const char* category);

void exb_set_initializer_property(struct exb_initializer*, const char* key, const char* value);

const char* exb_version();

void exb_dump_model_include_optimizer(struct exb_context*, const char* path, const char* model_sign);

void exb_dump_model(struct exb_context*, const char* path, const char* model_sign);

void exb_load_model(struct exb_context*, const char* path);

void exb_create_model(struct exb_connection*, const char* path, int32_t replica_num, int32_t shard_num = -1);

struct exb_variable* exb_get_model_variable(struct exb_connection*, const char* model_sign, int32_t variable_id, int pull_timeout = -1);

void exb_release_model_variable(struct exb_variable*);

void exb_barrier(struct exb_context*, const char* name, exb_string* value = NULL);

void exb_start_monitor(struct exb_context*);

struct exb_channel* exb_channel_create();
void exb_channel_delete(struct exb_channel*);
void exb_channel_close(struct exb_channel*);
void exb_channel_write(struct exb_channel*, void*);
bool exb_channel_read(struct exb_channel*, void**);

void exb_mutex_lock(struct exb_mutex*);
void exb_mutex_unlock(struct exb_mutex*);
void exb_mutex_lock_shared(struct exb_mutex*);
void exb_mutex_unlock_shared(struct exb_mutex*);
void exb_mutex_upgrade(struct exb_mutex*);
void exb_mutex_downgrade(struct exb_mutex*);

void* exb_malloc(size_t size);
void exb_free(void* p);

void exb_info(const char* message);
void exb_warning(const char* message);
void exb_fatal(const char* message);

bool exb_should_persist_model(struct exb_context*);

void exb_persist_model(struct exb_context*, const char* path, const char* model_sign, size_t persist_pending_window);
void exb_restore_model(struct exb_context*, const char* path);

#ifdef __cplusplus
}
#endif

#endif
