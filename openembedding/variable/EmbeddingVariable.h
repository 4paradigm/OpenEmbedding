#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_VARIABLE_H

#include <limits>
#include "Meta.h"
#include "VariableAsyncTask.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

struct EmbeddingVariableContext {
    int variable_id = 0;
    int shard_id = 0;
    int shard_num = 0;
    int storage_id = 0;
};

class EmbeddingVariableBase {
    using key_type = uint64_t;
public:
    static std::unique_ptr<EmbeddingVariableBase> create(DataType datatype, size_t embedding_dim);
    virtual ~EmbeddingVariableBase() {}
    virtual void set_variable_context(const EmbeddingVariableContext&) = 0;
    virtual void set_batch_id(int64_t batch_id) = 0;
    virtual void load_config(const core::Configure& config) = 0;
    virtual void dump_config(core::Configure& config) = 0;
    virtual bool dump_persist(core::Configure& config) = 0;
    virtual void clear_weights() = 0; // clear initializer，weights. optimizer not change. reset slots.
    virtual size_t server_block_num_items() = 0;
    virtual void get_weights(const key_type* indices, size_t n,
          char* weights, char* states = nullptr) = 0;  // thread safe
    virtual void set_weights(const key_type* indices, size_t n,
          const char* weights, const char* states = nullptr) = 0;
   
    virtual void pull_weights(const key_type* indices, size_t n,
          char* weights, VariableAsyncTask& async_task) = 0;  // thread safe
    virtual void push_gradients(const key_type* indices, size_t n,
          const char* gradients, const key_type* counts, VariableAsyncTask& async_task) = 0; // thread safe
    virtual void update_weights() = 0;
    virtual size_t state_line_size() = 0;

    virtual size_t num_indices() = 0;
    virtual int create_reader() = 0; // thread safe
    virtual size_t read_indices(int reader_id, key_type* indices, size_t n) = 0; // thread safe for unique reader_id
    virtual uint64_t get_reader_cursor(int reader_id) = 0; // // thread safe for unique reader_id
    virtual void delete_reader(int reader_id) = 0; // thread safe
};

}
}
}

#endif
