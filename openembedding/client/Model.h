#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_MODEL_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_MODEL_H

#include "Meta.h"
#include "Connection.h"
#include "EmbeddingVariableHandle.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class Model {
public:
    Model(Connection* connection): _conn(connection) {}

    const ModelMeta& model_meta() {
        return _model_meta;
    }

    void set_model_status(ps::ModelStatus model_status);

    ps::Status test_status(const ps::Status& status);

    ps::Status update_model_meta(const ModelMeta& model_meta);

    ps::Status add_storage(int32_t storage_id, std::string storage_name);

    ps::Status add_variable(const ModelVariableMeta& variable);
    
    ps::Status access_storage(int32_t storage_id, EmbeddingStorageHandler*& storage)const;

    ps::Status access_variable(uint32_t variable_id, EmbeddingVariableHandle& handle)const;

    ps::Status dump_model(core::URIConfig uri, std::string model_sign, size_t num_files)const;

    ps::Status load_model(core::URIConfig uri);

    ps::Status load_model();

    ps::Status create_model(core::URIConfig uri);

    ps::Status create_model_storages(int32_t replica_num, int32_t shard_num = -1);

    void delete_model_storages();

    static ps::Status read_meta_file(const core::URIConfig& uri, ModelOfflineMeta& model_meta);

private:
    Connection* _conn = nullptr;
    ModelMeta _model_meta;
    // The file name of storage is the ordered rank of the storage_id in this model.
    std::unordered_map<int32_t, std::unique_ptr<EmbeddingStorageHandler>> _storages;
};

}
}
}

#endif
