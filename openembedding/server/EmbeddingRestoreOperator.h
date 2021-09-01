#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_RESTORE_OPERATOR_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_RESTORE_OPERATOR_H

#include <pico-ps/operator/RestoreOperator.h>
#include "EmbeddingStorage.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

// need restore initializer for default value
class EmbeddingRestoreOperator: public ps::RestoreOperator {
    typedef uint64_t key_type;
public:
    EmbeddingRestoreOperator(const core::Configure& config) : ps::RestoreOperator(config) {
        initialize_compress_info(config, "EmbeddingRestoreOperator", _compress_info);
    }

    ~EmbeddingRestoreOperator() override {}
    EmbeddingRestoreOperator(EmbeddingRestoreOperator&&) = default;
    EmbeddingRestoreOperator& operator=(EmbeddingRestoreOperator&&) = default;
    
    void generate_coordinated_restore_request(
          ps::CoordinatedRestoreRequestItem* req_item, std::vector<ps::PSRequest>& req)override;

    virtual void apply_coordinated_restore_request(
          ps::PSRequest& req, ps::Storage* storage, ps::PSResponse& resp)override;

    virtual void apply_coordinated_restore_response(ps::PSResponse& resp, ps::Storage* storage, ps::CoordinatedRestoreResponseItem* resp_item);

    virtual void restore(const core::URIConfig& uri, ps::RuntimeInfo& rt, ps::Storage* storage);

protected:
    ps::CompressInfo _compress_info;
};

typedef ps::ShardStorageOperator<EmbeddingStorage, EmbeddingRestoreOperator> EmbeddingStorageOperator;


}
}
}

#endif