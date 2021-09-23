#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_LOAD_OPERATOR_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_LOAD_OPERATOR_H

#include <pico-ps/operator/LoadOperator.h>
#include "EmbeddingInitOperator.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class EmbeddingLoadOperator: public ps::LoadOperator {
    typedef uint64_t key_type;
public:
    EmbeddingLoadOperator(const Configure& config): ps::LoadOperator(config), _push_op(config) {}

    virtual ~EmbeddingLoadOperator() {}

    EmbeddingLoadOperator(EmbeddingLoadOperator&&) = default;
    EmbeddingLoadOperator& operator=(EmbeddingLoadOperator&&) = default;


    void apply_load_response(ps::PSResponse& resp) override;

    void restore(const URIConfig&, ps::RuntimeInfo&, ps::Storage*) override;

    void create_stream(const URIConfig& uri, std::shared_ptr<void>& stream) override;

    size_t generate_push_items(std::shared_ptr<void>& stream_in,
          core::vector<std::unique_ptr<ps::PushItems>>& push_items) override;

    ps::PushOperator* push_operator() override {
        return &_push_op;
    }


protected:
    EmbeddingInitOperator _push_op;
};


}
}
}

#endif