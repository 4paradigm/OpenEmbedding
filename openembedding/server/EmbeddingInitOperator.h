#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_INIT_OPERATOR_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_INIT_OPERATOR_H

#include <pico-ps/common/EasyHashMap.h>
#include <pico-ps/operator/PushOperator.h>
#include "Meta.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class EmbeddingInitItems: public ps::PushItems {
public:
    EmbeddingVariableMeta meta;
    uint32_t variable_id = -1;
    uint64_t n = 0; // indices for push 
                  // vocabulary_size for resize or create
    bool clear_weights = false;
    const uint64_t* indices = nullptr; // for push
    const char* weights = nullptr;
    uint64_t state_line_size = 0; // != 0 means pushing optimizer state
    std::string variable_config; // for create
};

// for init, load, update context
class EmbeddingInitOperator : public ps::PushOperator {
public:
    EmbeddingInitOperator(const Configure& config) : ps::PushOperator(config) {
        initialize_compress_info(config, "EmbeddingInitOperator", _compress_info);
    }

    ~EmbeddingInitOperator()override {}

    EmbeddingInitOperator(EmbeddingInitOperator&&) = default;
    EmbeddingInitOperator& operator=(EmbeddingInitOperator&&) = default;

    void generate_request_data(core::vector<std::unique_ptr<ps::PushItems>>& push_items,
          ps::RuntimeInfo& rt,
          std::unique_ptr<ps::PushRequestData>& push_request_data) override;

    void generate_push_request(
          std::vector<ps::PushRequestData*>& push_request_data,
          ps::RuntimeInfo& rt,
          std::vector<ps::PSRequest>& reqs) override;

    void generate_store_request(ps::RuntimeInfo& rt,
          std::vector<ps::PSRequest>& reqs) override;

    void apply_async_push_request(ps::RuntimeInfo& rt,
          ps::PSRequest& req,
          ps::Storage* storage,
          ps::Storage*,
          ps::PSResponse& resp) override;

    void apply_sync_push_request(ps::RuntimeInfo&,
          ps::PSRequest&,
          ps::Storage*,
          ps::PSResponse&) override {
        return;
    }

    void apply_store_request(ps::RuntimeInfo&,
          ps::PSRequest&,
          ps::Storage*,
          ps::Storage*,
          ps::Storage*,
          std::function<void(ps::PSResponse&&)>) override {  
        return;
    }

    void apply_response(ps::PSResponse& resp) override;

    std::unique_ptr<ps::Storage> create_delta_storage(ps::RuntimeInfo&) override {
        return nullptr;
    }

    std::unique_ptr<ps::Storage> create_incr_storage(ps::RuntimeInfo&) override {
        return nullptr;
    }

protected:
    ps::CompressInfo _compress_info;
};


}
}
}

#endif