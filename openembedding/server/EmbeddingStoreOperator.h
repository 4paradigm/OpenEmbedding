#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_STORE_OPERATOR_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_STORE_OPERATOR_H

#include <pico-ps/common/EasyHashMap.h>
#include <pico-ps/operator/PushOperator.h>
#include "EmbeddingStorage.h"
#include "EmbeddingPullOperator.h"
#include "RpcView.h"

namespace paradigm4 {
namespace pico {
namespace embedding {


class EmbeddingStoreOperator : public ps::UDFOperator<int, int> {
public:
    EmbeddingStoreOperator(const Configure& config):
          ps::UDFOperator<int, int>(config), _pull(config) {
        if (config.has("update_early_return")) {
            _early_return = config["update_early_return"].as<bool>();
        }
    }

    virtual ~EmbeddingStoreOperator() {}

    EmbeddingStoreOperator(EmbeddingStoreOperator&&) = default;
    EmbeddingStoreOperator& operator=(EmbeddingStoreOperator&&) = default;

    bool read_only() override { return false; }

    ps::Status generate_request(int&,
          ps::RuntimeInfo& rt, int&, std::vector<ps::PSRequest>& reqs) override;

    void apply_request(const ps::PSMessageMeta& psmeta, ps::PSRequest& req, 
          const ps::TableDescriptor& table, core::Dealer* dealer) override;

    ps::Status apply_response(ps::PSResponse& resp, int&, void* result) override;

protected:
    EmbeddingPullOperator _pull;
    bool _early_return = true;
};


}
}
}

#endif