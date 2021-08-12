#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_PUSH_OPERATOR_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_PUSH_OPERATOR_H

#include <pico-ps/common/EasyHashMap.h>
#include <pico-ps/operator/PushOperator.h>
#include "EmbeddingStorage.h"
#include "EmbeddingPullOperator.h"
#include "RpcView.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

// key <--> index
// value <--> gradients
class EmbeddingPushItems {
public:
    uint32_t variable_id = -1;
    EmbeddingVariableMeta meta;

    const uint64_t* indices = nullptr;
    uint64_t n = 0;
    const char* gradients = nullptr;
};

struct EmbeddingPushRequestData {
    struct ShardData {
        size_t indices_base = 0;
        size_t gradients_base = 0;
        core::vector<uint64_t> num_indices; // prefix count
        ps::RpcVector<uint64_t> indices;
        ps::RpcVector<char> gradients;
        ps::RpcVector<uint64_t> counts;
    };
    
    EmbeddingPushRequestData(): offsets(-1) {}

    void init(size_t shard_num);

    template<class T>
    void operator()(TypeCase<T>, EmbeddingPushItems& items);

    EasyHashMap<uint64_t, size_t> offsets;
    core::vector<ShardData> shards;
};


class EmbeddingPushOperator : public ps::UDFOperator<core::vector<EmbeddingPushItems>, EmbeddingPushRequestData> {
public:
    EmbeddingPushOperator(const Configure& config):
          ps::UDFOperator<core::vector<EmbeddingPushItems>, EmbeddingPushRequestData>(config) {
        initialize_compress_info(config, "EmbeddingPushOperator", _compress_info);
    }

    virtual ~EmbeddingPushOperator() {}

    EmbeddingPushOperator(EmbeddingPushOperator&&) = default;
    EmbeddingPushOperator& operator=(EmbeddingPushOperator&&) = default;

    bool read_only() override { return false; }

    ps::Status generate_request(core::vector<EmbeddingPushItems>& block_items,
          ps::RuntimeInfo& rt, EmbeddingPushRequestData& data, std::vector<ps::PSRequest>& reqs) override;

    void apply_request(const ps::PSMessageMeta& psmeta, ps::PSRequest& req, 
          const ps::TableDescriptor& table, core::Dealer* dealer) override;


    ps::Status apply_response(ps::PSResponse& resp, EmbeddingPushRequestData&, void* result) override;

protected:

    ps::CompressInfo _compress_info;
};


}
}
}

#endif