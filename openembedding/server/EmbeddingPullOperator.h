#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_PULL_OPERATOR_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_PULL_OPERATOR_H

#include <pico-ps/common/EasyHashMap.h>
#include <pico-ps/operator/PullOperator.h>
#include <pico-ps/operator/UDFOperator.h>
#include "EmbeddingStorage.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

struct EmbeddingPullItems {
    uint32_t variable_id = 0;
    EmbeddingVariableMeta meta;

    const uint64_t* indices = nullptr;
    uint64_t n = 0;

    int64_t batch_id = 0;
    
};

struct EmbeddingPullResults {
    const uint64_t* indices = nullptr;
    uint64_t n = 0;
    
    char* weights = nullptr;
    bool should_persist = false;
};

struct EmbeddingPullRequestData {
    struct ShardData {
        size_t cursor = 0;
        core::vector<uint64_t> num_indices; // prefix count
        ps::RpcVector<uint64_t> indices;
        BinaryArchive weights;
    };
    
    EmbeddingPullRequestData() {}
    
    void init(size_t shard_num, size_t block_num);

    size_t waiting_reqs = 0;
    core::vector<EasyHashMap<uint64_t, size_t>> block_offsets;
    core::vector<EmbeddingPullItems> block_items;
    std::unordered_map<int, core::vector<int32_t>> node_shards;
    core::vector<ShardData> shards;
};

class EmbeddingPullOperator: public ps::UDFOperator<core::vector<EmbeddingPullItems>, EmbeddingPullRequestData> {
public:
    EmbeddingPullOperator(const Configure& config):
          ps::UDFOperator<core::vector<EmbeddingPullItems>, EmbeddingPullRequestData>(config) {
        initialize_compress_info(config, "EmbeddingPullOperator", _compress_info);
        _algo = ps::initialize_shard_pick_algo(config);
        if (config.has("read_only")) {
            _read_only = config["read_only"].as<bool>();
        }
    }

    ~EmbeddingPullOperator() override {}

    EmbeddingPullOperator(EmbeddingPullOperator&&) = default;
    EmbeddingPullOperator& operator=(EmbeddingPullOperator&&) = default;

    bool read_only() override { return _read_only; }

    ps::Status generate_request(core::vector<EmbeddingPullItems>& block_items, 
          ps::RuntimeInfo& rt, EmbeddingPullRequestData& data, std::vector<ps::PSRequest>& reqs)override;

    void apply_request(const ps::PSMessageMeta& psmeta, ps::PSRequest& req, 
          const ps::TableDescriptor& table, core::Dealer* dealer) override;

    /// TODO: check context version 
    void apply_request_pull(const ps::PSMessageMeta& psmeta, ps::PSRequest& req, 
          const ps::TableDescriptor& table, core::Dealer* dealer);

    ps::Status apply_response(ps::PSResponse& resp, EmbeddingPullRequestData& data, void* result) override;

protected:
    bool _read_only = false;
    ps::CompressInfo _compress_info;
    ps::PickAlgo _algo;
};


}
}
}

#endif