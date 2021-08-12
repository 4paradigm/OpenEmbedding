#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_DUMP_OPERATOR_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_DUMP_OPERATOR_H

#include <pico-ps/operator/DumpOperator.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

class EmbeddingDumpOperator : public ps::ShardStorageDumpOperator {
public:
    
    EmbeddingDumpOperator(const core::Configure& conf) : ps::ShardStorageDumpOperator(conf) {}

    virtual ~EmbeddingDumpOperator() {}

    EmbeddingDumpOperator(EmbeddingDumpOperator&&) = default;
    EmbeddingDumpOperator& operator=(EmbeddingDumpOperator&&) = default;

    void apply_request(ps::RuntimeInfo& rt,
          ps::PSRequest& req,
          ps::Storage* storage,
          ps::PSResponse& resp_ret)override;

    std::unique_ptr<ps::ForEachResult> init_result_impl() {
        return nullptr;
    }

    void merge_result_impl(const ps::ForEachResult&, ps::ForEachResult&,
          const ps::CarriedItem&)override  {}
};

}
}
}

#endif