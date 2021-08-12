#include "EmbeddingPushOperator.h"

#include <pico-ps/common/EasyHashMap.h>
#include <pico-ps/operator/PushOperator.h>
#include "EmbeddingStorage.h"
#include "EmbeddingPullOperator.h"
#include "RpcView.h"

namespace paradigm4 {
namespace pico {
namespace embedding {


void EmbeddingPushRequestData::init(size_t shard_num) {
    if (shard_num != shards.size()) {
        shards.clear();
    }
    shards.resize(shard_num);
    for (ShardData& shard: shards) {
        shard.indices_base = 0;
        shard.gradients_base = 0;
        shard.num_indices.clear();
        shard.indices.clear();
        shard.gradients.clear();
        shard.counts.clear();
    }
}

template<class T>
void EmbeddingPushRequestData::operator()(TypeCase<T>, EmbeddingPushItems& items) {
    offsets.clear();
    size_t shard_num = shards.size();
    for (ShardData& shard: shards) {
        shard.indices_base = shard.indices.size();
        shard.gradients_base = shard.gradients.size();
    }
    size_t line_size = items.meta.line_size();
    const char* gradients = items.gradients;
    for (size_t i = 0; i < items.n; ++i) {
        uint64_t index = items.indices[i];
        auto& shard = shards[index % shard_num];
        if (offsets.count(index)) {
            size_t offset = offsets.at(index);
            T* sum = reinterpret_cast<T*>(shard.gradients.data() +
                    shard.gradients_base + offset * line_size);
            const T* grad = reinterpret_cast<const T*>(gradients);
            for (size_t j = 0; j < items.meta.embedding_dim; ++j) {
                sum[j] += grad[j];
            }
            ++shard.counts[shard.indices_base + offset];
        } else {
            offsets.force_emplace(index, shard.indices.size() - shard.indices_base);
            shard.indices.push_back(index / shard_num);
            shard.gradients.insert(shard.gradients.end(), gradients, gradients + line_size);
            shard.counts.push_back(1);
        }
        gradients += line_size;
    }
    for (ShardData& shard: shards) {
        shard.num_indices.push_back(shard.indices.size());
    }
}


ps::Status EmbeddingPushOperator::generate_request(core::vector<EmbeddingPushItems>& block_items,
        ps::RuntimeInfo& rt, EmbeddingPushRequestData& data, std::vector<ps::PSRequest>& reqs) {
    VTIMER(1, embedding_push, generate_push_request, ms);
    int32_t global_shard_num = rt.global_shard_num();
    data.init(global_shard_num);
    if (global_shard_num <= 0) {
        return ps::Status::NoReplica("no shard");
    }
    
    for (EmbeddingPushItems& items: block_items) {
        for (size_t i = 0; i < items.n; ++i) {
            uint64_t index = items.indices[i];
            if (index >= items.meta.vocabulary_size) {
                return ps::Status::Error("embedding index out of range");
            }
        }
        items.meta.datatype.invoke(data, items);
    }

    for (auto& p: rt.nodes()) {
        int32_t shard_num = p.second.size();
        int32_t block_num = block_items.size();
        reqs.emplace_back(p.first, 8 + shard_num * block_num * 12);
        auto& req = reqs.back();
        req << shard_num << block_num;
        for (int32_t shard_id: p.second) {
            auto& shard = data.shards[shard_id];
            req << shard_id;
            serialize(req.lazy(), _compress_info, RpcView<uint64_t>(shard.indices));
            serialize(req.lazy(), _compress_info, RpcView<char>(shard.gradients));
            serialize(req.lazy(), _compress_info, RpcView<uint64_t>(shard.counts));
            uint64_t offset = 0;
            for (int i = 0; i < block_num; ++i) {
                size_t num_indices = shard.num_indices[i];
                req << block_items[i].variable_id << block_items[i].meta << num_indices - offset;
                offset = num_indices;
            }
        }
    }
    return ps::Status();
}

void EmbeddingPushOperator::apply_request(const ps::PSMessageMeta& psmeta, ps::PSRequest& req, 
        const ps::TableDescriptor& table, core::Dealer* dealer) {
    VTIMER(1, embedding_push, apply_request, ms);

    auto& st = *(static_cast<EmbeddingStorage*>(table.storage.get()));
    core::shared_lock_guard<EmbeddingStorage> l(st);

    int32_t shard_num, block_num;
    req >> shard_num >> block_num;
    
    core::vector<data_block_t> holders;
    while (shard_num--) {
        int32_t shard_id;
        req >> shard_id;
        RpcView<uint64_t> view_indices;
        RpcView<char> view_gradients;
        RpcView<uint64_t> view_counts;
        deserialize(req.lazy(), _compress_info, view_indices);
        deserialize(req.lazy(), _compress_info, view_gradients);
        deserialize(req.lazy(), _compress_info, view_counts);
        uint64_t* indices = view_indices.data;
        char* gradients = view_gradients.data;
        uint64_t* counts = view_counts.data;

        auto& shard = *(st.get(shard_id));
        core::shared_lock_guard<ps::ShardData> sl(shard);
        EmbeddingShard& ht = *boost::any_cast<EmbeddingShard>(&shard.data);;
        for (int i = 0; i < block_num; ++i) {
            uint32_t variable_id;
            EmbeddingVariableMeta meta;
            uint64_t num_indices;
            req >> variable_id >> meta >> num_indices;
            SCHECK(ht.contains(variable_id) && meta == ht.meta(variable_id));
            ht[variable_id].push_gradients(indices, num_indices, gradients, counts);
            indices += num_indices;
            gradients += num_indices * meta.line_size();
            counts += num_indices;
        }
        holders.push_back(std::move(view_indices.holder));
        holders.push_back(std::move(view_gradients.holder));
        holders.push_back(std::move(view_counts.holder));
    }
    // 将request的lazy archive都copy出来才能send_response
    ps::PSResponse resp(req);
    resp << psmeta;
    dealer->send_response(std::move(resp.rpc_response()));
    core::lock_guard<core::RWSpinLock> pl(st.mutex);
    for (data_block_t& holder: holders) {
        st.holders.push_back(std::move(holder));
    }
}

ps::Status EmbeddingPushOperator::apply_response(ps::PSResponse& resp, EmbeddingPushRequestData&, void* result) {
    SCHECK(result == nullptr) << "return no result!";
    SCHECK(resp.archive().is_exhausted());
    return ps::Status();
}

}
}
}
