#include "EmbeddingInitOperator.h"
#include <pico-ps/common/EasyHashMap.h>
#include <pico-ps/operator/PushOperator.h>
#include "EmbeddingStorage.h"

namespace paradigm4 {
namespace pico {
namespace embedding {


struct EmbeddingInitRequestData: ps::PushRequestData {
    struct ShardData {
        ps::RpcVector<uint64_t> indices;
        ps::RpcVector<char> weights;
        ps::RpcVector<char> states;
    };
    
    EmbeddingInitRequestData(): offsets(-1) {}

    void init(size_t shard_num, const EmbeddingInitItems& items) {
        this->items = items;
        offsets.clear();
        shards.resize(shard_num);
        for (ShardData& shard : shards) {
            shard.indices.clear();
            shard.weights.clear();
            shard.states.clear();
        }
        if (items.indices) {
            int32_t global_shard_num = shards.size();
            size_t line_size = items.meta.line_size();
            for (size_t i = 0; i < items.n; ++i) {
                uint64_t index = items.indices[i];
                ShardData& shard = shards[index % global_shard_num];
                shard.indices.push_back(index / global_shard_num);
                shard.weights.insert(shard.weights.end(),
                        items.weights + i * line_size,
                        items.weights + (i + 1) * line_size);
                if (items.state_line_size) {
                    shard.states.insert(shard.states.end(),
                        items.states + i * items.state_line_size,
                        items.states + (i + 1) * items.state_line_size);
                }
            }
        }
    }

    EmbeddingInitItems items;
    EasyHashMap<uint64_t, size_t> offsets;
    core::vector<ShardData> shards;
};

void EmbeddingInitOperator::generate_request_data(core::vector<std::unique_ptr<ps::PushItems>>& push_items,
        ps::RuntimeInfo& rt,
        std::unique_ptr<ps::PushRequestData>& push_request_data) {
    VTIMER(1, embedding_push, generate_request_data, ms);

    if (push_request_data == nullptr) {
        push_request_data = std::make_unique<EmbeddingInitRequestData>();
    }
    auto& request_data = *static_cast<EmbeddingInitRequestData*>(push_request_data.get());
    
    SCHECK(push_items.size() == 1);
    auto& items = *static_cast<EmbeddingInitItems*>(push_items[0].get());
    request_data.init(rt.global_shard_num(), items);
}

void EmbeddingInitOperator::generate_push_request(
        std::vector<ps::PushRequestData*>& push_request_data,
        ps::RuntimeInfo& rt,
        std::vector<ps::PSRequest>& reqs) {
    VTIMER(1, embedding_push, generate_push_request, ms);
    SCHECK(rt.global_shard_num() > 0);
    
    // Must be sent to all servers, because it is used to initialize.
    for (auto& p: rt.nodes()) {
        int32_t shard_num = p.second.size();
        int32_t block_num = push_request_data.size();
        reqs.emplace_back(p.first, 8 + shard_num * block_num * 12);
        auto& req = reqs.back();
        req << shard_num << block_num;
        for (int32_t shard_id: p.second) {
            req << shard_id;
            for (auto data: push_request_data) {
                auto& request_data = *static_cast<EmbeddingInitRequestData*>(data);
                req << request_data.items.variable_id << request_data.items.meta;
                req << request_data.items.clear_weights << request_data.items.variable_config;
                auto& shard_data = request_data.shards[shard_id];
                uint64_t shard_item_num = shard_data.indices.size();
                req << shard_item_num << request_data.items.state_line_size;
                if (shard_item_num != 0) {
                    BinaryArchive indices = vector_rpc_view(shard_data.indices);
                    BinaryArchive weights = vector_rpc_view(shard_data.weights);
                    ps_serialize(req.lazy(), _compress_info, std::move(indices));
                    ps_serialize(req.lazy(), _compress_info, std::move(weights));
                    if (request_data.items.state_line_size) {
                        BinaryArchive states = vector_rpc_view(shard_data.states);
                        ps_serialize(req.lazy(), _compress_info, std::move(states));
                    }
                }
            }
        }
    }
}

void EmbeddingInitOperator::generate_store_request(ps::RuntimeInfo& rt,
        std::vector<ps::PSRequest>& reqs) {
    VTIMER(1, embedding_store, generate_store_request, ms);
    for (const auto& p: rt.nodes()) {
        reqs.emplace_back(p.first);
    }
}

void EmbeddingInitOperator::apply_async_push_request(ps::RuntimeInfo& rt,
        ps::PSRequest& req,
        ps::Storage* storage,
        ps::Storage*,
        ps::PSResponse& resp) {
    VTIMER(1, embedding_push, apply_async_push_request, ms);

    int32_t shard_num, block_num;
    req >> shard_num >> block_num;

    auto& st = *static_cast<EmbeddingStorage*>(storage);
    core::shared_lock_guard<EmbeddingStorage> l(st);
    while (shard_num--) {
        int32_t shard_id;
        req >> shard_id;

        SCHECK(rt.local_shards().count(shard_id) != 0) 
                << "Bad Request: invalid shard_id = " << shard_id;

        auto& shard = *(st.get(shard_id));
        core::lock_guard<ps::ShardData> sl(shard);
        auto& ht = *boost::any_cast<EmbeddingShard>(&shard.data);
        for (int i = 0; i < block_num; ++i) {
            uint32_t variable_id;
            EmbeddingVariableMeta meta;
            req >> variable_id >> meta;
            bool clear_weights;
            std::string config_str;
            req >> clear_weights >> config_str;

            EmbeddingVariableBase& variable = ht.get(variable_id, meta);
            if (clear_weights) {
                variable.clear_weights();
            }
            if (!config_str.empty()) {
                core::Configure variable_config;
                variable_config.load(config_str);
                if (meta.use_hash_table()) {
                    variable_config.node()["table"] = "hash";
                }
                variable.load_config(variable_config);
            }
            
            uint64_t shard_item_num, state_line_size;
            req >> shard_item_num >> state_line_size;

            if (shard_item_num == 0) {
                continue;
            }
            BinaryArchive indices, weights, states;
            ps_deserialize(req.lazy(), _compress_info, indices);
            ps_deserialize(req.lazy(), _compress_info, weights);
            if (state_line_size) {
                ps_deserialize(req.lazy(), _compress_info, states);
                SCHECK(state_line_size == variable.state_line_size());
            }
            variable.set_weights(
                  reinterpret_cast<uint64_t*>(indices.cursor()),
                  shard_item_num, weights.cursor(), states.cursor());
        }
    }
    resp = ps::PSResponse(req);
}

void EmbeddingInitOperator::apply_response(ps::PSResponse& resp) {
    SCHECK(resp.archive().is_exhausted());
}

}
}
}
