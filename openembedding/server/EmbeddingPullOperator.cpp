#include "EmbeddingPullOperator.h"

#include <pico-ps/common/EasyHashMap.h>
#include <pico-ps/operator/PullOperator.h>
#include <pico-ps/operator/UDFOperator.h>
#include "EmbeddingStorage.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

void EmbeddingPullRequestData::init(size_t shard_num, size_t block_num) {
    if (block_num != block_offsets.size()) {
        block_offsets.clear();
    }
    if (shard_num != shards.size()) {
        node_shards.clear();
        shards.clear();
    }
    waiting_reqs = 0;
    for (auto& pair: node_shards) {
        pair.second.clear();
    }
    shards.resize(shard_num);
    for (ShardData& shard: shards) {
        shard.cursor = 0;
        shard.num_indices.clear();
        shard.indices.clear();
        shard.weights.clear();
    }
    while (block_offsets.size() < block_num) {
        block_offsets.emplace_back(-1);
    }
    for (auto& offsets: block_offsets) {
        offsets.clear();
    }
}


ps::Status EmbeddingPullOperator::generate_request(core::vector<EmbeddingPullItems>& block_items, 
        ps::RuntimeInfo& rt, EmbeddingPullRequestData& data, std::vector<ps::PSRequest>& reqs) {  
    data.block_items = block_items;
    VTIMER(1, embedding_pull, generate_request, ms);
    if (block_items.empty()) {
        return ps::Status();
    }
    int32_t global_shard_num = rt.global_shard_num();
    data.init(global_shard_num, block_items.size());
    
    std::vector<int> selected_nodes = rt.pick_one_replica(_algo);
    for (int32_t shard_id = 0; shard_id < rt.global_shard_num(); ++shard_id) {
        int node_id = selected_nodes[shard_id];
        if (node_id == -1) {
            return ps::Status::NoReplica("");
        } else {
            data.node_shards[node_id].push_back(shard_id);
        }
    }
    
    for (size_t k = 0; k < block_items.size(); ++k) {
        auto& offsets = data.block_offsets[k];
        const EmbeddingPullItems& items = block_items[k];
        size_t line_size = items.meta.line_size();
        if (items.batch_id != block_items[0].batch_id) {
            return ps::Status::Error("request batch_id not same");
        }
        for (size_t i = 0; i < items.n; ++i) {
            uint64_t index = items.indices[i];
            if (index >= items.meta.vocabulary_size) {
                return ps::Status::Error("embedding index out of range");
            }
            if (!offsets.count(index)) {
                int32_t shard_id = index % global_shard_num;
                auto& shard = data.shards[shard_id];
                shard.indices.push_back(index / global_shard_num);
                offsets.force_emplace(index, shard.cursor);
                shard.cursor += line_size;
            }
        }
        for (auto& shard: data.shards) {
            shard.num_indices.push_back(shard.indices.size());
        }
    }

    for (auto& p: rt.nodes()) {
        int node_id = p.first;
        int32_t shard_num = data.node_shards[node_id].size();
        int32_t block_num = block_items.size();
        reqs.emplace_back(node_id, block_num * shard_num * 24);
        auto& req = reqs.back();
        req << block_items[0].batch_id << shard_num << block_num;
        bool hit_node = false;
        for (int32_t shard_id: data.node_shards[node_id]) {
            auto& shard = data.shards[shard_id];
            req << shard_id;
            ps::ps_serialize(req.lazy(), _compress_info, ps::vector_rpc_view(shard.indices));
            uint64_t offset = 0;
            for (int i = 0; i < block_num; ++i) {
                size_t num_indices = shard.num_indices[i];
                req << block_items[i].variable_id << block_items[i].meta << num_indices - offset;
                offset = num_indices;
            }
            
            if (!shard.indices.empty()) {
                hit_node = true;
            }
        }
        if (!hit_node) {
            reqs.pop_back();
        }
    }
    data.waiting_reqs = reqs.size();
    return ps::Status();
}

void EmbeddingPullOperator::apply_request(const ps::PSMessageMeta& psmeta, ps::PSRequest& req, 
        const ps::TableDescriptor& table, core::Dealer* dealer) {
    VTIMER(1, embedding_pull, apply_request, ms);
        
    auto& st = *(static_cast<EmbeddingStorage*>(table.storage.get()));
    core::shared_lock_guard<EmbeddingStorage> l(st);
    int64_t batch_id;
    req >> batch_id;
    {
        core::lock_guard<core::RWSpinLock> pl(st.pending_mutex);
        if (st.batch_id < batch_id) {
            size_t delta = batch_id - st.batch_id - 1;
            if (delta < 1024) {
                while (st.pending.size() <= delta) {
                    st.pending.emplace_back();
                }
                st.pending[delta].push_back({psmeta, std::move(req)});
            } else {
                ps::PSResponse resp(req);
                resp.rpc_response().set_error_code(core::RpcErrorCodeType::ELOGICERROR);
                resp << ps::Status::InvalidConfig("request too large version") << psmeta;
                dealer->send_response(std::move(resp.rpc_response()));
            }
            return;
        }
    }

    apply_request_pull(psmeta, req, table, dealer);
}


/// TODO: check context version 
void EmbeddingPullOperator::apply_request_pull(const ps::PSMessageMeta& psmeta, ps::PSRequest& req, 
        const ps::TableDescriptor& table, core::Dealer* dealer) {
    static thread_local size_t buffer_size = 0;
    auto& st = *(static_cast<EmbeddingStorage*>(table.storage.get()));
    
    bool error = false;
    BinaryArchive indices;
    int32_t shard_num, block_num;
    req >> shard_num >> block_num;
    ps::PSResponse resp(req, 4 + shard_num * 8);
    resp << shard_num;
    while (shard_num--) {
        int32_t shard_id;
        req >> shard_id;
        resp << shard_id;
        ps::ps_deserialize(req.lazy(), _compress_info, indices);
        core::BinaryArchive weights(true);
        weights.reserve(buffer_size);
        auto& shard = *(st.get(shard_id));
        core::shared_lock_guard<core::RWSpinLock> guard(shard._lock);
        EmbeddingShard& ht = *boost::any_cast<EmbeddingShard>(&shard.data);;
        for (int i = 0; i < block_num; ++i) {
            uint32_t variable_id;
            EmbeddingVariableMeta meta;
            uint64_t num_indices;
            req >> variable_id >> meta >> num_indices;
            weights.prepare_write(num_indices * meta.line_size());
            if (ht.contains(variable_id) && meta == ht.meta(variable_id)) {
                const uint64_t* pindices = reinterpret_cast<const uint64_t*>(indices.cursor());
                if (_read_only) {
                    ht[variable_id].get_weights(pindices, num_indices, weights.end());
                } else {
                    VariableAsyncTask async_task(variable_id,
                          st.async_tasks, st.shared_mutex(), shard._lock);
                    ht[variable_id].pull_weights(pindices, num_indices, weights.end(), async_task);
                    if (async_task) {
                        VariableAsyncTaskThreadPool::singleton().submit(std::move(async_task));
                    }
                }
            } else {
                error = true;
            }
            indices.advance_cursor(num_indices * sizeof(uint64_t));
            weights.advance_end(num_indices * meta.line_size());
        }
        buffer_size = std::max(buffer_size, weights.capacity());
        ps::ps_serialize(resp.lazy(), _compress_info, std::move(weights));
    }
    if (error) {
        resp.rpc_response().set_error_code(core::RpcErrorCodeType::ELOGICERROR);
        resp << ps::Status::InvalidConfig("client server variable meta not match");
    }
    resp << psmeta;
    dealer->send_response(std::move(resp.rpc_response()));
}

ps::Status EmbeddingPullOperator::apply_response(ps::PSResponse& resp, EmbeddingPullRequestData& data, void* result) {
    static thread_local core::Accumulator<core::SumAggregator<size_t>> acc_indices("pull_indices");
    static thread_local core::Accumulator<core::SumAggregator<size_t>> acc_unique("pull_unique");

    SCHECK(result) << "result buffer not set!";
    
    auto& block_items = *static_cast<core::vector<EmbeddingPullResults>*>(result);
    VTIMER(1, embedding_pull, apply_response, ms);
    int32_t shard_num;
    resp >> shard_num;
    while (shard_num--) {
        int32_t shard_id;
        resp >> shard_id;
        ps_deserialize(resp.lazy(), _compress_info, data.shards[shard_id].weights);
    }

    --data.waiting_reqs;
    int32_t global_shard_num = data.shards.size();
    if (data.waiting_reqs == 0) {
        for (size_t k = 0; k < block_items.size(); ++k) {
            auto& offsets = data.block_offsets[k];
            const EmbeddingPullResults& items = block_items[k];
            size_t line_size = data.block_items[k].meta.line_size();
            for (size_t i = 0; i < items.n; ++i) {     
                int32_t shard_id = items.indices[i] % global_shard_num;
                size_t offset = offsets.at(items.indices[i]);
                const char* p = data.shards[shard_id].weights.cursor() + offset;
                memcpy(items.weights + i * line_size, p, line_size);
            }

            if (core::pico_is_evaluate_performance()) {
                acc_indices.write(items.n);
                acc_unique.write(offsets.size());
            }

        }
    }
    return ps::Status();
}


}
}
}
