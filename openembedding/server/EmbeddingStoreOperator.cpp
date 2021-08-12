#include "EmbeddingStoreOperator.h"

#include <pico-ps/common/EasyHashMap.h>
#include <pico-ps/operator/PushOperator.h>
#include "EmbeddingStorage.h"
#include "EmbeddingPullOperator.h"
#include "RpcView.h"

namespace paradigm4 {
namespace pico {
namespace embedding {



ps::Status EmbeddingStoreOperator::generate_request(int&,
        ps::RuntimeInfo& rt, int&, std::vector<ps::PSRequest>& reqs) {
    VTIMER(1, embedding_push, generate_push_request, ms);
    for (auto& node: rt.nodes()) {
        reqs.emplace_back(node.first);
    }
    return ps::Status();
}

void EmbeddingStoreOperator::apply_request(const ps::PSMessageMeta& psmeta, ps::PSRequest& req, 
        const ps::TableDescriptor& table, core::Dealer* dealer) {
    VTIMER(1, embedding_update, apply_request, ms);
    ps::PSResponse resp(req);
    resp << psmeta;
    if (_early_return) {
        dealer->send_response(std::move(resp.rpc_response()));
    }
    
    auto& rt = *table.runtime_info;
    auto& st = *(static_cast<EmbeddingStorage*>(table.storage.get()));
    core::shared_lock_guard<EmbeddingStorage> l(st);
    for (int32_t shard_id: rt.local_shards()) {
        auto& shard = *(st.get(shard_id));
        core::lock_guard<ps::ShardData> sl(shard);
        EmbeddingShard& ht = *boost::any_cast<EmbeddingShard>(&shard.data);
        for (uint32_t variable_id: ht.variable_ids()) {
            ht[variable_id].update_weights();
        }
    }

    if (!_early_return) {
        dealer->send_response(std::move(resp.rpc_response()));
    }
    core::vector<PendingRequest> reqs;
    {
        core::lock_guard<core::RWSpinLock> pl(st.mutex);
        // store和push不应同时发生，否则holders释放错误
        st.holders.clear();
        
        if (!st.pending.empty()) {
            reqs = std::move(st.pending.front());
            st.pending.pop_front();
        }
        st.version += 1;
    }
    // 开始处理version + 1的pull
    for (PendingRequest& pend: reqs) {
        ps::Status status;
        if (status.ok()) {
            _pull.apply_request_pull(pend.psmeta, pend.request, table, dealer);
        } else {
            ps::PSResponse resp(pend.request);
            resp.rpc_response().set_error_code(RpcErrorCodeType::ELOGICERROR);
            resp << status << pend.psmeta;
            dealer->send_response(std::move(resp.rpc_response()));
        }
    }
}

ps::Status EmbeddingStoreOperator::apply_response(ps::PSResponse& resp, int&, void* result) {
    SCHECK(result == nullptr) << "return no result!";
    SCHECK(resp.archive().is_exhausted());
    return ps::Status();
}



}
}
}
