#include "EmbeddingStoreOperator.h"

#include <pico-ps/common/EasyHashMap.h>
#include <pico-ps/operator/PushOperator.h>
#include "EmbeddingStorage.h"
#include "EmbeddingPullOperator.h"
#include "RpcView.h"
#include "PersistManager.h"

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
    auto& rt = *table.runtime_info;
    auto& st = *(static_cast<EmbeddingStorage*>(table.storage.get()));
    core::shared_lock_guard<EmbeddingStorage> l(st);
    VariableAsyncTask::wait(st.async_tasks);

    for (int32_t shard_id: rt.local_shards()) {
        auto& shard = *(st.get(shard_id));
        shard.lock(); // TODO: use guard
    }

    if (_early_return) {
        dealer->send_response(std::move(resp.rpc_response()));
    }
    
    for (int32_t shard_id: rt.local_shards()) {
        auto& shard = *(st.get(shard_id));
        EmbeddingShard& ht = *boost::any_cast<EmbeddingShard>(&shard.data);
        for (uint32_t variable_id: ht.variable_ids()) {
            ht[variable_id].update_weights();
        }
        shard.unlock();
    }

    if (!_early_return) {
        dealer->send_response(std::move(resp.rpc_response()));
    }
    core::vector<PendingRequest> reqs;
    {
        core::lock_guard<core::RWSpinLock> pl(st.pending_mutex);
        // Store and push should not happen at the same time, otherwise holders.clear() will cause error.
        st.holders.clear();
        
        if (!st.pending.empty()) {
            reqs = std::move(st.pending.front());
            st.pending.pop_front();
        }
        st.batch_id += 1;
    }
    // Start processing the pull requests of batch_id + 1.
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
