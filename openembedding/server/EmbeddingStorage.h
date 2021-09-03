#ifndef PARADIGM4_PICO_PS_EMBEDDING_EMBEDDING_STORAGE_H
#define PARADIGM4_PICO_PS_EMBEDDING_EMBEDDING_STORAGE_H

#include "Meta.h"
#include "EmbeddingVariable.h"
#include <pico-ps/operator/StorageOperator.h>

namespace paradigm4 {
namespace pico {
/*! \brief namespace of parameter server */
namespace embedding {

class EmbeddingShard {
public:
    bool insert_variable(uint32_t variable_id,
          std::unique_ptr<EmbeddingVariableBase> variable,
          const EmbeddingVariableMeta& meta) {
        if (variable_id >= _variables.size()) {
            _variables.resize(variable_id + 1);
            _metas.resize(variable_id + 1);
        }
        if (_variables[variable_id]) {
            return false;
        } 
        if (variable) {
            _metas[variable_id] = meta;
            _variables[variable_id] = std::move(variable);
            _variable_ids.push_back(variable_id);
            return true;
        }
        return false;  
    }

    bool contains(uint32_t variable_id)const  {
        return variable_id < _variables.size() && _variables[variable_id];
    }

    EmbeddingVariableBase& operator[](uint32_t variable_id) {
        SCHECK(contains(variable_id)) << variable_id;
        return *_variables[variable_id];
    }

    const std::vector<uint32_t>& variable_ids()const {
        return _variable_ids;
    }

    const EmbeddingVariableMeta& meta(uint32_t variable_id) {
        SCHECK(contains(variable_id)) << variable_id;
        return _metas[variable_id];
    }

    EmbeddingVariableBase& get(uint32_t variable_id, const EmbeddingVariableMeta& meta) {
        if (!contains(variable_id)) {
            auto pvar = EmbeddingVariableBase::create(meta.datatype, meta.embedding_dim);
            SCHECK(insert_variable(variable_id, std::move(pvar), meta));
        }
        SCHECK(this->meta(variable_id) == meta)
            << this->meta(variable_id).to_json_node().dump() << " " << meta.to_json_node().dump();
        return (*this)[variable_id];
    }
private:
    std::vector<uint32_t> _variable_ids;
    std::vector<EmbeddingVariableMeta> _metas;
    std::vector<std::shared_ptr<EmbeddingVariableBase>> _variables;
};

struct PendingRequest {
    ps::PSMessageMeta psmeta;
    ps::PSRequest request;
};

class EmbeddingStorage : public ps::ShardStorage  {
public:
    using ps::ShardStorage::_shards;
    typedef uint64_t key_type;
    typedef EmbeddingShard shard_type;
    EmbeddingStorage(const std::unordered_set<int32_t>& shard_id, const Configure&) {
        for (const auto& id : shard_id) {
            create_shard(id);
        }
    }

    void clear() override {
        for (auto& shard : _shards) {
            shard.second->data = shard_type();
        }
    }

    virtual bool create_shard(int32_t shard_id) override {
        core::lock_guard<RWSpinLock> lk(this->_mtx);
        if (_shards.count(shard_id) != 0) {
            return false;
        }
        _shards.emplace(shard_id, std::make_unique<ps::ShardData>());
        _shards[shard_id]->data = EmbeddingShard();
        _shards_meta.emplace(shard_id, std::make_unique<ps::ShardDataMeta>());
        _shards_meta[shard_id]->on_dcpmm = false;
        return true;
    }

    //no use
    virtual size_t shard_size(int32_t) override {
        return 0;
    }

    //no use
    virtual size_t shard_memory_usage(int32_t) override {
        return 0;
    }

    virtual ps::ShardIterator* get_shard_iterator(int32_t, int32_t) override {
        SLOG(FATAL) << "No implementation";
        return nullptr;
    }

    core::RWSpinLock& shared_mutex() {
        return this->_mtx;
    }

    core::RWSpinLock pending_mutex;
    int64_t batch_id = 0;
    std::atomic<size_t> async_tasks = {0};
    core::deque<core::vector<PendingRequest>> pending;
    core::vector<data_block_t> holders;
};


}
}
}


#endif