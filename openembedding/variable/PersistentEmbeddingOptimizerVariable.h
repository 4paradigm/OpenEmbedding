#ifndef PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_OPTIMIZER_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_OPTIMIZER_VARIABLE_H

#include <limits>
#include <pico-core/VirtualObject.h>
#include "Meta.h"
#include "PersistentEmbeddingTable.h"
#include "EmbeddingOptimizerVariable.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class EmbeddingVariableKeyReaderInterface;
class EmbeddingVariableInterface;

template<class Table, class Optimizer>
class PersistentEmbeddingOptimizerVariable: public EmbeddingOptimizerVariableBasic<Table, Optimizer> {
    using key_type = typename Table::key_type;
    using T = typename Optimizer::weight_type;
public:
    PersistentEmbeddingOptimizerVariable(size_t embedding_dim, key_type empty_key)
        : EmbeddingOptimizerVariableBasic<Table, Optimizer>(embedding_dim, empty_key),
          _cache(empty_key) {}

    void set_weights(const key_type* keys, size_t n, const T* weights, const T* states)override {
        EmbeddingOptimizerVariableBasic<Table, Optimizer>::set_weights(keys, n, weights, states);
        this->_table.next_batch();
    }

    void copy_from(EmbeddingOptimizerVariableInterface<key_type, T>&& other, size_t block_num_items)override {
        EmbeddingOptimizerVariableBasic<Table, Optimizer>::copy_from(std::move(other), block_num_items);

        key_type item_key;
        typename EmbeddingHashTable<key_type, T>::Reader item_reader(*this->_new_weights);
        while (item_reader.read_key(item_key)) {
            _cache.try_emplace(item_key, this->_table.set_value(item_key));
        }
    }

    void pull_weights(const key_type* keys, size_t n,
          T* weights, VariableAsyncTask& async_task) override {
        size_t dim = this->embedding_dim();
        size_t value_dim = dim + this->embedding_optimizer()->state_dim(dim);

        PresistentAsyncDone async_done;
        async_done.variable = this;
        async_done.keys.assign(keys, keys + n);
        async_done.values.resize(n * value_dim);
        
        core::vector<size_t> new_keys;
        for (size_t i = 0; i < n; ++i) {
            const T* value = this->_table.get_value(keys[i]);
            if (value == nullptr) {
                new_keys.push_back(i);
            } else {
                std::copy_n(value, dim, weights + i * dim);
                std::copy_n(value, value_dim, async_done.values.data() + i * value_dim);
            }
        }

        if (!new_keys.empty()) {
            core::lock_guard<core::RWSpinLock> lock(_lock);
            for (size_t i: new_keys) {
                T* value = this->_new_weights->update_value(keys[i]);
                if (value == nullptr) {
                    value = this->_new_weights->set_value(keys[i]);
                    this->_initializer->train_init(value, dim);
                }
                std::copy_n(value, dim, weights + i * dim);
                std::copy_n(value, dim, async_done.values.data() + i * value_dim);
            }
        }
        async_task.set_done(std::move(async_done));
    }
    
    virtual void push_gradients(const key_type* keys, size_t n,
          const T* gradients, const uint64_t* counts, VariableAsyncTask&) override {
        this->_gradients->push_gradients({keys, n, gradients, counts});
    }

    virtual void update_weights() override {
        size_t dim = this->embedding_dim();
        key_type item_key;
        const T* item_value = nullptr;
        typename EmbeddingHashTable<key_type, T>::Reader item_reader(*this->_new_weights);
        while ((item_value = item_reader.read_item(item_key))) {
            T* value = _cache.at(item_key);
            std::copy_n(item_value, dim, value);
            this->_optimizer.train_init({value + dim, dim});
        }
        auto block = this->_gradients->reduce_gradients();
        const T* grad = block.gradients;

        for (size_t i = 0; i < block.n; ++i) {
            auto it = _cache.find(block.keys[i]);
            T* value = nullptr;
            if (it == _cache.end()) {
                // happen when change table type or variable, or pull push not match. 
                value = this->_table.update_value(block.keys[i]);
                if (value == nullptr) {
                    value = this->_table.set_value(block.keys[i]);
                    this->_initializer->train_init(value, dim);
                    this->_optimizer.train_init({value + dim, dim});
                }
            } else {
                value = it->second;
            }
            this->_optimizer.update(value, {value + dim, dim}, block.counts[i], grad);
            grad += dim;
        }
        this->_new_weights->clear();
        this->_gradients->clear();
        this->_table.next_train_batch();
        _cache.clear();
        int64_t train_batch_id = this->_table.train_batch_id();
        if (PersistentManager::singleton().checkpoint() == train_batch_id) {
            this->_table.start_commit_checkpoint();
            if (this->_table.pending_checkpoints().size() > 2) {
                this->_table.flush_committing_checkpoint();
            }
            if (this->_table.checkpoints().size() > 2) {
                this->_table.pop_checkpoint();
            }
        } else {
            if (this->_table.hint_to_commit_checkpoint()) {
                PersistentManager::singleton().hint_checkpoint(train_batch_id);
            }
        }
    }
    core::RWSpinLock _lock;
    EasyHashMap<key_type, T*> _cache;

private:
    struct PresistentAsyncDone {
        core::vector<key_type> keys;
        core::vector<T> values;
        PersistentEmbeddingOptimizerVariable* variable = nullptr;
        void operator()() {
            if (keys.empty()) {
                return;
            }
            T* from = values.data();
            size_t value_dim = values.size() / keys.size();
            for (const key_type& key: keys) {
                auto pair = variable->_cache.try_emplace(key, nullptr);
                if (pair.second) {
                    T* value = variable->_table.set_value(key);
                    std::copy_n(from, value_dim, value);
                    pair.first->second = value;
                }
                from += value_dim;
            }
        }
    };
};

}
}
}

#endif