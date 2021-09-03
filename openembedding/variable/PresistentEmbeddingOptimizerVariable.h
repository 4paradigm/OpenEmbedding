#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_OPTIMIZER_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_OPTIMIZER_VARIABLE_H

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
class PresistentEmbeddingOptimizerVariable: public EmbeddingOptimizerVariableBasic<Table, Optimizer> {
    using key_type = typename Table::key_type;
    using T = typename Optimizer::weight_type;
public:
    PresistentEmbeddingOptimizerVariable(size_t embedding_dim, key_type empty_key)
        : EmbeddingOptimizerVariableBasic<Table, Optimizer>(embedding_dim, empty_key),
          _thread_id(PresistentWriteThreadPool::sington().thread_id.fetch_add(1)) {}

    virtual void pull_weights(const key_type* keys, size_t n,
          T* weights, VariableAsyncTask& async_task) override {
        size_t dim = this->embedding_dim();
        core::vector<size_t> new_keys;
        for (size_t i = 0; i < n; ++i) {
            const T* value = this->_table.get_value(keys[i]);
            if (value == nullptr) {
                new_keys.push_back(i);
            } else {
                std::copy_n(value, dim, weights + i * dim);
            }
        }

        if (!new_keys.empty()) {
            core::lock_guard<core::RWSpinLock> lock(_lock);
            for (size_t i: new_keys) {
                const T* value = this->_new_weights->get_value(keys[i]);
                if (value == nullptr) {
                    T* new_value = this->_new_weights->set_value(keys[i]);
                    this->_initializer->train_init(new_value, dim);
                    value = new_value;
                }
                std::copy_n(value, dim, weights + i * dim);
            }
        }
        if (!keys.empty()) {
            PresistentAsyncDone async_done;
            async_done.keys.assign(keys, n);
            async_done.weights.assign(weights, n * dim);
            async_done.variable = this;
            async_task.done = std::move(done);
        }
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
            auto it = _cache.find(value);
            if (it != _cache.end()) {
                T* value = it->second;
                this->_optimizer.train_init({value + dim, dim});
            }
        }
        auto block = this->_gradients->reduce_gradients();
        const T* grad = block.gradients;
        for (size_t i = 0; i < block.n; ++i) {
            auto it = _cache.find(value);
            if (it != _cache.end()) {
                T* value = it->second;
                this->_optimizer.update(value, {value + dim, dim}, block.counts[i], grad);
            }
            grad += dim;
        }
        this->_new_weights->clear();
        this->_gradients->clear();
        _cache.clear();
    }
    core::RWSpinLock _lock;
    EasyHashMap<key_type, T*> _cache;
    size_t _thread_id;

private:
    struct PresistentAsyncDone {
        core::vector<key_type> keys;
        core::vector<T> weights;
        PresistentEmbeddingOptimizerVariable* variable = nullptr;
        void operator()() {
            size_t dim = weights.size() / keys.size();
            T* from = weights.data();
            for (const key_type& key: keys) {
                T* value = variable->_table.write(key);
                if (variable->_cache.try_emplace(key, value).second) {
                    copy_n(from, dim, value);
                }
                from += dim;
            }
        }
    };
};

}
}
}

#endif
