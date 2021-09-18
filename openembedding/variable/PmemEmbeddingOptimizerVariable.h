#ifndef PARADIGM4_HYPEREMBEDDING_PMEM_EMBEDDING_OPTIMIZER_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_PMEM_EMBEDDING_OPTIMIZER_VARIABLE_H

#include <limits>
#include <pico-core/VirtualObject.h>
#include "Meta.h"
#include "PmemEmbeddingTable.h"
#include "EmbeddingOptimizerVariable.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class EmbeddingVariableKeyReaderInterface;
class EmbeddingVariableInterface;

template<class Table, class Optimizer>
class PmemEmbeddingOptimizerVariable: public EmbeddingOptimizerVariableBasic<Table, Optimizer> {
    using key_type = typename Table::key_type;
    using T = typename Optimizer::weight_type;
public:
    PmemEmbeddingOptimizerVariable(size_t embedding_dim, key_type empty_key)
        : EmbeddingOptimizerVariableBasic<Table, Optimizer>(embedding_dim, empty_key),
          _cache(empty_key) {}

    void copy_from(EmbeddingOptimizerVariableInterface<key_type, T>&& other, size_t block_num_items)override {
        EmbeddingOptimizerVariableBasic<Table, Optimizer>::copy_from(std::move(other), block_num_items);

        key_type item_key;
        typename EmbeddingHashTable<key_type, T>::Reader item_reader(*this->_new_weights);
        while (item_reader.read_key(item_key)) {
            _cache.try_emplace(item_key, this->_table.set_value(item_key));
        }
    }

    void set_variable_context(const EmbeddingVariableContext& variable_context) override {
        _variable_context = variable_context;
    }

    void set_batch_id(int64_t batch_id) override {
        _variable_batch_id = batch_id;
    }

    void load_config(const core::Configure& config) override {
        EmbeddingOptimizerVariableBasic<Table, Optimizer>::load_config(config);
        std::string pmem_pool_path;
        LOAD_CONFIG(config, pmem_pool_path);
        if (pmem_pool_path.empty()) {
            // new pmem pool
            if (_pmem_pool_path.empty()) {
                SCHECK(_cache.size() == 0);
                _pmem_pool_path = this->_table.create_pmem_pool();
            }
            SCHECK(!_pmem_pool_path.empty());
        } else {
            SCHECK(_cache.size() == 0);
            int64_t checkpoint = -1;
            LOAD_CONFIG(config, checkpoint);
            SCHECK(checkpoint != -1);
            _pmem_pool_path = pmem_pool_path;
            SCHECK(this->_table.load_pmem_pool(_pmem_pool_path, checkpoint));
        }
    }

    bool persist_config(size_t persist_pending_window, core::Configure& config) override {
        auto& _table = this->_table;
        int64_t checkpoint = _table.start_commit_checkpoint();
        std::string hit_rate = "0.0";
        if (_table.set_count()) {
            size_t rate1000 = 1000 * _table.hit_count() / _table.set_count();
            hit_rate = std::to_string(rate1000 / 10) + "." + std::to_string(rate1000 % 10);
        }
        while (_table.pending_checkpoints().size() > persist_pending_window) {
            size_t flush_count = _table.flush_count();
            _table.flush_committing_checkpoint();
            SLOG(INFO) << "flush committing checkpoint, "
                        << _table.flush_count() - flush_count << " item flushed.";
        }
        while (_table.checkpoints().size() > persist_pending_window) {
            _table.pop_checkpoint();
        }

        SLOG(INFO) << "batch id " << _variable_batch_id
                << ", variable id " << _variable_context.variable_id
                << ", hit rate " << hit_rate << "%"
                << ", flushed " << _table.flush_count()
                << ", all " << _table.set_count()
                << ", checkpoints " << show(_table.checkpoints())
                << ", pending checkpoints " << show(_table.pending_checkpoints())
                << ", pmem items " << _table.num_pmem_items()
                << ", cache items " << _table.num_cache_items();

        this->dump_config(config);
        std::string pmem_pool_path = _pmem_pool_path;
        SAVE_CONFIG(config, pmem_pool_path);
        SAVE_CONFIG(config, checkpoint);
        return true;
    }

    bool should_persist() override {
        return this->_table.should_commit_checkpoint();
    }

    void set_weights(const key_type* keys, size_t n, const T* weights, const T* states)override {
        EmbeddingOptimizerVariableBasic<Table, Optimizer>::set_weights(keys, n, weights, states);
        this->_table.next_work();
    }

    void pull_weights(const key_type* keys, size_t n,
          T* weights, VariableAsyncTask& async_task) override {
        size_t dim = this->embedding_dim();
        size_t value_dim = dim + this->embedding_optimizer()->state_dim(dim);

        PresistentAsyncDone async_done;
        async_done.variable = this;
        async_done.keys.assign(keys, keys + n);
        async_done.values.resize(n * value_dim);
        async_done.hints.resize(n);
        core::vector<size_t> new_keys;
        for (size_t i = 0; i < n; ++i) {
            const T* value = this->_table.get_value(keys[i], async_done.hints[i]);
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
        this->_table.next_work();
        _cache.clear();
    }

private:
    std::string show(const std::deque<int64_t>& vals) {
        std::string show_vals = "[";
        for (int64_t val: vals) {
            show_vals += std::to_string(val);
            show_vals += " ";
        }
        if (show_vals.back() == ' ') {
            show_vals.pop_back();
        }
        show_vals += "]";
        return show_vals;
    }

    struct PresistentAsyncDone {
        core::vector<key_type> keys;
        core::vector<T> values;
        core::vector<typename Table::ItemHint> hints;
        PmemEmbeddingOptimizerVariable* variable = nullptr;
        void operator()() {
            if (keys.empty()) {
                return;
            }
            T* from = values.data();
            size_t value_dim = values.size() / keys.size();
            for (size_t i = 0; i < keys.size(); ++i) {
                auto pair = variable->_cache.try_emplace(keys[i], nullptr);
                if (pair.second) {
                    T* value = variable->_table.set_value(keys[i], hints[i]);
                    std::copy_n(from, value_dim, value);
                    pair.first->second = value;
                }
                from += value_dim;
            }
        }
    };

    size_t _variable_batch_id = 0;
    std::string _pmem_pool_path;
    EmbeddingVariableContext _variable_context;
    core::RWSpinLock _lock;
    EasyHashMap<key_type, T*> _cache;
};

}
}
}

#endif
