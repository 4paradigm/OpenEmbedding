#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_VARIABLE_H

#include <limits>
#include <pico-core/VirtualObject.h>
#include "Meta.h"
#include "Factory.h"
#include "EmbeddingTable.h"
#include "EmbeddingInitializer.h"
#include "EmbeddingOptimizer.h"
#include "MpscGradientReducer.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class Key>
class EmbeddingVariableKeyReader: core::NoncopyableObject  {
public:
    virtual size_t read_keys(uint64_t* indices, size_t n) = 0;
};


template<class Key, class T>
class EmbeddingVariable: core::NoncopyableObject  {
    using key_type = Key;
public:
    virtual void optimizer_load_config(const core::Configure& config) = 0;
    virtual void optimizer_dump_config(core::Configure& config) = 0;
    virtual size_t optimizer_state_dim() = 0;
    virtual size_t embedding_dim() = 0;
    virtual void set_initializer(std::unique_ptr<EmbeddingInitializer<T>> initializer) = 0
    virtual void get_weights(const key_type* keys, size_t n, T* weights) = 0; // thread safe
    virtual void set_weights(const key_type* keys, size_t n, const T* weights) = 0;
    virtual void push_gradients(const key_type* keys, size_t n,
          const T* gradients, const uint64_t* counts) = 0; // thread safe
    virtual void update_weights() = 0;
    virtual void get_states(const key_type* keys, size_t n, T* states) = 0;
    virtual void set_states(const key_type* keys, size_t n, const T* states) = 0;
    std::unique_ptr<EmbeddingVariableKeyReader<key_type>> key_reader() = 0;
};

template<class Table, class Optimizer>
class EmbeddingVariableOptimizer {
    using key_type = typename Table::key_type;
    using T = typename Optimizer::weight_type;
public:
    EmbeddingVariableOptimizer(size_t embedding_dim, key_type empty_key)
        : _embedding_dim(embedding_dim),
          _table(embedding_dim + _optimizer.state_dim(embedding_dim), empty_key),
          _reducer(embedding_dim, empty_key),
          _new_items(embedding_dim, empty_key) {}

    ~EmbeddingVariableBase() {}
    
    void optimizer_load_config(const core::Configure& config)override {
        _optimizer.load_config(config);
    }

    void optimizer_dump_config(core::Configure& config)override {
        _optimizer.dump_config(config);
    }

    size_t optimizer_state_dim()override {
        return _optimizer.state_dim(_embedding_dim);
    }

    size_t embedding_dim()override {
        return _embedding_dim;
    }

    void set_initializer(std::unique_ptr<EmbeddingInitializer<T>> initializer)override {
        _initializer = std::move(initializer);
    }
    
    virtual void get_weights(const key_type* keys, size_t n, T* weights)override {
        core::vector<size_t> new_keys;
        for (size_t i = 0; i < n; ++i) {
            T* value = _table.get_value(keys[i]);
            if (value == nullptr) {
                new_keys.push_back(i);
            } else {
                std::copy_n(value, _embedding_dim, weights + i * _embedding_dim);
            }
        }

        if (!new_keys.empty()) {
            core::lock_guard<core::RWSpinLock> lock(_lock);
            for (size_t i: new_keys) {
                T* value = _new_items.get_value(keys[i]);
                if (value == nullptr) {
                    value = _new_items.set_value(keys[i]);
                    _initializer->train_init(value, _embedding_dim);
                }
                std::copy_n(value, _embedding_dim, weights + i * _embedding_dim);
            }
        }
    }
    
    virtual void set_weights(const key_type* keys, size_t n, const T* weights)override {
        for (size_t i = 0; i < n; ++i) {
            T* value = _table.set_value(keys[i]);
            std::copy_n(weights + i * _embedding_dim, _embedding_dim, value);
        }
    }
    
    virtual void push_gradients(const key_type* keys, size_t n,
          const T* gradients, uint64_t counts)override {
        _reducer.push_block({keys, n, gradients, counts})
    }

    virtual void update_weights()override {
        MpscGradientReducer<key_type, T>::block_type block = _reducer.reduce();
        for (size_t i = 0; i < block.n; ++i) {
            T* value = _table[block.keys[i]];
            T* state = OptimizerStateView<T>(value + _embedding_dim, _embedding_dim);
        }
        _reducer.clear();
    }


    virtual void get_states(const key_type* keys, size_t n, T* states)override {
        size_t dim = state_dim();
        size_t value_dim = _embedding_dim + dim;
        for (size_t i = 0; i < n; ++i) {
            T* value = _table.get_value(keys[i]);
            SCHECK(value);
            std::copy_n(value + _embedding_dim, dim, states + i * value_dim);
        }
    }
    
    virtual void set_states(const key_type* keys, size_t n, const T* states)override {
        size_t dim = state_dim();
        size_t value_dim = _embedding_dim + dim;
        for (size_t i = 0; i < n; ++i) {
            T* value = _table.set_value(keys[i]);
            std::copy_n(states + i * value_dim, dim, value + _embedding_dim);
        }
    }

    std::unique_ptr<EmbeddingVariableKeyReader<key_type>> key_reader()override {
        
    }

private:
    class KeyReader: public EmbeddingVariableKeyReader<key_type> {

    }

    size_t _embedding_dim = 0;
    Optimizer _optimizer;
    Table _table;
    std::unique_ptr<EmbeddingInitializer<T>> _initializer;
    MpscGradientReducer<key_type, T> _reducer;
    
    core::RWSpinLock _lock;
    EmbeddingTable<key_type> _new_items;
};


}
}
}

#endif
