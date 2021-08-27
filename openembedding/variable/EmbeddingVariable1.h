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

class EmbeddingVariableKeyReaderInterface;
class EmbeddingVariableInterface;

template<class Key>
class EmbeddingVariableKeyReader: core::NoncopyableObject {
public:
    virtual size_t read_keys(key_type* keys, size_t n) = 0;
};

template<class Key, class T>
class EmbeddingVariable: core::NoncopyableObject {
    using key_type = Key;
public:
    virtual void optimizer_load_config(const core::Configure& config) = 0;
    virtual void optimizer_dump_config(core::Configure& config) = 0;
    virtual size_t optimizer_state_dim() = 0;
    virtual size_t embedding_dim() = 0;
    virtual void set_initializer(const core::Configure& config) = 0;
    virtual void get_initializer(core::Configure& config) = 0;
    virtual void get_weights(const key_type* keys, size_t n, T* weights) = 0;
    virtual void set_weights(const key_type* keys, size_t n, const T* weights) = 0;
    virtual void get_states(const key_type* keys, size_t n, T* states) = 0;
    virtual void set_states(const key_type* keys, size_t n, const T* states) = 0;
    std::unique_ptr<EmbeddingVariableKeyReader<key_type>> key_reader() = 0;

    virtual void pull_weights(const key_type* keys, size_t n, T* weights) = 0; // thread safe
    virtual void push_gradients(const key_type* keys, size_t n,
          const T* gradients, const uint64_t* counts) = 0; // thread safe
    virtual void update_weights() = 0;
};

template<class Table, class Optimizer>
class EmbeddingOptimizerVariableBasic {
    using key_type = typename Table::key_type;
    using T = typename Optimizer::weight_type;
public:
    EmbeddingOptimizerVariableBasic(size_t embedding_dim, key_type empty_key)
        : _embedding_dim(embedding_dim),
          _table(embedding_dim + _optimizer.state_dim(embedding_dim), empty_key),
          _reducer(embedding_dim, empty_key),
          _new_items(embedding_dim, empty_key) {}

    ~EmbeddingOptimizerVariableBasic() {}
    
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
        T* output = weights;
        size_t dim = embedding_dim();
        for (size_t i = 0; i < n; ++i) {
            const T* value = this->_table.get_value(keys[i]);
            if (value == nullptr) {
                this->_initializer->train_init(output, dim);
            } else {
                std::copy_n(value, dim, output);
            }
            output += dim;
        }
    }
    
    virtual void set_weights(const key_type* keys, size_t n, const T* weights)override {
        for (size_t i = 0; i < n; ++i) {
            T* value = _table.set_value(keys[i]);
            std::copy_n(weights + i * _embedding_dim, _embedding_dim, value);
        }
    }

    virtual void get_states(const key_type* keys, size_t n, T* states)override {
        T* output = states;
        size_t state_dim = optimizer_state_dim();
        for (size_t i = 0; i < n; ++i) {
            const T* value = _table.get_value(keys[i]);
            if (value == nullptr) {
                _optimizer.train_init({output, _embedding_dim});
            } else {
                std::copy_n(value + _embedding_dim, state_dim, output);
            }
            output += state_dim;
        }
    }
    
    virtual void set_states(const key_type* keys, size_t n, const T* states)override {
        size_t state_dim = optimizer_state_dim();
        for (size_t i = 0; i < n; ++i) {
            T* value = _table.set_value(keys[i]);
            std::copy_n(states + i * state_dim, state_dim, value + _embedding_dim);
        }
    }

    std::unique_ptr<EmbeddingVariableKeyReader<key_type>> key_reader()override {
        return std::make_unique<EmbeddingVariableKeyReader<key_type>>(_table);
    }

protected:
    class KeyReader: public EmbeddingVariableKeyReader {
        KeyReader(const Table& table): _reader(table) {}
        size_t read_keys(key_type* keys, size_t n) override {
            size_t i = 0;
            while (i < n && _reader.read(keys[i])) {
                ++i;
            }
            return i;
        }
        typename Table::Reader _reader;
    };

    size_t _embedding_dim = 0;
    Optimizer _optimizer;
    Table _table;
    std::unique_ptr<EmbeddingInitializer<T>> _initializer;

    core::RWSpinLock _lock;
    EmbeddingTable<key_type, T> _new_items;
};

template<class Table, class Optimizer>
class EmbeddingOptimizerVariable: public EmbeddingOptimizerVariableBasic<Table, Optimizer> {
    using key_type = typename Table::key_type;
    using T = typename Optimizer::weight_type;
public:
    EmbeddingOptimizerVariable(size_t embedding_dim, key_type empty_key)
        : EmbeddingOptimizerVariableBasic<Table, Optimizer>(embedding_dim, empty_key) {}

    virtual void pull_weights(const key_type* keys, size_t n, T* weights)override {
        size_t dim = this->_embedding_dim;
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
                const T* value = this->_new_items.get_value(keys[i]);
                if (value == nullptr) {
                    T* new_value = this->_new_items.set_value(keys[i]);
                    this->_initializer->train_init(new_value, dim);
                    value = new_value;
                }
                std::copy_n(value, dim, weights + i * dim);
            }
        }
    }
    
    virtual void push_gradients(const key_type* keys, size_t n,
          const T* gradients, uint64_t counts)override {
        _reducer.push_block({keys, n, gradients, counts});
    }

    virtual void update_weights()override {
        size_t dim = this->_embedding_dim;
        auto block = _reducer.reduce();
        T* grad = block.gradients;
        for (size_t i = 0; i < block.n; ++i) {
            T* value = this->_table[block.keys[i]];
            OptimizerStateView<T> state_view(value + dim, dim);
            this->_optimizer.update(value, state_view, block.count[i], grad);
            grad += dim;
        }
        _reducer.clear();
    }
    MpscGradientReducer<key_type, T> _reducer;
};

}
}
}

#endif
