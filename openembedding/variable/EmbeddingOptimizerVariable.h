#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_OPTIMIZER_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_OPTIMIZER_VARIABLE_H

#include <limits>
#include <pico-core/VirtualObject.h>
#include "Meta.h"
#include "EmbeddingTable.h"
#include "EmbeddingInitializer.h"
#include "EmbeddingOptimizer.h"
#include "MpscGradientReducer.h"
#include "VariableAsyncTask.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class EmbeddingVariableKeyReaderInterface;
class EmbeddingVariableInterface;

template<class Key>
class EmbeddingVariableKeyReader: core::VirtualObject {
    using key_type = Key;
public:
    virtual uint64_t cursor() = 0;
    virtual size_t read_keys(key_type* keys, size_t n) = 0;
};

template<class Key, class T>
class EmbeddingOptimizerVariableInterface: core::VirtualObject {
    using key_type = Key;
public:
    EmbeddingOptimizerVariableInterface(size_t embedding_dim, key_type empty_key)
        : _embedding_dim(embedding_dim),
          _new_weights(std::make_unique<EmbeddingHashTable<key_type, T>>(embedding_dim, empty_key)),
          _gradients(std::make_unique<MpscGradientReducer<key_type, T>>(embedding_dim, empty_key)),
          _initializer(std::make_unique<EmbeddingConstantInitializer<T>>()) {}
    virtual ~EmbeddingOptimizerVariableInterface() {}

    virtual EmbeddingTable<key_type, T>* embedding_table() = 0;
    virtual EmbeddingOptimizer<T>* embedding_optimizer() = 0;

    virtual void get_weights(const key_type* keys, size_t n,
          T* weights, T* states = nullptr) = 0;  // thread safe
    virtual void set_weights(const key_type* keys, size_t n,
          const T* weights, const T* states = nullptr) = 0;
    virtual std::unique_ptr<EmbeddingVariableKeyReader<key_type>> create_key_reader() = 0;

    virtual void pull_weights(const key_type* keys, size_t n,
          T* weights, VariableAsyncTask& async_task) = 0; // thread safe
    virtual void push_gradients(const key_type* keys, size_t n,
          const T* gradients, const uint64_t* counts, VariableAsyncTask& async_task) = 0; // thread safe
    virtual void update_weights() = 0;

    virtual void copy_from(EmbeddingOptimizerVariableInterface<key_type, T>&& other, size_t block_num_items) {
        size_t state_dim = other.embedding_optimizer()->state_dim(embedding_dim());
        std::vector<key_type> indices(block_num_items);
        std::vector<T> weights(indices.size() * embedding_dim());
        std::vector<T> states(indices.size() * state_dim);
        auto reader = other.create_key_reader();
        size_t n = 0;
        while ((n = reader->read_keys(indices.data(), indices.size()))) {
            if (embedding_optimizer()->category() == other.embedding_optimizer()->category()) {
                other.get_weights(indices.data(), n, weights.data(), states.data());
                this->set_weights(indices.data(), n, weights.data(), states.data());
            } else {
                other.get_weights(indices.data(), n, weights.data());
                this->set_weights(indices.data(), n, weights.data());
            }
        }
        
        _new_weights = std::move(other._new_weights);
        _gradients = std::move(other._gradients);
        _initializer = std::move(other._initializer);
    }

    size_t embedding_dim() {
        return _embedding_dim;
    }
    
    std::unique_ptr<EmbeddingInitializer<T>>& embedding_initializer() {
        return _initializer;
    }

private:
    size_t _embedding_dim = 0;
protected:
    std::unique_ptr<EmbeddingHashTable<key_type, T>> _new_weights;
    std::unique_ptr<MpscGradientReducer<key_type, T>> _gradients;
    std::unique_ptr<EmbeddingInitializer<T>> _initializer;
};

template<class Table, class Optimizer>
class EmbeddingOptimizerVariableBasic: public EmbeddingOptimizerVariableInterface<
      typename Table::key_type, typename Optimizer::weight_type> {
    using key_type = typename Table::key_type;
    using T = typename Optimizer::weight_type;
public:
    EmbeddingOptimizerVariableBasic(size_t embedding_dim, key_type empty_key)
        : EmbeddingOptimizerVariableInterface<key_type, T>(embedding_dim, empty_key),
          _table(embedding_dim + _optimizer.state_dim(embedding_dim), empty_key) {}

    ~EmbeddingOptimizerVariableBasic() {}

    EmbeddingTable<key_type, T>* embedding_table()override {
        return &_table;
    }

    EmbeddingOptimizer<T>* embedding_optimizer()override {
        return &_optimizer;
    }

    virtual void get_weights(const key_type* keys, size_t n, T* weights, T* states)override {
        size_t dim = this->embedding_dim();
        if (states == nullptr) {
            for (size_t i = 0; i < n; ++i) {
                const T* value = _table.get_value(keys[i]);
                if (value == nullptr) {
                    this->_initializer->train_init(weights, dim);
                } else {
                    std::copy_n(value, dim, weights);
                }
                weights += dim;
            }
        } else {
            size_t state_dim = _optimizer.state_dim(dim);;
            for (size_t i = 0; i < n; ++i) {
                const T* value = _table.get_value(keys[i]);
                if (value == nullptr) {
                    this->_initializer->train_init(weights, dim);
                    _optimizer.train_init({states, dim});
                } else {
                    std::copy_n(value, dim, weights);
                    std::copy_n(value + dim, state_dim, states);
                }
                weights += dim;
                states += state_dim;
            }
        }
    }
    
    virtual void set_weights(const key_type* keys, size_t n, const T* weights, const T* states)override {
        size_t dim = this->embedding_dim();
        if (states == nullptr) {
            for (size_t i = 0; i < n; ++i) {
                T* value = _table.set_value(keys[i]);
                _optimizer.train_init({value + dim, dim});
                std::copy_n(weights, dim, value);
                weights += dim;
            }
        } else {
            size_t state_dim = _optimizer.state_dim(dim);
            for (size_t i = 0; i < n; ++i) {
                T* value = _table.set_value(keys[i]);
                std::copy_n(weights, dim, value);
                std::copy_n(states, state_dim, value + dim);
                weights += dim;
                states += state_dim;
            }
        }
    }

    std::unique_ptr<EmbeddingVariableKeyReader<key_type>> create_key_reader()override {
        return std::make_unique<KeyReader>(_table);
    }

protected:
    class KeyReader: public EmbeddingVariableKeyReader<key_type> {
    public:
        KeyReader(Table& table): _reader(table) {}
        
        uint64_t cursor() override {
            return _cursor;
        }

        size_t read_keys(key_type* keys, size_t n) override {
            size_t i = 0;
            while (i < n && _reader.read_key(keys[i])) {
                ++_cursor;
                ++i;
            }
            return i;
        }

    private:
        size_t _cursor = 0;
        typename Table::Reader _reader;
    };

    Optimizer _optimizer;
    Table _table;
};

template<class Table, class Optimizer>
class EmbeddingOptimizerVariable: public EmbeddingOptimizerVariableBasic<Table, Optimizer> {
    using key_type = typename Table::key_type;
    using T = typename Optimizer::weight_type;
public:
    EmbeddingOptimizerVariable(size_t embedding_dim, key_type empty_key)
        : EmbeddingOptimizerVariableBasic<Table, Optimizer>(embedding_dim, empty_key) {}

    virtual void pull_weights(const key_type* keys, size_t n,
          T* weights, VariableAsyncTask&)override {
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
                T* value = this->_new_weights->update_value(keys[i]);
                if (value == nullptr) {
                    value = this->_new_weights->set_value(keys[i]);
                    this->_initializer->train_init(value, dim);
                }
                std::copy_n(value, dim, weights + i * dim);
            }
        }
    }
    
    virtual void push_gradients(const key_type* keys, size_t n,
          const T* gradients, const uint64_t* counts, VariableAsyncTask&)override {
        this->_gradients->push_gradients({keys, n, gradients, counts});
    }

    virtual void update_weights()override {
        size_t dim = this->embedding_dim();
        key_type item_key;
        const T* item_value = nullptr;
        typename EmbeddingHashTable<key_type, T>::Reader item_reader(*this->_new_weights);
        while ((item_value = item_reader.read_item(item_key))) {
            T* value = this->_table.set_value(item_key);
            std::copy_n(item_value, dim, value);
            this->_optimizer.train_init({value + dim, dim});
        }
        auto block = this->_gradients->reduce_gradients();
        const T* grad = block.gradients;
        for (size_t i = 0; i < block.n; ++i) {
            T* value = this->_table.update_value(block.keys[i]);
            if (value == nullptr) {
                value = this->_table.set_value(block.keys[i]);
                this->_initializer->train_init(value, dim);
                this->_optimizer.train_init({value + dim, dim});
            }
            this->_optimizer.update(value, {value + dim, dim}, block.counts[i], grad);
            grad += dim;
        }
        this->_new_weights->clear();
        this->_gradients->clear();
    }

    core::RWSpinLock _lock;
};

}
}
}

#endif
