#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_VARIABLE_H

#include <limits>
#include "Meta.h"
#include "Factory.h"
#include "EmbeddingInitializer.h"
#include "EmbeddingOptimizer.h"
#include <pico-ps/common/EasyHashMap.h>

namespace paradigm4 {
namespace pico {
namespace embedding {


struct RWSpinLockGuard {
    RWSpinLockGuard(core::RWSpinLock& lock): _lock(&lock) {
        _lock->lock_shared();
    }

    ~RWSpinLockGuard() {
        if (_write) {
            _lock->unlock();
        } else {
            _lock->unlock_shared();
        }
    }
    
    void write() {
        if (_write) {
            return;
        }
        _write = true;
        _lock->upgrade();
    }

    core::RWSpinLock* _lock;
    bool _write = false;
};

class EmbeddingVariableIndexReader {
public:
    EmbeddingVariableIndexReader() {}
    virtual ~EmbeddingVariableIndexReader() {}
    EmbeddingVariableIndexReader(const EmbeddingVariableIndexReader&) = delete;
    EmbeddingVariableIndexReader& operator=(const EmbeddingVariableIndexReader&) = delete;

    virtual int reader_id() = 0;
    virtual size_t cursor() = 0;
    virtual size_t read(uint64_t* indices, size_t n) = 0;
};

class EmbeddingVariableBase {
public:
    virtual ~EmbeddingVariableBase() {}
    virtual uint64_t vocabulary_size() = 0;
    virtual void load_config(const core::Configure& config) = 0;
    virtual void dump_config(core::Configure& config) = 0;
    virtual void vocabulary_resize(uint64_t vocabulary_size) = 0;
    virtual void get_weights(const uint64_t* indices, size_t n, char* weights, RWSpinLockGuard& guard) = 0;
    virtual void set_weights(const uint64_t* indices, size_t n, const char* weights) = 0;
    
    virtual void read_only_get_weights(const uint64_t* indices, size_t n, char* weights) = 0;
    virtual void push_gradients(const uint64_t* indices, size_t n,
          const char* gradients, const uint64_t* counts) = 0; // thread safe
    virtual void update_weights() = 0;
    virtual size_t state_line_size() = 0;
    virtual void get_states(const uint64_t* indices, size_t n, char* states) = 0;
    virtual void set_states(const uint64_t* indices, size_t n, const char* states) = 0;

    virtual void clear_weights() = 0; //清空initializer，weights。optimizer不变，slots重置。

    virtual bool is_sparse() = 0;

    virtual size_t num_indices() = 0;
    virtual EmbeddingVariableIndexReader& get_reader(int reader_id) = 0; // thread safe
    virtual void release_reader(int reader_id) = 0; // thread safe
};

template<class T>
class EmbeddingVariable: public EmbeddingVariableBase {
public:
    EmbeddingVariable(uint64_t embedding_dim): _embedding_dim(embedding_dim),
          _table(-1), _offsets(-1) {
        SCHECK(_embedding_dim > 0);
    }

    uint64_t vocabulary_size() override {
        return _vocabulary_size;
    }

    void load_config(const core::Configure& config) override {
        std::string initializer;
        size_t n = num_indices();
        LOAD_CONFIG(config, initializer);
        // 设置initializer会重置所有weights和slots
        // 以第一次的设置的initializer为准，忽略initializer重复设置。
        if (!_initializer) {
            _initializer = Factory<EmbeddingInitializer<T>>::singleton().create(initializer, config[initializer]);
            SCHECK(_initializer) << "create initializer " << initializer 
                  << " with datatype " << DataType::from<T>().to_string() << " failed";
            resize_dense(0);
            resize_dense(n);
        }

        std::string optimizer;
        LOAD_CONFIG(config, optimizer);
        if (_optimizer && optimizer == _optimizer->category()) {
            // 支持修改learning_rate等
            _optimizer->load_config(config[optimizer]);
        } else if (!optimizer.empty()) {
            SCHECK(_optimizer == nullptr) << "not support change optimizer.";
            _optimizer = Factory<EmbeddingOptimizer<T>>::singleton().create(optimizer, config[optimizer]);
            SCHECK(_optimizer) << "create optimizer " << optimizer 
                  << " with datatype " << DataType::from<T>().to_string() << " failed";
            _state_dim = _optimizer->state_dim(_embedding_dim);
            _states.resize(n * _state_dim);
            for (size_t i = 0; i < n; ++i) {
                _optimizer->train_init(state_view(i));
            }
        }
    }

    void dump_config(core::Configure& config) override {
        if (_initializer) {
            std::string initializer = _initializer->category();
            SAVE_CONFIG(config, initializer);

            core::Configure conf;
            _initializer->dump_config(conf);
            config.node()[initializer] = conf.node();
        }
        if (_optimizer) {
            std::string optimizer = _optimizer->category();
            SAVE_CONFIG(config, optimizer);
        
            core::Configure conf;
            _optimizer->dump_config(conf);
            config.node()[optimizer] = conf.node();
        }
    }

    T* vec(size_t i) {
        return _weights.data() + i * _embedding_dim;
    }

    OptimizerStateView<T> state_view(uint64_t i) {
        return OptimizerStateView<T>(_states.data() + i * _state_dim, _embedding_dim);
    }

    void resize_dense(uint64_t n) {
        uint64_t left = num_indices();
        if (left != n) {
            SCHECK(_readers.empty()) << "write while reading.";
        }
        
        if (_weights.capacity() < _embedding_dim * n) {
            _weights.reserve(_embedding_dim * n * 1.5);
        }
        if (_weights.capacity() < _embedding_dim * n) {
            _weights.reserve(_embedding_dim * n * 1.5);
        }
        _weights.resize(n * _embedding_dim);
        _states.resize(n * _state_dim);
        for (uint64_t i = left; i < n; ++i) {
            if (_initializer) {
                _initializer->train_init(vec(i), _embedding_dim);
            }
            if (_optimizer) {
                _optimizer->train_init(state_view(i));
            }
        }
    }

    size_t find(uint64_t index) {
        // sparse也要check，因为EasyHashMap使用了(-1 max uint64_t)表示empty
        SCHECK(index < _vocabulary_size);
        if (is_sparse()) {
            auto it = _table.try_emplace(index, _table.size());
            if (it.second) {
                resize_dense(_table.size());
            }
            index = it.first->second;
        }
        return index;
    }

    size_t find(uint64_t index, RWSpinLockGuard& guard) {
        SCHECK(index < _vocabulary_size);
        if (is_sparse()) {
            if (!_table.count(index)) {
                guard.write();
                if (!_table.count(index)) {
                    _table.force_emplace(index, _table.size());
                    resize_dense(_table.size());
                }
            }
            index = _table.at(index);
        }
        return index;
    }

    void vec_read_only_find(uint64_t index, T* out) {
        SCHECK(index < _vocabulary_size);
        if (is_sparse()) {
            if (!_table.count(index)) {
                if (_initializer) {
                    _initializer->train_init(out, _embedding_dim);
                } else {
                    std::fill_n(out, _embedding_dim, 0);
                }
                return;
            }
            index = _table.at(index);
        } 
        std::copy_n(vec(index), _embedding_dim, out);
    }

    // 这个仅仅是一个shard的vocabulary_size
    void vocabulary_resize(uint64_t vocabulary_size) override {
        // 先不支持sparse变dense
        if (is_sparse()) {
            return;
        };

        uint64_t left = _vocabulary_size;
        _vocabulary_size = vocabulary_size;
        if (is_sparse()) {
            // dense变sparse
            for (uint64_t i = 0; i < left; ++i) {
                _table.force_emplace(i, i);
            }
        } else {
            resize_dense(vocabulary_size);
        }
    }

    
    void get_weights(const uint64_t* indices, size_t n, char* weights, RWSpinLockGuard& guard) override {
        T* to = reinterpret_cast<T*>(weights);
        for (size_t i = 0; i < n; ++i) {
            std::copy_n(vec(find(indices[i], guard)), _embedding_dim, to);
            to += _embedding_dim;
        }
    }

    void read_only_get_weights(const uint64_t* indices, size_t n, char* weights) override {
        T* to = reinterpret_cast<T*>(weights);
        for (size_t i = 0; i < n; ++i) {
            vec_read_only_find(indices[i], to);
            to += _embedding_dim;
        }
    }
    

    void set_weights(const uint64_t* indices, size_t n, const char* weights) override {
        const T* from = reinterpret_cast<const T*>(weights);
        for (size_t i = 0; i < n; ++i) {
            std::copy_n(from, _embedding_dim, vec(find(indices[i])));
            from += _embedding_dim;
        }
    }

    size_t state_line_size() override {
        return _state_dim * sizeof(T);
    }

    void get_states(const uint64_t* indices, size_t n, char* states)override {
        T* to = reinterpret_cast<T*>(states);
        for (size_t i = 0; i < n; ++i) {
            std::copy_n(state_view(find(indices[i]))[0], _state_dim, to);
            to += _state_dim;
        }
    }

    void set_states(const uint64_t* indices, size_t n, const char* states) override {
        const T* from = reinterpret_cast<const T*>(states);
        for (size_t i = 0; i < n; ++i) {
            std::copy_n(from, _state_dim, state_view(find(indices[i]))[0]);
            from += _state_dim;
        }
    }

    void push_gradients(const uint64_t* indices, size_t n,
          const char* gradients, const uint64_t* counts) override {
        std::lock_guard<core::RWSpinLock> guard(_block_lock);
        GradientBlock block = {indices, n, reinterpret_cast<const T*>(gradients), counts};
        _blocks.push_back(block);
    }

    void update_weights()override {
        _offsets.clear();
        _gradients.clear();
        _counts.clear();
        core::vector<GradientBlock> blocks;
        {
            std::lock_guard<core::RWSpinLock> guard(_block_lock);
            blocks = _blocks;
            _blocks.clear();
        }
        for (GradientBlock block: blocks) {
            const T* grad = block.gradients;
            for (size_t i = 0; i < block.n; ++i) {
                uint64_t index = block.indices[i];
                if (_offsets.count(index)) {
                    size_t offset = _offsets.at(index);
                    T* sum = _gradients.data() + offset * _embedding_dim;
                    for (size_t j = 0; j < _embedding_dim; ++j) {
                        sum[j] += grad[j];
                    }
                    _counts[offset] += block.counts[i];
                } else {
                    _offsets.force_emplace(index, _offsets.size());
                    _gradients.insert(_gradients.end(), grad, grad + _embedding_dim);
                    _counts.push_back(block.counts[i]);
                }
                grad += _embedding_dim;
            }
        }
        if (_optimizer) {
            for (auto& pair: _offsets) {
                const T* grad = _gradients.data() + pair.second * _embedding_dim;
                size_t i = find(pair.first);
                _optimizer->update(vec(i), state_view(i), _counts[pair.second], grad);
            }
        }
    }

    void clear_weights() override {
        _table.clear();
        resize_dense(0);
        _initializer = nullptr;
        if (!is_sparse()) {
            resize_dense(_vocabulary_size);
        }
    }

    bool is_sparse() override {
        return _vocabulary_size >= (1ull << 63);
    }

    size_t num_indices() override {
        return _weights.size() / _embedding_dim;
    }

    EmbeddingVariableIndexReader& get_reader(int reader_id) override {
        std::lock_guard<core::RWSpinLock> guard(_reader_lock);
        if (reader_id == -1) {
            reader_id = _next_reader_id++;
            _readers[reader_id] = std::make_unique<MyReader>(reader_id, num_indices());
            if (is_sparse()) {
                _readers[reader_id]->_is_sparse = true;
                _readers[reader_id]->_it = _table.begin();
            }   
        }
        return *_readers[reader_id];
    }

    void release_reader(int reader_id) override {
        SCHECK(reader_id >= 0);
        std::lock_guard<core::RWSpinLock> guard(_reader_lock);
        _readers.erase(reader_id);
        if (_readers.empty()) {
            _next_reader_id = 0;
        }
    }

    class MyReader: public EmbeddingVariableIndexReader {
    public:
        MyReader(int reader_id, size_t count): _reader_id(reader_id), _end(count) {}

        int reader_id() override {
            return _reader_id;
        }
        size_t cursor() override {
            return _cursor;
        }
        size_t read(uint64_t* indices, size_t n) override {
            size_t i = 0;
            while (i < n && _cursor < _end) {
                if (_is_sparse) {
                    indices[i] = _it->first;
                    ++_it;
                } else {
                    indices[i] = _cursor;
                }
                ++_cursor;
                ++i;
            }
            return i;
        }
    
        int _reader_id = 0;
        size_t _cursor = 0;
        size_t _end = 0;
        bool _is_sparse = false;
        EasyHashMap<uint64_t, size_t>::iterator _it;
    };



    std::unique_ptr<EmbeddingInitializer<T>> _initializer;
    std::unique_ptr<EmbeddingOptimizer<T>> _optimizer;
    size_t _embedding_dim = 0;
    size_t _vocabulary_size = 0;
    size_t _state_dim = 0;
    core::vector<T> _weights;
    core::vector<T> _states;
    EasyHashMap<uint64_t, size_t> _table;
    struct GradientBlock {
        const uint64_t* indices;
        size_t n;
        const T* gradients;
        const uint64_t* counts;
    };

    core::RWSpinLock _block_lock;
    core::vector<GradientBlock> _blocks;
    EasyHashMap<uint64_t, size_t> _offsets;
    core::vector<T> _gradients;
    core::vector<uint64_t> _counts;

    core::RWSpinLock _reader_lock;
    std::unordered_map<int, std::unique_ptr<MyReader>> _readers;
    int _next_reader_id = 0;
};

class EmbeddingVariableCreator {
public:
    template<class T>
    void operator()(TypeCase<T>, size_t embedding_dim, std::unique_ptr<EmbeddingVariableBase>& variable) {
        variable = std::make_unique<EmbeddingVariable<T>>(embedding_dim);
    }

    static std::unique_ptr<EmbeddingVariableBase> create(DataType datatype, size_t embedding_dim) {
        std::unique_ptr<EmbeddingVariableBase> variable;
        datatype.invoke(EmbeddingVariableCreator(), embedding_dim, variable);
        SCHECK(variable) << "unknown datatype: " << datatype.to_string();
        return variable;
    };
};


}
}
}

#endif
