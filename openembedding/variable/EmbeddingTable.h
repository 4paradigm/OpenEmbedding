#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_TABLE_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_TABLE_H

#include <pico-core/pico_log.h>
#include <pico-ps/common/EasyHashMap.h>
#include "Factory.h"
#include "EmbeddingVariable.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class Key, class T>
class EmbeddingTable {
public:
    using key_type = Key;
    ~EmbeddingTable() {};
    virtual std::string category() = 0;
    virtual uint64_t num_items() = 0;
    virtual void reserve_items(uint64_t num_items) = 0;
};

template<class Key, class T>
class EmbeddingHashTable: public EmbeddingTable<Key, T> {
public:
    using key_type = Key;

    class Reader {
    public:
        Reader(EmbeddingHashTable<key_type, T>& table)
            : _it(table._table.begin()), _end(table._table.end()) {}

        bool read_key(key_type& out) {
            if (_it == _end) {
                return false;
            }
            out = _it->first;
            ++_it;
            return true;
        }

        const T* read_item(key_type& out) {
            if (_it == _end) {
                return nullptr;
            }
            out = _it->first;
            T* result = _it->second;
            ++_it;
            return result;
        }
    private:
        typename EasyHashMap<key_type, T*>::iterator _it, _end;
    };

    EmbeddingHashTable(size_t value_dim, key_type empty_key)
        : _table(empty_key), _value_dim(value_dim),
          _block_dim(_value_dim * (63 * 1024 / sizeof(T) / _value_dim + 1)) {}

    std::string category()override {
        return "hash";
    }    

    uint64_t num_items() override {
        return _table.size();
    }

    void reserve_items(uint64_t num_items) {
        _table.reserve(num_items);
    }

    // thread safe
    const T* get_value(const key_type& key) {
        return update_value(key);
    }

    // not thread safe
    T* set_value(const key_type& key) {
        auto it = _table.find(key);
        if (it == _table.end()) {
            if (_p == 0) {
                _pool.emplace_back(_block_dim);
            }
            it = _table.force_emplace(key, _pool.back().data() + _p);
            _p += _value_dim;
            if (_p == _block_dim) {
                _p = 0;
            }
        }
        return it->second;
    }

    T* update_value(const key_type& key) {
        auto it = _table.find(key);
        if (it == _table.end()) {
            return nullptr;
        }
        return it->second;
    }

    void clear() {
        _table.clear();
        if (!_pool.empty()) {
            while (_pool.size() > 1) {
                _pool.pop_back();
            }
            _p = _value_dim;
            if (_p == _block_dim) {
                _p = 0;
            }
        }
    }

private:
    EasyHashMap<key_type, T*> _table;
    std::deque<core::vector<T>> _pool;
    size_t _value_dim = 0;
    size_t _block_dim = 0;
    size_t _p = 0;
};

template<class Key, class T>
class EmbeddingArrayTable: public EmbeddingTable<Key, T> {
public:
    using key_type = Key;

    class Reader {
    public:
        Reader(EmbeddingArrayTable& table)
            : _key(0), _table(&table) {}

        bool read_key(key_type& out) {
            while (_key < _table->_upper_bound && _table->get_value(_key) == nullptr) {
                ++_key;
            }
            if (_key < _table->_upper_bound) {
                out = _key;
                ++_key;
                return true;
            }
            return false;
        }

    private:
        key_type _key = 0;
        EmbeddingArrayTable* _table = nullptr;
    };

    explicit EmbeddingArrayTable(size_t value_dim, const key_type&)
        : _value_dim(value_dim) {}
    

    std::string category() override {
        return "array";
    }

    uint64_t num_items() override {
        return _num_items;
    }

    void reserve_items(uint64_t num_items) override {
        _upper_bound = num_items;
        _table.resize(num_items * _value_dim);
        _valid.resize(num_items);
    }

    // thread safe
    const T* get_value(key_type key) {
        return update_value(key);
    }

    T* set_value(key_type key) {
        if (key >= _upper_bound) {
            reserve_items(key + 1);
        }
        if (_num_items < _upper_bound && !_valid[key]) {
            _valid[key] = true;
            _num_items += 1;
        }
        return _table.data() + key * _value_dim;
    }
    
    T* update_value(key_type key) {
        if (key < _upper_bound) {
            if (_num_items == _upper_bound || _valid[key]) {
                return _table.data() + key * _value_dim;
            }
        }
        return nullptr;
    }

private:
    size_t _value_dim = 0;
    size_t _num_items = 0;
    size_t _upper_bound = 0;
    std::vector<T> _table;
    std::vector<bool> _valid;
};


}
}
}

#endif