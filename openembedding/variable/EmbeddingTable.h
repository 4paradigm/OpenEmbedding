#ifndef PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_VARIABLE_H

#include <pico-core/pico_log.h>
#include <pico-ps/common/EasyHashMap.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class Key, class T>
class EmbeddingTable {
public:
    using key_type = Key;
    EmbeddingTable(size_t value_dim, key_type empty_key)
        : _table(empty_key), _value_dim(value_dim),
          _block_dim(_value_dim * (15 * 1024 / sizeof(T) / _value_dim + 1)) {}

    class Reader {
    public:
        Reader(const EmbeddingTable<key_type, T>& table)
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

    size_t num_items()const {
        return _table.size();
    }

    // thread safe
    const T* get_value(const key_type& key) {
        auto it = _table.find(key);
        if (it == _table.end()) {
            return nullptr;
        }
        return it->second;
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

    T* operator[](const key_type& key) {
        return set_value(key);
    }
private:
    EasyHashMap<key_type, T*> _table;
    std::deque<core::vector<T>> _pool;
    size_t _value_dim = 0;
    size_t _block_dim = 0;
    size_t _p = 0;
};

template<class Key, class T>
class EmbeddingArray {
public:
    using key_type = Key;
    explicit EmbeddingArray(size_t value_dim)
        : _value_dim(value_dim) {}

    class Reader {
    public:
        Reader(EmbeddingArray& table)
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
        EmbeddingArray* _table = nullptr;
    };

    size_t num_items()const {
        return _num_items;
    }

    // thread safe
    const T* get_value(key_type key) {
        if (key < _num_items) {
            if (_num_items == _upper_bound || _valid[key]) {
                return _table.data() + key * _value_dim;
            }
        }
        return nullptr;
    }

    T* set_value(key_type key) {
        if (key >= _upper_bound) {
            _upper_bound = key + 1;
            _valid.resize(_upper_bound);
            _table.resize(_upper_bound * _value_dim);
        }
        if (_num_items != _upper_bound && !_valid[key]) {
            _valid[key] = true;
            _num_items += 1;
        }
        return _table.data() + key * _value_dim;
    }

    T* operator[](key_type key) {
        return set_value(key);
    }

private:
    size_t _value_dim = 0;
    size_t _num_items = 0;
    size_t _upper_bound = 0;
    core::vector<T> _table;
    core::vector<bool> _valid;
};


}
}
}

#endif