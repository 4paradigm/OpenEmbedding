#ifndef PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_VARIABLE_H


#include <limits>
#include <pico-ps/common/EasyHashMap.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

inline size_t embedding_align(size_t size) {
    return (size + 7) / 8 * 8;
}

template<class Key>
class EmbeddingTable {
    enum { BLOCK_SIZE = 16 * 1024 };
public:
    using key_type = Key;
    explicit EmbeddingTable(key_type empty_key, size_t value_size)
        : _table(empty_key), _value_size(value_size),
          _item_size(embedding_align(value_size)),
          _block_size((BLOCK_SIZE + _item_size) / _item_size) {
        _cache_head.prev = _cache_head.next = &_cache_head;
    }


    explicit class KeyReader {
    public:
        KeyReader(const EmbeddingTable<key_type>& table)
            : _it(table._table.begin()), _end(table._table.end()) {}

        bool read_key(key_type& out) {
            if (_it == _end) {
                return false;
            }
            out = _it->first;
            ++_it;
            return true;
        }
    private:
        EasyHashMap<key_type, char*>::iterator _it, _end;
    };


    size_t num_items() {
        return _table.size();
    }

    // thread safe
    const char* get_value(const key_type& key) {
        auto it = _table.find(key);
        if (it == _table.end()) {
            return nullptr;
        }
        return it->second;
    }


    // not thread safe
    char* set_value(const key_type& key) {
        auto it = _table.find(key);
        if (it == _table.end()) {
            if (_p == 0) {
                _pool.emplace_back(_block * _item_size);
            }
            it = _table.force_emplace(key, _pool.back().data() + _p);
            _p += _item_size;
            if (_p == _item_size * _block) {
                _p = 0;
            }
        }
        return it->second;
    }

    char* operator[](const key_type& key) {
        return set_value(key);
    }

private:
    size_t _value_size = 0;
    size_t _item_size = 0;
    size_t _block = 0;
    EasyHashMap<key_type, char*> _table;
    std::deque<core::vector<char>> _pool;
    size_t _p = 0;
};

class EmbeddingArray {
public:
    explicit EmbeddingArray(size_t value_size)
        : _value_size(value_size), _item_size(embedding_align(value_size + 1)) {}

    class KeyReader {
    public:
        KeyReader(EmbeddingArray& table)
            : _key(0), _table(&table) {}

        bool read_key(size_t& out) {
            while (_key < _table->num_items() && _table.get_value(_key) == nullptr) {
                ++_key;
            }
            if (_key < _table->num_items()) {
                out = _key;
                ++_key;
                return true;
            }
            return false;
        }

    private:
        size_t _key = 0;
        EmbeddingArray* _table = nullptr;
    };

    size_t num_items() {
        return _num_items;
    }

    // thread safe
    const char* get_value(size_t key) {
        if (key < _num_items && _table[key * _item_size + _value_size]) {
            return _table.data() + key * _item_size;
        }
        return nullptr;
    }

    // not thread safe
    char* set_value(size_t key) {
        if (key >= _num_items) {
            _num_items = key + 1;
            _table.resize(_num_items * _item_size);
        }
        _table[key * _item_size + _value_size] = true;
        return _table.data() + key * _item_size;
    }

    char* operator[](size_t key) {
        return set_value(key);
    }

private:
    size_t _value_size = 0;
    size_t _item_size = 0;
    size_t _num_items = 0;
    core::vector<char> _table;
};


}
}
}

#endif