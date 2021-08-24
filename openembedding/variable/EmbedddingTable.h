#ifndef PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_VARIABLE_H


#include <limits>
#include <pico-ps/common/EasyHashMap.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class Key>
class EmbeddingTable {
public:
    using key_type = Key;
    EmbeddingTable(key_type empty_key, size_t value_size)
        : _table(empty_key), _value_size(value_size), _pool(value_size), _pmem_pool(value_size) {
        _cache_head.prev = _cache_head.next = &_cache_head;
    }


    char* operator[](const key_type& key) {
        auto it = _table.find(key);
        if (it != _table.end()) {

        }
    }
    
    size_t _value_size = 0;
    EasyHashMap<key_type, char*> _table;
};

}
}
}

#endif