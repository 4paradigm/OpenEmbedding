#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_VARIABLE_H

#include <limits>
#include <pico-ps/common/EasyHashMap.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

class CacheMemoryManager {
public:
    static CacheMemoryManager& singleton() {
        static CacheMemoryManager manager;
        return manager;
    }

    void set_max_size(size_t max_size) {
        _max_size = max_size;
    }

    bool acquire(size_t size) {
        if (_size.fetch_add(size) + size < _max_size) {
            return true;
        } else {
            _size.fetch_sub(size);
            return false;
        }
    }

    void release(size_t size) {
        _size.fetch_sub(size);
    }

private:
    std::atomic<size_t> _max_size;
    std::atomic<size_t> _size;
};



template<class Key>
class PersistentEmbeddingTable {
    static_assert(std::is_trivially_copyable<Key>::value, "persistent table need trivally copyable key type.")
public:
    using key_type = Key;
    PersistentEmbeddingTable(key_type empty_key, size_t value_size)
        : _table(empty_key), _value_size(value_size), _pool(value_size), _pmem_pool(value_size) {
        _cache_head.prev = _cache_head.next = &_cache_head;
    }

    // thread safe
    const char* read(const key_type& key) {
        auto it = _table.find(key);
        if (it != _table.end()) {
            uintptr_t p = it->second;
            if (p & 1) {
                CacheItem* item = reinterpret_cast<CacheItem*>(p ^ 1);
                return item->data;
            } else {
                PersistentItem* item = reinterpret_cast<PersistentItem*>(p);
                return item->data;                
            }
        } else {
            return nullptr;
        }
    }

    // not thread safe.
    // Write only, should not be used to read and write.
    // Return a buffer to write and the value is undefined. 
    char* write(const key_type& key, int64_t version) {
        CacheItem* item = nullptr;
        auto it = _table.find(key);
        if (it != _table.end()) {
            uintptr_t p = it->second;
            if (p & 1) {
                item = reinterpret_cast<CacheItem*>(p ^ 1);
                if (item->version < _submitting) {
                    flush(item);
                    PersistentItem* temp = _pmem_pool.acquire(key);
                    _pmem_pool.flush(temp);
                    _pmem_pool.release(temp);
                }
                item->erase();
            } else {
                _pmem_pool.release(reinterpret_cast<PersistentItem*>(p));
            }
        }
        if (item == nullptr) {
            item = _pool.acquire(key);
        }
        if (item == nullptr) {
            item = _cache_head.next;
            if (item != &_cache_head && item->version < version) {
                item->erase();
                _table.at(item->key) = reinterpret_cast<uintptr_t>(flush(item));
                item->key = key;
            } else {
                item = _pool.force_acquire(key);
            }
        }

        _cache_head.insert(item);
        if (p == _table.end()) {
            it->second = reinterpret_cast<uintptr_t>(item) | 1;
        } else {
            _table.force_emplace(key, reinterpret_cast<uintptr_t>(item) | 1);
        }
        item->version = version;
        return item->data;
    }

    void submit(int64_t version) {

    }

    void recycle_earliest_checkpoint() {

    }

    std::vector<int64_t> checkpoints() {

    }

private:
    PersistentItem* flush(CacheItem* item) {
        
    }

    enum { ALIGN = 8, OVERHEAD = 32 };
    
    struct PersistentItem {
        int64_t version;
        key_type key;
        char data[ALIGN];
    };

    struct CacheItem {
        int64_t version;
        key_type key;
        CacheItem* next;
        CacheItem* prev;
        char data[ALIGN];

        void insert(CacheItem* item) {
            item->prev = prev;
            prev->next = item;
            item->next = this;
            this->next = item;
        }

        void erase() {
            next->prev = prev;
            prev->next = next;
        }
    };

    // not thread safe
    class CacheMemoryPool {
        CacheMemoryPool(size_t value_size)
            : _item_size((value_size + sizeof(CacheItem)) / ALIGN * ALIGN) {}
        CacheItem* acquire(const key_type& key) {
            if (_expanding && CacheMemoryManager::singleton().acquire(_item_size + OVERHEAD)) {
                return force_acquire();
            }
            _expanding = false;
            return nullptr;
        }

        CacheItem* force_acquire(const key_type& key) {
            _pool.emplace_back(_item_size);
            CacheItem* item = reinterpret_cast<CacheItem*>(_pool.back().data());
            item->next = item->prev = nullptr;
            _key_constructor.construct(item->key, key);
            return item;
        }
    private:
        std::deque<core::vector<char>> _pool;
        bool _expanding = true;
        size_t _item_size = 0;

        std::allocator<key_type> _key_constructor;
    };

    class PersistentMemoryPool {
    public:
        PersistentMemoryPool(size_t value_size)
            : _value_size((value_size + sizeof(PersistentItem) - 1 + 7) / 8 * 8) {}
        PersistentItem* acquire() {

        }

        void flush(PersistentItem*) {
            
        }

        void release(PersistentItem*) {

        }

    private:
        std::allocator<key_type> _key_constructor;
    };

    int64_t _submitting = -1;

    size_t _value_size = 0;
    EasyHashMap<key_type, uintptr_t> _table;
    CacheItem _cache_head;

    CacheMemoryPool _pool;
    PersistentMemoryPool _pmem_pool;
};



}
}
}

#endif
