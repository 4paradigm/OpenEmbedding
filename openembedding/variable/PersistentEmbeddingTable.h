#ifndef PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_VARIABLE_H

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



template<class Key, class T>
class PersistentEmbeddingTable {
    static_assert(std::is_trivially_copyable<Key>::value, "persistent table need trivally copyable key type.")
public:
    using key_type = Key;
    PersistentEmbeddingTable(size_t value_size, key_type empty_key)
        : _table(empty_key), _value_size(value_size), _pool(value_size), _pmem_pool(value_size) {
        _cache_head.prev = _cache_head.next = &_cache_head;
    }

    class KeyReader {
    public:
        KeyReader(EmbeddingTable<Key>& table)
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
        EasyHashMap<key_type, uintptr_t>::iterator _it, _end;
    };

    size_t num_items() {
        return _table.size();
    }

    // thread safe
    const T* get_value(const key_type& key) {
        auto it = _table.find(key);
        if (it == _table.end()) {
            return nullptr;
        }
        uintptr_t p = it->second;
        if (p & 1) {
            CacheItem* item = reinterpret_cast<CacheItem*>(p ^ 1);
            return item->data;
        } else {
            PersistentItem* item = reinterpret_cast<PersistentItem*>(p);
            return item->data;                
        }
    }

    // not thread safe.
    // Write only, should not be used to read and write.
    // Return a buffer to write and the value is undefined. 
    T* set_value(const key_type& key) {
        CacheItem* item = nullptr;
        auto it = _table.find(key);
        if (it != _table.end()) {
            uintptr_t p = it->second;
            if (p & 1) {
                item = reinterpret_cast<CacheItem*>(p ^ 1);
                if (item->version < _submitting) {
                    _pmem_pool.release(flush(item));
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
            if (item != &_cache_head && item->version < _version) {
                item->erase();
                _table.at(item->key) = reinterpret_cast<uintptr_t>(flush(item));
                item->key = key;
            } else {
                item = _pool.force_acquire(key);
            }
        }

        _cache_head.insert(item);
        if (it == _table.end()) {
            _table.force_emplace(key, reinterpret_cast<uintptr_t>(item) | 1);
        } else {
            it->second = reinterpret_cast<uintptr_t>(item) | 1;
        }
        item->version = _version;

        if (_submitting != -1 && _cache_head.next->version >= _submitting) {
            _checkpoints.push_back(_submitting);
            if (_pending.empty()) {
                _submitting = -1;
            } else {
                _submitting = _pending.front();
                _pending.pop();
            }
            // TODO _pmem_pool.update_checkpoint(_submitting);
            // 10 20 30 | 30 | 20
            // 10 20    | 20 | 20
        }
        return item->data;
    }

    int64_t version() {
        return _version;
    }

    void next_batch() {
        ++_version; 
    }

    bool hint_submit() {
        return !_pmem_pool.expanding() && _submitting == -1;
    }

    void submit() {
        if (_submitting == -1) {
            _submitting = _version;
        } else {
            _pending.push(_version);
        }
    }

    void pop_checkpoint() {
        /// TODO
        _checkpoints.pop_front();
    }

    int64_t flush_checkpoint() {
        /// TODO
        int64_t version = _submitting;
    }

    // 20 submit()
    // 30 checkpoints() --> []
    // 40 checkpoints() --> [20] && submit()
    // 50 checkpoints() --> [20]
    // 60 checkpoints() --> [20 40] && submit()
    // 80 checkpoints() --> [20 40 60] && submit() && pop_checkpoint() && [40, 60]
    //
    // 20 submit()
    // 40 submit()
    // 60 submit() && checkpoints() --> [] && flush_checkpoint() && checkpoints() --> [20] 
    // 
    const std::deque<int64_t>& checkpoints() {
        return _checkpoints;
    }

private:
    PersistentItem* flush(CacheItem* item) {
        PersistentItem* pmem_item = _pmem_pool.acquire(item->key);
        pmem_item->version = item->version;
        memcpy(pmem_item->data, item->data, value_size);
        _pmem_pool.flush(pmem_item);
        return pmem_item;
    }

    enum { ALIGN = 8, OVERHEAD = 32 };
    
    struct PersistentItem {
        int64_t version;
        key_type key;
        T data[1];
    };

    struct CacheItem {
        int64_t version;
        key_type key;
        CacheItem* next;
        CacheItem* prev;
        T data[1];

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

        bool expanding() {
            return _expanding;
        }
    private:
        std::deque<core::vector<T>> _pool;
        bool _expanding = true;
        size_t _item_size = 0;

        std::allocator<key_type> _key_constructor;
    };

    class PersistentMemoryPool {
    public:
        PersistentMemoryPool(size_t value_size)
            : _item_size((value_size + sizeof(PersistentItem) - 1 + 7) / 8 * 8) {}
        PersistentItem* acquire(const key_type& key) {
            PersistentItem* pmem_item; 
            // TODO allocate pmem

            _key_constructor.construct(pmem_item->key, key); // pmem_item->key = key
            return pmem_item;
        }

        void flush(PersistentItem* pmem_item) {
            // flush((char*)pmem_item, _item_size);
        }

        void release(PersistentItem* pmem_item) {
            // 
        }

        // release 0 0 0
        // release 1
        // release 2
        // acquire --> new
        // release_version(1)
        // acquire --> 0
        // acquire --> 0
        // acquire --> 0
        // acquire --> new 
        // release_version(2)
        // acquire --> 1
        void release_version(int64_t version) {
            // 
        }
    private:
        size_t _item_size = 0;
        std::allocator<key_type> _key_constructor;
    };

    std::deque<int64_t> _checkpoints;
    std::queue<int64_t> _pending;

    int64_t _version = 0;
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
