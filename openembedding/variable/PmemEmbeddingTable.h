#ifndef PARADIGM4_HYPEREMBEDDING_PMEM_EMBEDDING_TABLE_H
#define PARADIGM4_HYPEREMBEDDING_PMEM_EMBEDDING_TABLE_H

#include <pico-ps/common/EasyHashMap.h>
#include "PmemEmbeddingItemPool.h"
#include "EmbeddingTable.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class EmbeddingArrayIndexTag {
public:
    static std::string category() { return "array"; }

    template<class Key, class Pointer>
    class EmbeddingIndex {
    public:
        using key_type = Key;
        EmbeddingIndex(key_type) {}

        void reserve_items(key_type num_items) {
            _table.reserve(num_items);
        }
        Pointer get_pointer(key_type key) {
            return key < _table.size() ? _table[key] : Pointer();
        }
        Pointer& set_pointer(key_type key) {
            if (key >= _table.size()) {
                _table.resize(key + 1);
            }
            return _table[key];
        }

        class Reader {
        public:
            Reader(EmbeddingIndex& table): _table(&table) {}

            bool read_key(key_type& out) {
                while (_key < _table->_table.size() && !_table->get_pointer(_key)) {
                    ++_key;
                }
                if (_key < _table->_table.size()) {
                    out = _key;
                    ++_key;
                    return true;
                }
                return false;
            }

        private:
            key_type _key = 0;
            EmbeddingIndex* _table = nullptr;
        };
        
    private:
        std::vector<Pointer> _table;
    };
};

struct EmbeddingHashIndexTag {
public:
    static std::string category() { return "hash"; }

    template<class Key, class Pointer>
    class EmbeddingIndex {
    public:
        using key_type = Key;

        EmbeddingIndex(key_type empty_key): _table(empty_key) {}

        void reserve_items(size_t num_items) {
            _table.reserve(num_items);
        }

        Pointer get_pointer(const Key& key) {
            auto it = _table.find(key);
            return it == _table.end() ? Pointer() : it->second;
        }

        Pointer& set_pointer(const Key& key) {
            return _table.try_emplace(key, Pointer()).first->second;
        }

        class Reader {
        public:
            Reader(EmbeddingIndex& table)
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
            typename EasyHashMap<key_type, Pointer>::iterator _it, _end;
        };

    private:
        EasyHashMap<Key, Pointer> _table;
    };
};

template<class Key, class T, class EmbeddingIndexTag>
class PmemEmbeddingTable: public EmbeddingTable<Key, T> {
public:
    using key_type = Key;
    static_assert(std::is_trivially_copyable<Key>::value, "pmem table need trivally copyable key type.");

    struct PmemItemHead {
        int64_t work_id = -1;
        key_type key = key_type();
    };

    struct CacheItemHead {
        using PmemItem = typename EmbeddingItemPool<PmemItemHead, T>::Item;
        using CacheItem = typename EmbeddingItemPool<CacheItemHead, T>::Item;
        int64_t work_id = -1;
        key_type key = key_type();
        CacheItem* next = nullptr;
        CacheItem* prev = nullptr;

        void erase() {
            next->prev = prev;
            prev->next = next;
        }

        void insert_prev(CacheItem* item) {
            item->prev = prev;
            prev->next = item;

            item->next = static_cast<CacheItem*>(this);
            this->prev = item;
        }
    };
    
    using PmemItem = typename EmbeddingItemPool<PmemItemHead, T>::Item;
    using CacheItem = typename CacheItemHead::CacheItem;

    struct ItemPointer {
        ItemPointer() {}
        ItemPointer(CacheItem* item): _p(reinterpret_cast<uintptr_t>(item) | 1) {}
        ItemPointer(PmemItem* pmem_item): _p(reinterpret_cast<uintptr_t>(pmem_item)) {}

        explicit operator bool()const {
            return _p;
        }

        bool is_cache_item()const {
            return _p & 1;
        }
        CacheItem* as_cache_item()const {
            return reinterpret_cast<CacheItem*>(_p ^ 1);
        }
        PmemItem* as_pmem_item()const {
            return reinterpret_cast<PmemItem*>(_p);
        }
    private:
        uintptr_t _p = 0;
    };

    struct ItemHint {
        friend PmemEmbeddingTable;
    private:
        key_type key;
        int64_t work_id = -1;
        int64_t prev_work_id = -1;
    };

    using EmbeddingIndex = typename EmbeddingIndexTag::template EmbeddingIndex<Key, ItemPointer>;
    class Reader {
    public:
        Reader(PmemEmbeddingTable& table): _base(table._table) {}
        bool read_key(key_type& out) { return _base.read_key(out); }
    private:
        typename EmbeddingIndex::Reader _base;
    };

    PmemEmbeddingTable(size_t value_dim, key_type empty_key)
        : _value_dim(value_dim), _empty_key(empty_key),
          _table(empty_key), _cache_pool(value_dim), _pmem_pool(value_dim) {
        _cache_head = _cache_pool.new_item();
        _cache_head->prev = _cache_head->next = _cache_head;
    }

    ~PmemEmbeddingTable() {}

    bool load_pmem_pool(const std::string& pmem_pool_path, int64_t checkpoint) {
        size_t value_dim = _value_dim;
        key_type empty_key = _empty_key;
        this->~PmemEmbeddingTable();
        new (this) PmemEmbeddingTable(value_dim, empty_key);
        if (_pmem_pool.load_pmem_pool(pmem_pool_path, checkpoint, _table, _num_items)) {
            _work_id = _committing = checkpoint;
            return true;
        }
        return false;
    }

    std::string category() override {
        return "pmem." + EmbeddingIndexTag::category();
    }

    uint64_t num_items() override {
        return _num_items;
    }

    void reserve_items(uint64_t n) override {
        _table.reserve_items(n);
    }

    void prefetch_reserve_cache(uint64_t n) {
        _cache_pool.prefetch_reserve(n);
    }

    std::string pmem_pool_path() {
        return _pmem_pool.pmem_pool_path();
    }

    // thread safe
    const T* get_value(const key_type& key) {
        ItemPointer it = _table.get_pointer(key);
        if (it) {
            if (it.is_cache_item()) {
                return it.as_cache_item()->data;
            } else {
                return it.as_pmem_item()->data;
            }
        }
        return nullptr;
    }

    const T* get_value(const key_type& key, ItemHint& hint) {
        hint.key = key;
        hint.work_id = _work_id;
        ItemPointer it = _table.get_pointer(key);
        if (it) {
            if (it.is_cache_item()) {
                hint.prev_work_id = it.as_cache_item()->work_id;
                return it.as_cache_item()->data;
            } else {
                hint.prev_work_id = it.as_pmem_item()->work_id;
                return it.as_pmem_item()->data;
            }
        }
        return nullptr;
    }


    // not thread safe.
    // Write only, should not be used to read and write.
    // Return a buffer to write and the value is undefined. 
    T* set_value(const key_type& key, const ItemHint& hint = ItemHint()) {
        ++_set_count;
        CacheItem* item = nullptr;

        // only one new key, no rehash after, hold it reference should be safe.
        ItemPointer& it = _table.set_pointer(key);
        if (it) {
            if (it.is_cache_item()) {
                ++_hit_count;
                item = it.as_cache_item();
                if (item->work_id < _committing) {
                    _pmem_pool.push_item(flush_to_pmem_item(item));
                }
                item->erase();
                _cache_head->insert_prev(item);
            } else {
                PmemItem* pmem_item = it.as_pmem_item();
                int64_t prev_work_id = 0;
                if (hint.work_id == _work_id && hint.key == key) {
                    prev_work_id = hint.prev_work_id;
                } else {
                    prev_work_id = pmem_item->work_id;
                }
                if (prev_work_id < _committing) {
                    _pmem_pool.push_item(pmem_item);
                } else {
                    _pmem_pool.free_item(pmem_item);
                }
                item = cache_miss_new_item();
            }
        } else {
            ++_num_items;
            item = cache_miss_new_item();
        }
        it = item;
        item->key = key;
        item->work_id = _work_id;
        return item->data;
    }

    T* update_value(key_type key) {
        T* result = nullptr;
        const T* value = get_value(key);
        if (value) {
            result = set_value(key);
            std::copy_n(value, _value_dim, result);
        }
        return result;
    }

    int64_t work_id() {
        return _work_id;
    }

    void next_work() {
        ++_work_id;
        if (!_pendings.empty() && _cache_head->next->work_id >= _pendings.front()) {
            _pmem_pool.push_checkpoint(_pendings.front());
            _pendings.pop_front();
        }
        // It is more efficient to create all pools at the same time.
        if (_pmem_pool.pmem_pool_path().empty()) {
            _pmem_pool.create_pmem_pool();
        }
    }

    bool should_commit_checkpoint() {
        return !_cache_pool.expanding() && _pendings.empty();
    }

    int64_t start_commit_checkpoint() {
        if (_work_id > _committing) {
            _committing = _work_id;
            _pendings.push_back(_committing);
        }
        _cache_pool.rebalance();
        return _committing;
    }

    void pop_checkpoint() {
        _pmem_pool.pop_checkpoint();
    }

    void flush_committing_checkpoint() {
        SCHECK(!_pendings.empty());
        CacheItem* item = _cache_head->next;
        while (item != _cache_head && item->work_id < _pendings.front()) {
            item->erase();
            _table.set_pointer(item->key) = flush_to_pmem_item(item);
            _cache_pool.delete_item(item);
            item = _cache_head->next;
        }
        if (!_pendings.empty()) {
            _pmem_pool.push_checkpoint(_pendings.front()); 
            _pendings.pop_front();
        }
        _cache_pool.rebalance();
    }

    const std::deque<int64_t>& checkpoints() {
        return _pmem_pool.checkpoints();
    }

    const std::deque<int64_t>& pending_checkpoints() {
        return _pendings;
    }

    size_t cache_item_memory_cost() {
        return _cache_pool.item_memory_cost();
    }

    size_t hit_count() {
        return _hit_count;
    }

    size_t set_count() {
        return _set_count;
    }

    size_t flush_count() {
        return _flush_count;
    }

    size_t num_cache_items() {
        return _cache_pool.num_items();
    }

    size_t num_pmem_items() {
        return _pmem_pool.num_items();
    }

    size_t num_freespace_items() {
        return _pmem_pool.num_freespace_items();
    }

    // debug
    uint64_t get_avaiable_freespace_slots() {
        return _pmem_pool.get_avaiable_freespace_slots();
    }

private:
    PmemItem* flush_to_pmem_item(CacheItem* item) {
        ++_flush_count;
        PmemItem* pmem_item = _pmem_pool.new_item();
        pmem_item->key = item->key;
        pmem_item->work_id = item->work_id;
        std::copy_n(item->data, _value_dim, pmem_item->data);
        _pmem_pool.flush_item(pmem_item);
        return pmem_item;
    }

    CacheItem* cache_miss_new_item() {
        CacheItem* item = _cache_pool.try_new_item();
        if (item == nullptr) {
            item = _cache_head->next;
            if (item != _cache_head && item->work_id < _work_id) {
                item->erase();
                _table.set_pointer(item->key) = flush_to_pmem_item(item);
            } else {
                item = _cache_pool.new_item();
            }
        }
        _cache_head->insert_prev(item);
        return item;
    }

    std::deque<int64_t> _pendings;

    // _train_batch_id will be dump and load as a configure property when changing variable type.
    // int64_t _train_batch_id = 0;
    int64_t _train_batch_id = 0;
    int64_t _work_id = 0; 
    int64_t _committing = 0;
    
    size_t _value_dim = 0;
    key_type _empty_key = key_type();
    uint64_t _num_items = 0;
    EmbeddingIndex _table;
    CacheItem* _cache_head = nullptr;

    CacheItemPool<CacheItemHead, T> _cache_pool;
    PmemItemPool<PmemItemHead, T> _pmem_pool;

    size_t _hit_count = 0;
    size_t _set_count = 0;
    size_t _flush_count = 0;
};

template<class Key, class T>
using PmemEmbeddingArrayTable = PmemEmbeddingTable<Key, T, EmbeddingArrayIndexTag>;

template<class Key, class T>
using PmemEmbeddingHashTable = PmemEmbeddingTable<Key, T, EmbeddingHashIndexTag>;


}
}
}

#endif
