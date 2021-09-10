#ifndef PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_TABLE_H
#define PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_TABLE_H

#include <limits>
#include <pico-ps/common/EasyHashMap.h>

#include <libpmemobj++/pool.hpp>
#include <libpmemobj++/p.hpp>
#include <libpmemobj++/make_persistent.hpp>
#include <libpmemobj++/transaction.hpp>
#include <libpmemobj++/persistent_ptr.hpp>
#include <libpmemobj++/container/string.hpp>
#include <libpmemobj++/container/vector.hpp>
#include <libpmemobj++/container/segment_vector.hpp>
//#include <libpmemobj++/container/concurrent_hash_map.hpp>

#include <sys/stat.h>
#include <string>
#include <queue>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include <pico-core/RpcServer.h>

#include "persist.h"
#include "EmbeddingItemPool.h"
#include "EmbeddingTable.h"
#include "PersistentManager.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class Key, class T>
class PersistentEmbeddingTable: public EmbeddingTable<Key, T> {
public:
    using key_type = Key;
    static_assert(std::is_trivially_copyable<Key>::value, "persistent table need trivally copyable key type.");
    
    struct PersistentItemHead {
        int64_t work_id = 0;
        key_type key = key_type();
    };

    struct CacheItemHead {
        using PersistentItem = typename EmbeddingItemPool<PersistentItemHead, T>::Item;
        using CacheItem = typename EmbeddingItemPool<CacheItemHead, T>::Item;
        int64_t work_id = 0;
        key_type key = key_type();
        PersistentItem* pmem = nullptr;
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
    
    using PersistentItem = typename EmbeddingItemPool<PersistentItemHead, T>::Item;
    using CacheItem = typename CacheItemHead::CacheItem;

    struct ItemPointer {
        ItemPointer(CacheItem* item): _p(reinterpret_cast<uintptr_t>(item) | 1) {}
        ItemPointer(PersistentItem* pmem_item): _p(reinterpret_cast<uintptr_t>(pmem_item)) {}

        bool is_cache_item()const {
            return _p & 1;
        }
        CacheItem* as_cache_item()const {
            return reinterpret_cast<CacheItem*>(_p ^ 1);
        }
        PersistentItem* as_persistent_item()const {
            return reinterpret_cast<PersistentItem*>(_p);
        }
    private:
        uintptr_t _p = 0;
    };

    class Reader {
    public:
        Reader(PersistentEmbeddingTable& table)
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
        typename EasyHashMap<key_type, ItemPointer>::iterator _it, _end;
    };

    PersistentEmbeddingTable(size_t value_dim, key_type empty_key)
        : _value_dim(value_dim), _table(empty_key), _cache_pool(value_dim), _pmem_pool(value_dim) {
        _cache_head = _cache_pool.new_item();
        _cache_head->work_id = std::numeric_limits<int64_t>::max();
        _cache_head->prev = _cache_head->next = _cache_head;
    }

    ~PersistentEmbeddingTable() {}

    void open_pmem_pool(const std::string& pmem_pool_path) {
        _pmem_pool.open_pmem_pool(pmem_pool_path);
    }

    std::string category() override {
        return "mixpmem";
    }

    size_t num_items() override {
        return _table.size();
    }

    // thread safe
    const T* get_value(const key_type& key) {
        auto it = _table.find(key);
        if (it == _table.end()) {
            return nullptr;
        }
        if (it->second.is_cache_item()) {
            return it->second.as_cache_item()->data;
        } else {
            return it->second.as_persistent_item()->data;
        }
    }

    // not thread safe.
    // Write only, should not be used to read and write.
    // Return a buffer to write and the value is undefined. 
    T* set_value(const key_type& key) {
        ++_set_count;
        CacheItem* item = nullptr;
        auto it = _table.find(key);
        if (it != _table.end()) {
            if (it->second.is_cache_item()) {
                ++_hit_count;
                item = it->second.as_cache_item();
                if (item->work_id < _committing) {
                    flush_to_pmem_item(item);
                    _pmem_pool.push_item(item->pmem);
                    item->pmem = nullptr;
                }
                item->erase();
                _cache_head->insert_prev(item);
                it->second = item;
            } else {
                PersistentItem* pmem_item = it->second.as_persistent_item();
                // if (pmem_item->version < _committing) {
                //     _pmem_pool.push_item(pmem_item);
                // } else {
                //     _pmem_pool.free_item(pmem_item);
                // }
                _pmem_pool.push_item(pmem_item);
                item = cache_miss_new_item();
                it->second = item;
            }
        } else {
            item = cache_miss_new_item();
            _table.force_emplace(key, item);
        }
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
            _checkpoints.push_back(_pendings.front());
            _pmem_pool.push_checkpoint(_pendings.front());
            _pendings.pop_front();
        }
    }

    bool hint_to_commit_checkpoint() {
        return !_cache_pool.expanding() && _pendings.empty();
    }

    void start_commit_checkpoint() {  //trans start
        _committing = _work_id;
        _pendings.push_back(_committing);
    }

    void pop_checkpoint() {
        _checkpoints.pop_front();
        _pmem_pool.pop_checkpoint();
    }

    void flush_committing_checkpoint() {
        SCHECK(!_pendings.empty());
        CacheItem* item = _cache_head->next;
        while (item != _cache_head && item->work_id < _pendings.front()) {
            flush_to_pmem_item(item);
            item = item->next;
        }

        if (!_pendings.empty()) {
            _checkpoints.push_back(_pendings.front());
            _pmem_pool.push_checkpoint(_pendings.front()); 
            _pendings.pop_front();
        }
    }

    const std::deque<int64_t>& checkpoints() {
        return _checkpoints;
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

    // debug
    uint64_t get_avaiable_freespace_slots() {
        return _pmem_pool.get_avaiable_freespace_slots();
    }
    uint64_t get_all_freespace_slots() {
        return _pmem_pool.get_all_freespace_slots();
    }

private:
    void flush_to_pmem_item(CacheItem* item) {
        if (item->pmem == nullptr) {
            ++_flush_count;
            PersistentItem* pmem_item = _pmem_pool.new_item();
            pmem_item->key = item->key;
            pmem_item->work_id = item->work_id;
            std::copy_n(item->data, _value_dim, pmem_item->data);
            _pmem_pool.flush_item(pmem_item);
            item->pmem = pmem_item;
        }
    }

    CacheItem* cache_miss_new_item() {
        CacheItem* item = _cache_pool.try_new_item();
        if (item == nullptr) {
            item = _cache_head->next;
            if (item != _cache_head && item->work_id < _work_id) {
                item->erase();
                flush_to_pmem_item(item);
                _table.at(item->key) = item->pmem;
                item->pmem = nullptr;
            } else {
                item = _cache_pool.new_item();
            }
        }
        _cache_head->insert_prev(item);
        return item;
    }

    // not thread safe
    class CacheItemPool: public EmbeddingItemPool<CacheItemHead, T> {
    public:
        CacheItemPool(size_t value_dim)
            : EmbeddingItemPool<CacheItemHead, T>(value_dim) {}

        ~CacheItemPool() {
            PersistentManager::singleton().release_cache(_acquired);
        }

        size_t item_memory_cost() {
            return this->item_size() + 16; // 16 for PersistentItemPool free space overhead
        }

        CacheItem* try_new_item() {
            if (!_free_items.empty()) {
                CacheItem* item = _free_items.back();
                _free_items.pop_back();
                return item;
            }
            if (_expanding) {
                if (PersistentManager::singleton().acquire_cache(item_memory_cost())) {
                    _acquired += item_memory_cost();
                    return this->new_item();
                } else {
                    _expanding = false;
                    SLOG(INFO) << "dram cache is full"
                               << ", cache size: " << _acquired
                               << ", number of cache items: "
                               << _acquired / item_memory_cost();
                }
            }
            return nullptr;
        }

        CacheItem* new_item() {
            ++_num_items;
            return EmbeddingItemPool<CacheItemHead, T>::new_item();
        }

        size_t num_items() {
            return _num_items;
        }

        void free_item(CacheItem* item) {
            _free_items.push_back(item);
        }

        bool expanding() {
            return _expanding;
        }

    private:
        std::deque<CacheItem*> _free_items;
        size_t _acquired = 0;
        size_t _num_items = 0;
        bool _expanding = true;
    };

    class PersistentItemPool: public EmbeddingItemPool<PersistentItemHead, T> {
    private:
        struct pmem_storage_type  {
            pmem::obj::segment_vector<pmem::obj::vector<char>,
                  pmem::obj::fixed_size_vector_policy<>> buf;
            //pmem::obj::persistent_ptr<pmem::obj::vector<T>> buf;
            pmem::obj::p<int64_t> checkpoint;
        };

        using storage_pool_t = pmem::obj::pool<pmem_storage_type>;
        struct space_item {
            int space_id;
            PersistentItem* pmem_item;
        };

    public:
        // max_pool_size: 单位G
        PersistentItemPool(size_t value_dim)
            :  EmbeddingItemPool<PersistentItemHead, T>(value_dim) {}

        PersistentItem* new_item() {
            SCHECK(_is_open);
            PersistentItem* pmem_item = nullptr;
            // TODO allocate pmem
            if (!_space_items.empty() && _space_items.front().space_id < _first_space_id) {
                // get space from _space_items
                pmem_item = _space_items.front().pmem_item;
                _space_items.pop_front();
            } else {
                // allocate new space at PMem
                pmem::obj::transaction::run(_storage_pool, [&] {
                    _storage_pool.root()->buf.emplace_back(this->item_size());
                    pmem_item = reinterpret_cast<PersistentItem*>(_storage_pool.root()->buf.back().data()); 
                    EmbeddingItemPool<PersistentItemHead, T>::construct(pmem_item);
                });
            }
            return pmem_item;
        }

        size_t num_items() {
            return _storage_pool.root()->buf.size();
        }

        void flush_item(PersistentItem* pmem_item) {
            clflush((char*)pmem_item, this->item_size());
        }

        void free_item(PersistentItem* pmem_item) {
            _space_items.push_front({-1, pmem_item});
        }

        void push_item(PersistentItem* pmem_item) {
            _space_items.push_back({_current_space_id, pmem_item});
        }

        const int64_t& get_checkpoint_batch_id() {
            return _storage_pool.root()->checkpoint.get_ro();
        }

        void push_checkpoint(int64_t work_id) {
            // requirement: only be called once after each checkpoint
            pmem::obj::transaction::run(_storage_pool, [&] {
                _storage_pool.root()->checkpoint = work_id;
            });

            // maintain the internal checkpoint counter
            ++_current_space_id;
        }

        void pop_checkpoint() {
            ++_first_space_id;
        }
        
        // for debug only
        uint64_t get_avaiable_freespace_slots() {
            uint64_t counter = 0;
            for (auto space: _space_items){
                if (space.space_id < _first_space_id){
                    ++counter;
                } else {
                    break;
                }
            }
            return counter;
        }

        uint64_t get_all_freespace_slots() {
            return _space_items.size();
        }

        bool open_pmem_pool(const std::string& pool_path, size_t max_pool_size = 700) {
            SCHECK(!_is_open);
            struct stat statBuff;
            std::string pool_set_path = pool_path + "/pool_set";
            if (stat(pool_set_path.c_str(), &statBuff) == 0) {
                //exist, recovery
                _storage_pool = storage_pool_t::open(pool_set_path, "layout");
                _is_open = true;
                return recovery();                
            } else {
                // new file, create file.
                std::string cmd = "mkdir -p ";
                cmd += pool_path;
                const int dir_err = system(cmd.c_str());
                if (-1 == dir_err) {
                    printf("Error creating directory!n");
                    exit(1);
                }
                std::ofstream outfile (pool_set_path);
                outfile << "PMEMPOOLSET" << std::endl;
                outfile << "OPTION SINGLEHDR" << std::endl;
                //outfile << "300G "+pool_path << std::endl;
                outfile << std::to_string(max_pool_size)+"G "+pool_path << std::endl;
                outfile.flush();
                outfile.close();
                _storage_pool = storage_pool_t::create(pool_set_path, "layout", 0, S_IWUSR | S_IRUSR);
                _is_open = true;
                return true;
            }
        }

        bool recovery() {
            ///TODO: scan & recovery process
            return true;
        }

    private:
        bool _is_open = false;
        storage_pool_t _storage_pool;
        std::deque<space_item> _space_items;
        int _current_space_id = 0;
        int _first_space_id = 0;
    };

    std::string _pmem_pool_path;
    std::deque<int64_t> _checkpoints;
    std::deque<int64_t> _pendings;

    // _train_batch_id will be dump and load as a configure property when changing variable type.
    // int64_t _train_batch_id = 0;
    int64_t _train_batch_id = 0;
    int64_t _work_id = 0; 
    int64_t _committing = 0;
    size_t _value_dim = 0;
    EasyHashMap<key_type, ItemPointer> _table;
    CacheItem* _cache_head = nullptr;

    CacheItemPool _cache_pool;
    PersistentItemPool _pmem_pool;

    size_t _hit_count = 0;
    size_t _set_count = 0;
    size_t _flush_count = 0;
};



}
}
}

#endif
