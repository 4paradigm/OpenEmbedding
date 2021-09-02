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
//#include <libpmemobj++/container/concurrent_hash_map.hpp>

#include <sys/stat.h>
#include <string>
#include <queue>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>
#include "persist.h"
#include "EmbeddingItemPool.h"

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
    std::atomic<size_t> _max_size = {0};
    std::atomic<size_t> _size = {0};
};



template<class Key, class T>
class PersistentEmbeddingTable {
    static_assert(std::is_trivially_copyable<Key>::value, "persistent table need trivally copyable key type.");
    
    struct PersistentItemHead {
        int64_t batch_id;
        key_type key;
    };

    struct CacheItemHead {
        int64_t batch_id;
        key_type key;
        CacheItem* next;
        CacheItem* prev;
    };
    
    using PersistentItem = typename EmbeddingItemPool<PersistentItemHead, T>::Item;
    using CacheItem = typename EmbeddingItemPool<CacheItem, T>::Item;
    
public:
    using key_type = Key;
    PersistentEmbeddingTable(size_t value_size, key_type empty_key, const std::string& pool_path)
        : _table(empty_key), _value_size(value_size), _pool(value_size), _pmem_pool(value_size, pool_path) {
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
                if (item->batch_id < _committing) {
                    _pmem_pool.push_item(flush_item(item));
                }
                item->erase();
            } else {
                _pmem_pool.push_item(reinterpret_cast<PersistentItem*>(p));
            }
        }
        if (item == nullptr) {
            item = _pool.new_item();
        }
        if (item == nullptr) {
            item = _cache_head.next;
            if (item != &_cache_head && item->batch_id < _batch_id) {
                item->erase();
                _table.at(item->key) = reinterpret_cast<uintptr_t>(flush_item(item));
            } else {
                item = _pool.force_acquire();
            }
        }

        _cache_head.insert(item);
        if (it == _table.end()) {
            _table.force_emplace(key, reinterpret_cast<uintptr_t>(item) | 1);
        } else {
            it->second = reinterpret_cast<uintptr_t>(item) | 1;
        }
        item->key = key;
        item->batch_id = _batch_id;

        if (_committing != -1 && _cache_head.next->batch_id >= _committing) {
            _checkpoints.push_back(_committing);   //trans stop
            _pmem_pool.push_checkpoint(_committing);   //finished checkpointed id

            if (_pending.empty()) {
                _committing = -1;
            } else {
                _committing = _pending.front();
                _pending.pop();
            }
        }
        return item->data;
    }

    int64_t batch_id() {
        return _batch_id;
    }

    void next_batch() {
        ++_batch_id; 
    }

    bool hint_to_commit_checkpoint() {
        return !_pool.expanding() && _committing == -1;
    }

    void start_commit_checkpoint() {  //trans start
        if (_committing == -1) {
            _committing = _batch_id;
        } else {
            _pending.push(_batch_id);
        }
    }

    void pop_checkpoint() {
        _checkpoints.pop_front();
        _pmem_pool.pop_checkpoint();
        
    }

    int64_t flush_committing_checkpoint() {
        /// TODO
        int64_t batch_id = _committing;
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
    PersistentItem* flush_item(CacheItem* item) {
        PersistentItem* pmem_item = _pmem_pool.new_item()
        pmem_item->key = item->key;
        pmem_item->batch_id = item->batch_id;
        memcpy(pmem_item->data, item->data, _value_size);
        _pmem_pool.flush_item(pmem_item);
        return pmem_item;
    }

    // not thread safe
    class CacheItemPool: public EmbeddingItemPool<CacheItemHead, T> {
    public:
        CacheItemPool(size_t value_dim)
            : EmbeddingItemPool<CacheItemHead, T>(value_dim) {}

        CacheItem* new_item() {
            if (_expanding && CacheMemoryManager::singleton().new_item(this->item_size() + OVERHEAD)) {
                return force_new_item();
            }
            _expanding = false;
            return nullptr;
        }

        CacheItem* force_new_item() {
            EmbeddingItemPool<CacheItemHead, T>::new_item();
        }

        bool expanding() {
            return _expanding;
        }
    private:
        std::deque<core::vector<T>> _pool;
        bool _expanding = true;
    };

    class PersistentItemPool: public EmbeddingItemPool<PersistentItemHead, T> {
    private:
        struct pmem_storage_type  {
            pmem::obj::vector<pmem::obj::vector<T>> buf;
            //pmem::obj::persistent_ptr<pmem::obj::vector<T>> buf;
            pmem::obj::p<int64_t> checkpoint;
        };
        using storage_pool_t = pmem::obj::pool<pmem_storage_type>;
        struct free_space_vec {
            size_t space_id;
            core::vector<PersistentItem*> free_items;
        };
    public:
        // max_pool_size: 单位G
        PersistentItemPool(size_t value_dim, const std::string& pool_path, size_t max_pool_size = 700)
            :  EmbeddingItemPool<PersistentItemHead, T>(value_dim) {
            if (!open_pmem_pool(pool_path, max_pool_size)) {
                SLOG(FATAL) << "open_pmem_pool Error!";
            }
        }

        PersistentItem* new_item() {
            PersistentItem* pmem_item = nullptr;
            // TODO allocate pmem
            if (!_free_space.empty() && _free_space.front().space_id < _first_space_id) {
                // get space from _free_space
                pmem_item = _free_space.front().free_items.back();
                _free_space.front().free_items.pop_back();
                if (_free_space.front().free_items.empty()) {
                    _free_space.pop_front();
                }
            } else {
                // allocate new space at PMem
                _storage_pool.root()->buf.emplace_back(this->item_size());
                pmem_item = reinterpret_cast<PersistentItem*>(_storage_pool.root()->buf.back().data()); 
            }
            EmbeddingItemPool<PersistentItemHead, T>::contruct(pmem_item);
            return pmem_item;
        }

        void flush_item(PersistentItem* pmem_item) {
            clflush((char*)pmem_item, this->item_size());
        }

        void push_item(PersistentItem* pmem_item) {
            if (_free_space.empty()) {
                free_space_vec new_vec;
                new_vec.space_id = _next_space_id;
                _free_space.push(std::move(new_vec));
            }
            SCHECK(_free_space.back().space_id == _next_space_id);
            _free_space.back().free_items.emplace_back(pmem_item);
        }

        const int64_t& get_checkpoint_batch_id() {
            return _storage_pool.root()->checkpoint.get_ro();
        }

        void push_checkpoint(int64_t batch_id) {
            // requirement: only be called once after each checkpoint
            pmem::obj::transaction::run(_storage_pool, [&] {
                _storage_pool.root()->checkpoint = batch_id;
            });

            // maintain the internal checkpoint counter
            ++_next_space_id;
        }

        void pop_checkpoint() {
            ++_first_space_id;
        }
        
        // for debug only
        std::queue<free_space_vec>& debug_get_free_space() {
            return _free_space;
        }

    private:
        bool open_pmem_pool(const std::string& pool_path, size_t& max_pool_size) {
            struct stat statBuff;
            std::string pool_set_path = pool_path + "/pool_set";
            if (stat(pool_set_path.c_str(), &statBuff) == 0) {
                //exist, recovery
                _storage_pool = storage_pool_t::open(pool_set_path, "layout");
                return recovery();                
            } else {
                // new file, create file.
                std::string cmd = "mkdir -p ";
                cmd += pool_set_path;
                const int dir_err = system(cmd.c_str());
                if (-1 == dir_err)
                {
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
                return true;
            }
        }

        bool recovery(){
            ///TODO: scan & recovery process
            return true;
        }

        storage_pool_t _storage_pool;
        std::deque<free_space_vec> _free_space;
        size_t _next_space_id = 0;
        size_t _first_space_id = 0;
    };

    std::deque<int64_t> _checkpoints;
    std::queue<int64_t> _pending;

    int64_t _batch_id = 0;
    int64_t _committing = -1;
    size_t _value_size = 0;
    EasyHashMap<key_type, uintptr_t> _table;
    CacheItem _cache_head;

    CacheItemPool _pool;
    PersistentMemoryPool _pmem_pool;
};



}
}
}

#endif
