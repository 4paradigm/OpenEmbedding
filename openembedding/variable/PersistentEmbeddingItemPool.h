#ifndef PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_ITEM_POOL_H
#define PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_ITEM_POOL_H

#include <libpmemobj++/pool.hpp>
#include <libpmemobj++/p.hpp>
#include <libpmemobj++/make_persistent.hpp>
#include <libpmemobj++/transaction.hpp>
#include <libpmemobj++/persistent_ptr.hpp>
#include <libpmemobj++/container/string.hpp>
#include <libpmemobj++/container/vector.hpp>
#include <libpmemobj++/container/segment_vector.hpp>

#include <sys/stat.h>
#include <string>
#include <queue>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <unistd.h>


#include "persist.h"
#include "PersistentManager.h"

#include "EmbeddingItemPool.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

// not thread safe
template<class Head, class T>
class CacheItemPool {
    enum { PREFETCH = 64 };
public:
    using CacheItem = typename EmbeddingItemPool<Head, T>::Item;
    // 16 for PersistentItemPool free space overhead
    CacheItemPool(size_t value_dim)
        : _base_pool(value_dim),
          _item_memory_cost(_base_pool.item_size(_base_pool.value_dim()) + 16) {}

    ~CacheItemPool() {
        PersistentManager::singleton().release_cache(_acquired);
    }

    size_t item_memory_cost() {
        return _item_memory_cost;
    }

    CacheItem* try_new_item() {
        if (_expanding) {
            if (_prefetched == 0) {
                if (PersistentManager::singleton().acquire_cache(PREFETCH * _item_memory_cost)) {
                    _prefetched = PREFETCH;
                } else if (PersistentManager::singleton().acquire_cache(_item_memory_cost)) {
                    _prefetched = 1;
                }
            }
            if (_prefetched) {
                ++_acquired;
                --_prefetched;
                return this->new_item();
            } else {
                _expanding = false;
                SLOG(INFO) << "dram cache is full"
                            << ", cache size: " << _acquired * _item_memory_cost
                            << ", number of cache items: " << _acquired;
            }
        }
        return nullptr;
    }

    CacheItem* new_item() {
        ++_num_items;
        return _base_pool.new_item();
    }

    size_t num_items() {
        return _num_items;
    }

    void delete_item(CacheItem* item) {
        --_num_items;
        _released++;
        _base_pool.delete_item(item);
    }

    void rebalance() {
        PersistentManager::singleton().release_cache(_released * _item_memory_cost);
        _acquired -= _released;
        _expanding = true;
        _released = 0;
    }

    bool expanding() {
        return _expanding;
    }

private:
    EmbeddingItemPool<Head, T> _base_pool;
    size_t _item_memory_cost = 0;
    size_t _prefetched = 0;
    size_t _acquired = 0;
    size_t _released = 0;
    size_t _num_items = 0;
    bool _expanding = true;
};

template<class Head, class T>
class PersistentItemPool {
private:
    using PersistentItem = typename EmbeddingItemPool<Head, T>::Item;
    
    struct pmem_storage_type  {
        pmem::obj::segment_vector<pmem::obj::vector<char>,
                pmem::obj::fixed_size_vector_policy<>> buf;
        //pmem::obj::persistent_ptr<pmem::obj::vector<T>> buf;
        pmem::obj::vector<int64_t> checkpoints;
    };

    using storage_pool_t = pmem::obj::pool<pmem_storage_type>;
    struct space_item {
        int space_id = -1;
        PersistentItem* pmem_item = nullptr;
    };

public:
    PersistentItemPool(size_t value_dim): _value_dim(value_dim) {
        _item_size = EmbeddingItemPool<Head, T>::item_size(value_dim);
        if (_item_size >= 64) {
            _item_size = ItemPoolAllocator::aligned_size(_item_size, 128);
        }
        _block_size = ItemPoolAllocator::aligned_size(64 * 1024, _item_size);
    }

    PersistentItem* new_item() {
        SCHECK(_is_open);
        // TODO allocate pmem
        if (!_space_items.empty() && _space_items.front().space_id < _first_space_id) {
            // get space from _space_items
            PersistentItem* pmem_item = _space_items.front().pmem_item;
            _space_items.pop_front();
            return pmem_item;
        } else {
            // allocate new space at PMem
            pmem::obj::transaction::run(_storage_pool, [&] {
                _storage_pool.root()->buf.emplace_back(_block_size);
                char* buffer = _storage_pool.root()->buf.back().data();
                for (size_t p = 0; p < _block_size; p += _item_size) {
                    PersistentItem* pmem_item = reinterpret_cast<PersistentItem*>(buffer + p); 
                    EmbeddingItemPool<Head, T>::construct(pmem_item, _value_dim);
                    free_item(pmem_item);
                }
            });
            return new_item();
        }
    }

    size_t num_items() {
        return _storage_pool.root()->buf.size() * _block_size / _item_size;
    }

    void flush_item(PersistentItem* pmem_item) {
        clflush((char*)pmem_item, _item_size);
    }

    // not delete item, do not call destructor
    void free_item(PersistentItem* pmem_item) {
        _space_items.push_front({-1, pmem_item});
    }

    void push_item(PersistentItem* pmem_item) {
        _space_items.push_back({_current_space_id, pmem_item});
    }

    void push_checkpoint(int64_t work_id) {
        ++_current_space_id;
        _checkpoints.push_back(work_id);
        pmem::obj::transaction::run(_storage_pool, [&] {
            _storage_pool.root()->checkpoints.push_back(work_id);
        });
    }

    void pop_checkpoint() {
        ++_first_space_id;
        SCHECK(!_checkpoints.empty());
        _checkpoints.pop_front();
        pmem::obj::transaction::run(_storage_pool, [&] {
            _storage_pool.root()->checkpoints.erase(_storage_pool.root()->checkpoints.begin());
        });
    }

    const std::deque<int64_t>& checkpoints() {
        return _checkpoints;
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

    // max_pool_size GB
    bool open_pmem_pool(const std::string& pool_path, size_t max_pool_size = 700) {
        SCHECK(!_is_open);
        struct stat statBuff;
        std::string pool_set_path = pool_path + "/pool_set";
        if (stat(pool_set_path.c_str(), &statBuff) == 0) {
            //exist, recovery
            _storage_pool = storage_pool_t::open(pool_set_path, "layout");
            _is_open = true;
            return false;
            //return recovery();                
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

    template<class key_type, class ItemPointer>
    bool recovery(int64_t checkpoint, EasyHashMap<key_type, ItemPointer>& _table) {
        bool has = false;
        for (int64_t check: _storage_pool.root()->checkpoints) {
            if (checkpoint == check) {
                has = true;
            }
        }
        if (has) {
            for (auto& buffer: _storage_pool.root()->buf) {
                SCHECK(buffer.size() == _block_size);
                for (size_t p = 0; p < _block_size; p += _item_size) {
                    PersistentItem* pmem_item = reinterpret_cast<PersistentItem*>(buffer + p);
                    if (pmem_item->work_id < checkpoint && pmem_item->work_id != -1) {
                        auto it = _table.find(pmem_item->key);
                        if (it == _table.end()) {
                            _table.force_emplace(pmem_item->key, pmem_item);
                        } else {
                            if (pmem_item->work_id > it->second->work_id) {
                                free_item(it->second.as_persistent_item());
                                it->second = pmem_item;
                            } else {
                                free_item(pmem_item);
                            }
                        }
                    } else {
                        free_item(pmem_item);
                    }
                }
            }
        } else {
            SLOG(WARNING) << "not found checkpoint " << checkpoint << " in pmem";
        }
        return true;
    }

private:
    bool _is_open = false;
    size_t _value_dim = 0;
    size_t _item_size = 0;
    size_t _block_size = 0;
    storage_pool_t _storage_pool;
    std::deque<space_item> _space_items;
    std::deque<int64_t> _checkpoints;
    int _current_space_id = 0;
    int _first_space_id = 0;
};

}
}
}

#endif