#ifndef PARADIGM4_HYPEREMBEDDING_PMEM_EMBEDDING_ITEM_POOL_H
#define PARADIGM4_HYPEREMBEDDING_PMEM_EMBEDDING_ITEM_POOL_H

#include <libpmem.h>
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

#include "PersistManager.h"

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
    // 16 for PmemItemPool free space overhead
    CacheItemPool(size_t value_dim)
        : _base_pool(value_dim),
          _item_memory_cost(_base_pool.item_size(_base_pool.value_dim()) + 16) {}

    ~CacheItemPool() {
        PersistManager::singleton().dynamic_cache.release_cache(
              (_acquired + _prefetched) * _item_memory_cost);
        PersistManager::singleton().reserved_cache.release_cache(
              (_reserved) * _item_memory_cost);
    }

    size_t item_memory_cost() {
        return _item_memory_cost;
    }

    CacheItem* try_new_item() {
        if (_reserved_acquired < _reserved) {
            _reserved_acquired++;
        }
        if (_expanding) {
            if (_prefetched == 0) {
                prefetch(PREFETCH);
            }
            if (_prefetched) {
                ++_acquired;
                --_prefetched;
                return this->new_item();
            } else {
                _expanding = false;
                SLOG(INFO) << "dram cache is full, cache size: "
                      << (_acquired + _reserved) * _item_memory_cost
                      << ", acquired cache items: " << _acquired
                      << ", reserved cache items: " << _reserved;
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
        _released = std::min(_released, _acquired);
        PersistManager::singleton().dynamic_cache.release_cache(_released * _item_memory_cost);
        _acquired -= _released;
        _expanding = true;
        _released = 0;
    }

    bool expanding() {
        return _expanding;
    }

    bool prefetch_reserve(size_t n) {
        if (_expanding && PersistManager::singleton().
              reserved_cache.acquire_cache(n * _item_memory_cost)) {
            _reserved += n;
            return true;
        }
        return false;
    }

private:
    bool prefetch(size_t n) {
        if (PersistManager::singleton().dynamic_cache.acquire_cache(n * _item_memory_cost)) {
            _prefetched += n;
            return true;
        }
        return false;
    }

    EmbeddingItemPool<Head, T> _base_pool;
    size_t _item_memory_cost = 0;
    size_t _prefetched = 0;
    size_t _acquired = 0;
    size_t _released = 0;
    size_t _num_items = 0;
    size_t _reserved = 0;
    size_t _reserved_acquired = 0;
    bool _expanding = true;
};

template<class Head, class T>
class PmemItemPool {
private:
    using PmemItem = typename EmbeddingItemPool<Head, T>::Item;
    struct pmem_storage_type  {
        pmem::obj::persistent_ptr<pmem::obj::segment_vector<
              pmem::obj::persistent_ptr<pmem::obj::vector<char>>,
              pmem::obj::fixed_size_vector_policy<>>> buffers;
        pmem::obj::persistent_ptr<pmem::obj::vector<int64_t>> checkpoints;
    };

    using storage_pool_t = pmem::obj::pool<pmem_storage_type>;
    struct space_item {
        int space_id = -1;
        PmemItem* pmem_item = nullptr;
    };

public:
    PmemItemPool(size_t value_dim): _value_dim(value_dim) {
        _item_size = EmbeddingItemPool<Head, T>::item_size(value_dim);
        if (_item_size > 64) {
            _item_size = ItemPoolAllocator::aligned_size(_item_size, 128);
        }
        _block_size = ItemPoolAllocator::aligned_size(64 * 1024, _item_size);
    }

    ~PmemItemPool() {
        if (!_pmem_pool_path.empty()) {
            _storage_pool.close();
            SLOG(INFO) << "close pmem pool";
        }
    }

    std::string pmem_pool_path() {
        return _pmem_pool_path;
    }

    PmemItem* new_item() {
        // TODO allocate pmem
        if (!_space_items.empty() && _space_items.front().space_id < _first_space_id) {
            // get space from _space_items
            PmemItem* pmem_item = _space_items.front().pmem_item;
            _space_items.pop_front();
            return pmem_item;
        } else {
            if (_pmem_pool_path.empty()) {
                SCHECK(create_pmem_pool());
            }
            // allocate new space at PMem
            pmem::obj::transaction::run(_storage_pool, [&] {
                _storage_pool.root()->buffers->emplace_back(
                      pmem::obj::make_persistent<pmem::obj::vector<char>>(_block_size));
                char* buffer = _storage_pool.root()->buffers->back()->data();
                for (size_t p = 0; p < _block_size; p += _item_size) {
                    PmemItem* pmem_item = reinterpret_cast<PmemItem*>(buffer + p); 
                    EmbeddingItemPool<Head, T>::construct(pmem_item, _value_dim);
                    free_item(pmem_item);
                }
            });
            return new_item();
        }
    }

    size_t num_items() {
        if (_item_size == 0 || _pmem_pool_path.empty()) {
            return 0;
        }
        return _storage_pool.root()->buffers->size() * _block_size / _item_size;
    }

    void flush_item(PmemItem* pmem_item) {
        pmem_flush(pmem_item, _item_size);
    }

    // not delete item, do not call destructor
    void free_item(PmemItem* pmem_item) {
        _space_items.push_front({-1, pmem_item});
    }

    void push_item(PmemItem* pmem_item) {
        _space_items.push_back({_current_space_id, pmem_item});
    }

    void push_checkpoint(int64_t work_id) {
        ++_current_space_id;
        _checkpoints.push_back(work_id);
        pmem_drain();
        pmem::obj::transaction::run(_storage_pool, [&] {
            _storage_pool.root()->checkpoints->push_back(work_id);
        });
        pmem_drain();
    }

    void pop_checkpoint() {
        ++_first_space_id;
        SCHECK(!_checkpoints.empty());
        _checkpoints.pop_front();
        pmem::obj::transaction::run(_storage_pool, [&] {
            _storage_pool.root()->checkpoints->erase(_storage_pool.root()->checkpoints->begin());
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
    bool create_pmem_pool(size_t max_pool_size = 700) {
        SCHECK(_pmem_pool_path.empty());
        struct stat statBuff;
        std::string pool_path = PersistManager::singleton().new_pmem_pool_path();
        std::string pool_set_path = pool_path + "/pool_set";
        if (stat(pool_set_path.c_str(), &statBuff) != 0) {
            // new file, create file.
            std::string cmd = "mkdir -p ";
            cmd += pool_path;
            const int dir_err = system(cmd.c_str());
            if (-1 == dir_err) {
                printf("Error creating directory!n");
                exit(1);
            }
            std::ofstream outfile(pool_set_path);
            outfile << "PMEMPOOLSET" << std::endl;
            outfile << "OPTION SINGLEHDR" << std::endl;
            //outfile << "300G "+pool_path << std::endl;
            outfile << std::to_string(max_pool_size)+"G "+pool_path << std::endl;
            outfile.flush();
            outfile.close();

            _storage_pool = storage_pool_t::create(pool_set_path, "layout", 0, S_IWUSR | S_IRUSR);
            SLOG(INFO) << "create pmem pool " << pool_set_path;
            if (_storage_pool.root()->buffers == nullptr) {
                pmem::obj::transaction::run(_storage_pool, [&]{
                    _storage_pool.root()->buffers = pmem::obj::make_persistent<pmem::obj::segment_vector<
                          pmem::obj::persistent_ptr<pmem::obj::vector<char>>,
                          pmem::obj::fixed_size_vector_policy<>>>();
                });
            }
            if (_storage_pool.root()->checkpoints == nullptr) {
                pmem::obj::transaction::run(_storage_pool, [&]{
                    _storage_pool.root()->checkpoints = pmem::obj::make_persistent<pmem::obj::vector<int64_t>>();
                    _storage_pool.root()->checkpoints->push_back(0);
                });
            }
            _pmem_pool_path = pool_path;
        }
        return !_pmem_pool_path.empty();
    }

    template<class EmbeddingIndex>
    bool load_pmem_pool(const std::string& pool_path,
          int64_t checkpoint, EmbeddingIndex& _table, size_t& _num_items) {
        SCHECK(_pmem_pool_path.empty());
        struct stat statBuff;
        std::string pool_set_path = pool_path + "/pool_set";
        if (stat(pool_set_path.c_str(), &statBuff) == 0) {
            _storage_pool = storage_pool_t::open(pool_set_path, "layout");
            SLOG(INFO) << "load pmem pool " << pool_set_path;
            bool found = false;
            std::string show_checkpoints = "[ ";
            for (int64_t point: *_storage_pool.root()->checkpoints) {
                show_checkpoints += std::to_string(point) + " ";
                if (point == checkpoint) {
                    found = true;
                }
            };
            show_checkpoints += "]";
            if (found) {
                SLOG(INFO) << "found checkpoint " 
                      << checkpoint << " in " << pool_path << " " << show_checkpoints;
            } else {
                SLOG(WARNING) << "not found checkpoint "
                      << checkpoint << " in " << pool_path << " " << show_checkpoints;
                _storage_pool.close();
                return false;
            }

            for (auto& buffer: *(_storage_pool.root()->buffers)) {
                SCHECK(buffer->size() == _block_size);
                for (size_t p = 0; p < _block_size; p += _item_size) {
                    PmemItem* pmem_item = reinterpret_cast<PmemItem*>(buffer->data() + p);
                    if (pmem_item->work_id < checkpoint && pmem_item->work_id != -1) {
                        auto& it = _table.set_pointer(pmem_item->key);
                        if (it) {
                            if (pmem_item->work_id > it.as_pmem_item()->work_id) {
                                free_item(it.as_pmem_item());
                                it = pmem_item;
                            } else {
                                free_item(pmem_item);
                            }
                        } else {
                            it = pmem_item;
                            ++_num_items;
                        }
                    } else {
                        free_item(pmem_item);
                    }
                }
            }
            _pmem_pool_path = pool_path;
        } else {
            SLOG(WARNING) << "not found pmem pool " << pool_set_path;
        }
        return !_pmem_pool_path.empty();
    }
private:
    std::string _pmem_pool_path;
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