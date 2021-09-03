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
#include "EmbeddingTable.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class PersistentManager {
public:
    static PersistentManager& singleton() {
        static PersistentManager manager;
        return manager;
    }

    bool use_pmem() {
        return _pmem_pool_root_path.empty();
    }

    void set_pmem_pool_root_path(const std::string& path) {
        _pmem_pool_root_path = path;
    }

    std::string new_pmem_pool_path() {
        std::string name = std::to_string(_pool_id.fetch_add(1));
        return _pmem_pool_root_path + "/" + name;
    }

    void set_cache_size(size_t max_size) {
        _max_size = max_size;
    }

    bool acquire_cache(size_t size) {
        if (_size.fetch_add(size) + size < _max_size) {
            return true;
        } else {
            _size.fetch_sub(size);
            return false;
        }
    }

    void release_cache(size_t size) {
        _size.fetch_sub(size);
    }

private:
    std::atomic<size_t> _pool_id = {0};
    std::string _pmem_pool_root_path = "";
    std::atomic<size_t> _max_size = {0};
    std::atomic<size_t> _size = {0};
};

template<class Key, class T>
class PersistentEmbeddingTable: public EmbeddingTable<Key, T> {
public:
    using key_type = Key;
    static_assert(std::is_trivially_copyable<Key>::value, "persistent table need trivally copyable key type.");
    
    struct PersistentItemHead {
        int64_t batch_id;
        key_type key;
        T data[1];
    };

    struct CacheItemHead {
        using CacheItem = typename EmbeddingItemPool<CacheItemHead, T>::Item;
        int64_t batch_id;
        key_type key;
        CacheItem* next;
        CacheItem* prev;
        T data[1];

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
        typename EasyHashMap<key_type, uintptr_t>::iterator _it, _end;
    };

    PersistentEmbeddingTable(size_t value_dim, key_type empty_key)
        : _value_dim(value_dim), _table(empty_key), _pool(value_dim), _pmem_pool(value_dim) {
        _cache_head = _pool.force_new_item();
        _cache_head->prev = _cache_head->next = _cache_head;
    }

    ~PersistentEmbeddingTable() {}

    void load_config(const core::Configure& config) override {
        EmbeddingTable<Key, T>::load_config(config);
        std::string pmem_pool_path = _pmem_pool_path;
        LOAD_CONFIG(config, pmem_pool_path);
        if (pmem_pool_path != _pmem_pool_path) {
            SCHECK(!pmem_pool_path.empty()) << "Should not be empty.";
            _pmem_pool_path = pmem_pool_path;
            SCHECK(_pmem_pool.open_pmem_pool(_pmem_pool_path));
        }
    }

    void dump_config(core::Configure& config)const override {
        EmbeddingTable<Key, T>::dump_config(config);
        std::string pmem_pool_path = _pmem_pool_path;
        SAVE_CONFIG(config, pmem_pool_path);
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
        uintptr_t p = it->second;
        if (p & 1) {
            CacheItem* item = reinterpret_cast<CacheItem*>(p ^ 1);
            return item->data;
        } else {
            PersistentItem* pmem_item = reinterpret_cast<PersistentItem*>(p);
            return pmem_item->data;                
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
                    _pmem_pool.push_item(flush_to_pmem_item(item));
                }
                item->erase();
            } else {
                PersistentItem* pmem_item = reinterpret_cast<PersistentItem*>(p);
                // batch id read from pmem, 
                if (_pendings.empty()) {
                    _pmem_pool.free_item(pmem_item);
                } else {
                    _pmem_pool.push_item(pmem_item);
                }
            }
        }
        if (item == nullptr) {
            item = _pool.try_new_item();
        }
        if (item == nullptr) {
            item = _cache_head->next;
            if (item != _cache_head && item->batch_id < _batch_id) {
                item->erase();
                _table.at(item->key) = reinterpret_cast<uintptr_t>(flush_to_pmem_item(item));
            } else {
                item = _pool.force_new_item();
            }
        }

        _cache_head->insert_prev(item);
        if (it == _table.end()) {
            _table.force_emplace(key, reinterpret_cast<uintptr_t>(item) | 1);
        } else {
            it->second = reinterpret_cast<uintptr_t>(item) | 1;
        }
        item->key = key;
        item->batch_id = _batch_id;

        
        if (!_pendings.empty() && _cache_head->next->batch_id >= _pendings.front()) {
            _checkpoints.push_back(_pendings.front());
            _pmem_pool.push_checkpoint(_pendings.front()); 
            _pendings.pop();
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
        if (_committing != -1) {
            _pendings.push(_committing);
        } 
        _committing = _batch_id;
    }

    void pop_checkpoint() {
        _checkpoints.pop_front();
        _pmem_pool.pop_checkpoint();
    }

    void flush_committing_checkpoint() {
        /// TODO;
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
    PersistentItem* flush_to_pmem_item(CacheItem* item) {
        PersistentItem* pmem_item = _pmem_pool.new_item();
        pmem_item->key = item->key;
        pmem_item->batch_id = item->batch_id;
        memcpy(pmem_item->data, item->data, _value_dim);
        _pmem_pool.flush_item(pmem_item);
        return pmem_item;
    }

    // not thread safe
    class CacheItemPool: public EmbeddingItemPool<CacheItemHead, T> {
    public:
        CacheItemPool(size_t value_dim)
            : EmbeddingItemPool<CacheItemHead, T>(value_dim) {}

        CacheItem* try_new_item() {
            if (_expanding && PersistentManager::singleton().acquire_cache(this->item_size() + 16)) {
                return this->new_item();
            }
            _expanding = false;
            return nullptr;
        }

        CacheItem* force_new_item() {
            return this->new_item();
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
            int space_id = -1;
            core::vector<PersistentItem*> free_items;
            free_space_vec(int space_id): space_id(space_id) {}
        };
    public:
        // max_pool_size: 单位G
        PersistentItemPool(size_t value_dim)
            :  EmbeddingItemPool<PersistentItemHead, T>(value_dim) {}

        PersistentItem* new_item() {
            SCHECK(_is_open);
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
                EmbeddingItemPool<PersistentItemHead, T>::construct(pmem_item);
            }
            return pmem_item;
        }

        void flush_item(PersistentItem* pmem_item) {
            clflush((char*)pmem_item, this->item_size());
        }

        void free_item(PersistentItem* pmem_item) {
            if (_free_space.empty()) {
                _free_space.emplace_front(-1);
            }
            _free_space.front().free_items.push_back(pmem_item);
        }

        void push_item(PersistentItem* pmem_item) {
            if (_free_space.empty()) {
                _free_space.emplace_back(_current_space_id);
            }
            _free_space.back().free_items.push_back(pmem_item);
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
            ++_current_space_id;
        }

        void pop_checkpoint() {
            ++_first_space_id;
        }
        
        // for debug only
        std::queue<free_space_vec>& debug_get_free_space() {
            return _free_space;
        }

        bool open_pmem_pool(const std::string& pool_path, size_t max_pool_size = 10) {
            SCHECK(!_is_open);
            struct stat statBuff;
            std::string pool_set_path = pool_path + "/pool_set";
            if (stat(pool_set_path.c_str(), &statBuff) == 0) {
                //exist, recovery
                _storage_pool = storage_pool_t::open(pool_set_path, "layout");
                return recovery();                
            } else {
                // new file, create file.
                std::string cmd = "mkdir -p ";
                cmd += pool_path;
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
                _is_open = true;
                return true;
            }
        }

        bool recovery(){
            ///TODO: scan & recovery process
            return true;
        }

    private:
        bool _is_open = false;
        storage_pool_t _storage_pool;
        std::deque<free_space_vec> _free_space;
        int _current_space_id = 0;
        int _first_space_id = 0;
    };

    std::string _pmem_pool_path;
    std::deque<int64_t> _checkpoints;
    std::queue<int64_t> _pendings;

    int64_t _batch_id = 0;
    int64_t _committing = -1;
    size_t _value_dim = 0;
    EasyHashMap<key_type, uintptr_t> _table;
    CacheItem* _cache_head;

    CacheItemPool _pool;
    PersistentItemPool _pmem_pool;
};



}
}
}

#endif
