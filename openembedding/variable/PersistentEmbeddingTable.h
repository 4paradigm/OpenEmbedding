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
#include <pico-core/RpcServer.h>

#include "persist.h"
#include "EmbeddingItemPool.h"
#include "EmbeddingTable.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

class PersistentManager {
    PersistentManager() = default;
    PersistentManager(const PersistentManager&) = default;
public:
    static PersistentManager& singleton() {
        static PersistentManager manager;
        return manager;
    }

    bool use_pmem() {
        return !_pmem_pool_root_path.empty();
    }

    void initialize(const std::string& path) {
        _pmem_pool_root_path = path;
        _next_pool_id.store(0);
        _cache_size.store(0);
        _acquired_size.store(0);
        _hint_checkpoint.store(0);
        _checkpoint.store(0);
    }

    void set_cache_size(size_t cache_size) {
        _cache_size.store(cache_size);
    }

    std::string new_pmem_pool_path() {
        std::string name = std::to_string(_next_pool_id.fetch_add(1));
        return _pmem_pool_root_path + "/" + name;
    }

    bool acquire_cache(size_t size) {
        if (_acquired_size.fetch_add(size, std::memory_order_relaxed) + size >
              _cache_size.load(std::memory_order_relaxed)) {
            _acquired_size.fetch_sub(size, std::memory_order_relaxed);
            return false;
        }
        return true;
    }

    void release_cache(size_t size) {
        _acquired_size.fetch_sub(size);
    }

    void hint_checkpoint(int64_t train_batch_id) {
        core::lock_guard<core::RWSpinLock> guard(_lock);
        int64_t checkpoint = _checkpoint.load(std::memory_order_relaxed);
        if (train_batch_id >= checkpoint) {
            _checkpoint.store(train_batch_id + 2, std::memory_order_relaxed);
        }
    }

    void set_checkpoint(int64_t train_batch_id) {
        core::lock_guard<core::RWSpinLock> guard(_lock);
        int64_t checkpoint = _checkpoint.load(std::memory_order_relaxed);
        if (train_batch_id < checkpoint || train_batch_id > checkpoint + 2) {
            _checkpoint.store(train_batch_id, std::memory_order_relaxed);
        }
    }

    int64_t checkpoint() {
        return _checkpoint.load(std::memory_order_relaxed);
    }
    
private:
    std::string _pmem_pool_root_path = "";
    std::atomic<size_t> _next_pool_id = {0};
    std::atomic<size_t> _cache_size = {0};
    std::atomic<size_t> _acquired_size = {0};
    
    core::RWSpinLock _lock;
    std::atomic<int64_t> _hint_checkpoint = {0};
    std::atomic<int64_t> _checkpoint = {0};
};

template<class Key, class T>
class PersistentEmbeddingTable: public EmbeddingTable<Key, T> {
public:
    using key_type = Key;
    static_assert(std::is_trivially_copyable<Key>::value, "persistent table need trivally copyable key type.");
    
    struct PersistentItemHead {
        int64_t batch_id = 0;
        key_type key = key_type();
    };

    struct CacheItemHead {
        using PersistentItem = typename EmbeddingItemPool<PersistentItemHead, T>::Item;
        using CacheItem = typename EmbeddingItemPool<CacheItemHead, T>::Item;
        int64_t batch_id = 0;
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
        _cache_head->batch_id = std::numeric_limits<int64_t>::max();
        _cache_head->prev = _cache_head->next = _cache_head;
        SCHECK(PersistentManager::singleton().use_pmem());
        SCHECK(_pmem_pool.open_pmem_pool(PersistentManager::singleton().new_pmem_pool_path()));
    }

    ~PersistentEmbeddingTable() {}

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
        ++_num_all;
        CacheItem* item = nullptr;
        auto it = _table.find(key);
        if (it != _table.end()) {
            if (it->second.is_cache_item()) {
                ++_num_hit;
                item = it->second.as_cache_item();
                if (item->batch_id < _committing) {
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
        item->batch_id = _batch_id;
        return item->data;
    }

    int64_t batch_id() {
        return _batch_id;
    }

    int64_t train_batch_id() {
        return _train_batch_id;
    }

    void next_batch() {
        ++_batch_id;
        if (!_pendings.empty() && _cache_head->next->batch_id >= _pendings.front()) {
            _checkpoints.push_back(_pendings.front());
            _pmem_pool.push_checkpoint(_pendings.front());
            _pendings.pop_front();
        }
    }

    void next_train_batch() {
        ++_train_batch_id;
        next_batch();
    }

    bool hint_to_commit_checkpoint() {
        return !_cache_pool.expanding() && _pendings.empty();
    }

    void start_commit_checkpoint() {  //trans start
        _committing = _batch_id;
        _pendings.push_back(_committing);
        SLOG(INFO) << "checkpoints " << show(_checkpoints)
                   << ", pending checkpoints " << show(_pendings)
                   << ", train batch id " << _train_batch_id
                   << ", hit rate " << 100 * _num_hit / _num_all 
                   << "%, flushed " << _num_flush << ", all " << _num_all;
    }

    void pop_checkpoint() {
        _checkpoints.pop_front();
        _pmem_pool.pop_checkpoint();
    }

    void flush_committing_checkpoint() {
        SCHECK(!_pendings.empty());
        SLOG(INFO) << "flush committing checkpoint " << _pendings.front();

        CacheItem* item = _cache_head->next;
        while (item != _cache_head && item->batch_id < _pendings.front()) {
            flush_to_pmem_item(item);
            item = item->next;
        }

        if (!_pendings.empty()) {
            _checkpoints.push_back(_pendings.front());
            _pmem_pool.push_checkpoint(_pendings.front()); 
            _pendings.pop_front();
        }
    }

    // for debug only
    size_t num_cache_items() {
        return _cache_pool.num_items();
    }

    uint64_t get_pmem_vector_size() {
        return _pmem_pool.get_pmem_vector_size();
    }
    uint64_t get_avaiable_freespace_slots(){
        return _pmem_pool.get_avaiable_freespace_slots();
    }
    uint64_t get_all_freespace_slots(){
        return _pmem_pool.get_all_freespace_slots();
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

    const std::deque<int64_t>& pending_checkpoints() {
        return _pendings;
    }

    size_t cache_item_memory_cost() {
        return _cache_pool.item_memory_cost();
    }

private:
    std::string show(const std::deque<int64_t>& vals) {
        std::string show_vals = "[";
        for (int64_t val: vals) {
            show_vals += std::to_string(val);
            show_vals += " ";
        }
        if (show_vals.back() == ' ') {
            show_vals.pop_back();
        }
        show_vals += "]";
        return show_vals;
    }


    void flush_to_pmem_item(CacheItem* item) {
        if (item->pmem == nullptr) {
            ++_num_flush;
            PersistentItem* pmem_item = _pmem_pool.new_item();
            pmem_item->key = item->key;
            pmem_item->batch_id = item->batch_id;
            std::copy_n(item->data, _value_dim, pmem_item->data);
            _pmem_pool.flush_item(pmem_item);
            item->pmem = pmem_item;
        }
    }

    CacheItem* cache_miss_new_item() {
        CacheItem* item = _cache_pool.try_new_item();
        if (item == nullptr) {
            item = _cache_head->next;
            if (item != _cache_head && item->batch_id < _batch_id) {
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
            if (_free_space.empty() || _first_space_id <= _free_space.front().space_id) {
                _free_space.emplace_front(-1);
            }
            _free_space.front().free_items.push_back(pmem_item);
        }

        void push_item(PersistentItem* pmem_item) {
            if (_free_space.empty() || _current_space_id != _free_space.back().space_id) {
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
        uint64_t get_pmem_vector_size() {
            return _storage_pool.root()->buf.size();
        }
        uint64_t get_avaiable_freespace_slots(){
            uint64_t counter = 0;
            for (auto space: _free_space){
                if (space.space_id < _first_space_id){
                    counter += space.free_items.size();
                } else {
                    break;
                }
            }
            return counter;
        }
        uint64_t get_all_freespace_slots(){
            uint64_t counter = 0;
            for (auto space: _free_space){
                counter += space.free_items.size();
            }
            return counter;
        }

        bool open_pmem_pool(const std::string& pool_path, size_t max_pool_size = 10) {
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
        std::deque<free_space_vec> _free_space;
        int _current_space_id = 0;
        int _first_space_id = 0;
    };

    std::string _pmem_pool_path;
    std::deque<int64_t> _checkpoints;
    std::deque<int64_t> _pendings;

    // _train_batch_id will be dump and load as a configure property when changing variable type.
    // int64_t _train_batch_id = 0;
    int64_t _train_batch_id = 0;
    int64_t _batch_id = 0; 
    int64_t _committing = 0;
    size_t _value_dim = 0;
    EasyHashMap<key_type, ItemPointer> _table;
    CacheItem* _cache_head = nullptr;

    CacheItemPool _cache_pool;
    PersistentItemPool _pmem_pool;

    size_t _num_hit = 0;
    size_t _num_all = 0;
    size_t _num_flush = 0;
};



}
}
}

#endif
