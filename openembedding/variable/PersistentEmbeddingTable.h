#ifndef PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_VARIABLE_H
#define PARADIGM4_HYPEREMBEDDING_PERSISTENT_EMBEDDING_VARIABLE_H

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
    static_assert(std::is_trivially_copyable<Key>::value, "persistent table need trivally copyable key type.");
private:
    using key_type = Key;
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


public:
    PersistentEmbeddingTable(size_t value_size, key_type empty_key, std::string pool_path)
        : _table(empty_key), _value_size(value_size), _pool(value_size), _pmem_pool(value_size, std::move(pool_path)) {
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
            _checkpoints.push_back(_submitting);   //trans stop
            _pmem_pool.pmem_push_checkpoint(_submitting);   //finished checkpointed id

            if (_pending.empty()) {
                _submitting = -1;
            } else {
                _submitting = _pending.front();
                _pending.pop();
            }
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
        return !_pool.expanding() && _submitting == -1;
    }

    void push_checkpoint() {  //trans start
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
        memcpy(pmem_item->data, item->data, _value_size);
        _pmem_pool.flush(pmem_item);
        return pmem_item;
    }

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
    private:
        struct pmem_storage_type  {
            pmem::obj::vector<pmem::obj::vector<T>> buf;
            //pmem::obj::persistent_ptr<pmem::obj::vector<T>> buf;
            pmem::obj::p<int64_t> global_cp_version;
        };
        using storage_pool_t = pmem::obj::pool<pmem_storage_type>;
        struct free_space_vec{
            size_t id;
            core::vector<PersistentItem*> free_items;
        };
    public:
        //max_pool_size: 单位G
        PersistentMemoryPool(size_t value_size, std::string pool_path, size_t max_pool_size=700)
            : _item_size((value_size + sizeof(CacheItem)) / ALIGN * ALIGN) {
            if(!open_pmem_pool(pool_path, max_pool_size)){
                SLOG(ERROR)<<"open_pmem_pool Error!";
            }
        }

        PersistentItem* acquire(const key_type& key) {
            PersistentItem* pmem_item; 
            // TODO allocate pmem
            if(_free_space.empty() || _free_space.front().id > _released_version || 0 == _free_space.front().free_items.size()){
                // allocate new space at PMem
                _storage_pool.root()->buf.emplace_back(_item_size);
                pmem_item = reinterpret_cast<PersistentItem*>(_storage_pool.root()->buf.back().data()); 
            }else{
                // get space from _free_space
                pmem_item = _free_space.front().free_items.back();
                _free_space.front().free_items.pop_back();
            }
            _key_constructor.construct(pmem_item->key, key); // pmem_item->key = key
            return pmem_item;
        }

        void flush(PersistentItem* pmem_item) {
            clflush((char*)pmem_item, _item_size);
        }

        void release(PersistentItem* pmem_item) {
            if(_free_space.empty()){
                free_space_vec new_vec;
                new_vec.id = _checkpointing_version;
                _free_space.push(std::move(new_vec));
            }
            SCHECK(_free_space.front().id == _checkpointing_version);
            _free_space.front().free_items.emplace_back(std::move(pmem_item));
        }

        const int64_t& get_pmem_checkpointed_id(){
            return _storage_pool.root()->global_cp_version.get_ro();
        }

        void pmem_push_checkpoint(int64_t _completed_checkpoint) {
            // requirement: only be called once after each checkpoint
            pmem::obj::transaction::run(_storage_pool, [&] {
                _storage_pool.root()->global_cp_version = _completed_checkpoint;
            });

            // maintain the internal checkpoint counter
            ++_checkpointing_version;
            free_space_vec new_vec();
            new_vec.id = _checkpointing_version;
            _free_space.push(std::move(new_vec));
        }

        void pmem_pop_checkpoint(){
            ++_released_version;
        }
        
        // for debug only
        std::queue<free_space_vec>& debug_get_free_space(){
            return _free_space;
        }

    private:
        bool open_pmem_pool(const std::string& pool_path, size_t& max_pool_size){
            struct stat statBuff;
            std::string pool_set_path = pool_path + "/pool_set";
            if (stat(pool_set_path.c_str(), &statBuff) == 0) {
                //exist, recovery
                _storage_pool = storage_pool_t::open(pool_set_path, "layout");
                return recovery();                
            }else{
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

        size_t _item_size = 0;
        std::allocator<key_type> _key_constructor;
        storage_pool_t _storage_pool;
        //std::queue<core::vector<PersistentItem*>> _free_space;
        std::queue<free_space_vec> _free_space;
        size_t _checkpointing_version = 1;  //? 初始化值和上层调用方式有关，如果
        size_t _released_version = 0;
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
