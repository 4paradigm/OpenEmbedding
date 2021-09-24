#ifndef PARADIGM4_HYPEREMBEDDING_PERSIST_MANAGER_H
#define PARADIGM4_HYPEREMBEDDING_PERSIST_MANAGER_H

#include <pico-core/pico_log.h>
#include <pico-core/SpinLock.h>
#include <pico-core/FileSystem.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

class PersistManager {
    PersistManager() = default;
    PersistManager(const PersistManager&) = default;
public:
    class CacheManager {
    public:
        void initialize() {
            _cache_size.store(0);
            _acquired_size.store(0);
        }

        void set_cache_size(size_t cache_size) {
            _cache_size.store(cache_size);
        }

        bool acquire_cache(size_t size) {
            if (_acquired_size.fetch_add(size, std::memory_order_relaxed) + size >
                _cache_size.load(std::memory_order_relaxed)) {
                _acquired_size.fetch_sub(size, std::memory_order_relaxed);
                return false;
            }
            return true;
        }

        bool acquire_reserve_cache(size_t size) {
            if (3 * _acquired_size.load(std::memory_order_relaxed) <
                _cache_size.load(std::memory_order_relaxed)) {
                return acquire_cache(size);
            }
            return false;
        }

        void release_cache(size_t size) {
            _acquired_size.fetch_sub(size);
        }
    private:
        std::atomic<size_t> _cache_size = {0};
        std::atomic<size_t> _acquired_size = {0};
    };

    static PersistManager& singleton() {
        static PersistManager manager;
        return manager;
    }

    bool use_pmem() { // server & client
        return !_pmem_pool_root_path.empty();
    }

    void initialize(const std::string& path) {
        core::FileSystem::mkdir_p(path);
        _pmem_pool_root_path = path;
        _prefix = std::to_string(time(NULL)) + '-' + std::to_string(::getpid());
        _next_pool_id.store(0);
        reserved_cache.initialize();
        dynamic_cache.initialize();
    }

     std::string new_pmem_pool_path() {
        SCHECK(use_pmem());
        std::string name = std::to_string(_next_pool_id.fetch_add(1));
        while (name.size() < 6) name = "0" + name;
        return _pmem_pool_root_path + "/" + _prefix + "-" + name;
    }

    CacheManager reserved_cache;
    CacheManager dynamic_cache;
private:
    std::string _prefix;
    std::string _pmem_pool_root_path;
    std::atomic<size_t> _next_pool_id = {0};
};

}
}
}

#endif