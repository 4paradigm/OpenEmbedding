#ifndef PARADIGM4_HYPEREMBEDDING_OBJECT_POOL_H
#define PARADIGM4_HYPEREMBEDDING_OBJECT_POOL_H

#include <thread>
#include <algorithm>
#include <pico-ps/handler/PullHandler.h>
#include <pico-ps/handler/PushHandler.h>
#include "EmbeddingVariable.h"

namespace paradigm4 {
namespace pico {
namespace embedding {


template<class T>
class ObjectPool {
public:
    ObjectPool() {}
    ObjectPool(ObjectPool<T>&&) = default;
    ObjectPool<T>& operator=(ObjectPool<T>&&) = default;
    ObjectPool<T>& operator=(std::function<T()> initializer) {
        SCHECK(_initializer == nullptr);
        _initializer = initializer;
        return *this;
    }

    T acquire() {
        core::lock_guard<core::RWSpinLock> lk(*_lock);
        if (_pool.empty()) {
            if (_initializer) {
                return _initializer();
            } else {
                return nullptr;
            }
        } else {
            T p = std::move(_pool.back());
            _pool.pop_back();
            return p;
        }
    }

    void release(T&& p) {
        core::lock_guard<core::RWSpinLock> lk(*_lock);
        _pool.push_back(std::move(p));
    }

    void clear() {
        core::lock_guard<core::RWSpinLock> lk(*_lock);
        _pool.clear();
    }

    std::unique_ptr<core::RWSpinLock> _lock = std::make_unique<core::RWSpinLock>();
    std::function<T()> _initializer;
    std::deque<T> _pool;
};



}
}
}


#endif