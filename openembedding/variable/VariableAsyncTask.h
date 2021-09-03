#ifndef PARADIGM4_HYPEREMBEDDING_ASYNC_OPERATOR_THREAD_POOL_H
#define PARADIGM4_HYPEREMBEDDING_ASYNC_OPERATOR_THREAD_POOL_H

#include <pico-core/SpinLock.h>
#include <pico-core/RpcChannel.h>
#include <thread>

namespace paradigm4 {
namespace pico {
namespace embedding {

class VariableAsyncTask {
public:
    static void wait(std::atomic<size_t>& _counter) {
        for (int collisions = 0;; ++collisions) {
            for (int tests = 0; unlikely(_counter.load(std::memory_order_acquire) != 0); ++tests) {
                if (tests < 128) {
                    cpu_relax();
                } else {
                    static constexpr std::chrono::microseconds us0{0};
                    std::this_thread::sleep_for(us0);
                }
            }
        }
    }

    VariableAsyncTask() {}
    VariableAsyncTask(const VariableAsyncTask&) = delete;
    
    VariableAsyncTask(VariableAsyncTask&& other)
        : thread_id(other.thread_id),
          shard_lock(other.shard_lock),
          done(std::move(other.done)),
          _storage_lock(other._storage_lock),
          _counter(other._counter) {
        other.shard_lock = nullptr;
        other._storage_lock = nullptr;
        other._counter = nullptr;
    }

    VariableAsyncTask& operator=(VariableAsyncTask other) {
        this->~VariableAsyncTask();
        new (this) VariableAsyncTask(std::move(other));
        return *this;
    }
    
    ~VariableAsyncTask() {
        if (_storage_lock) {
            _storage_lock->unlock_shared();
        }
        _counter->fetch_sub(1, std::memory_order_release);
    }

    void async_init(core::RWSpinLock& storage_lock, std::atomic<size_t>& counter) {
        _storage_lock = &storage_lock;
        _counter = &counter;
        _storage_lock->lock_shared();
        _counter->fetch_add(1, std::memory_order_relaxed);
    }

    size_t thread_id = 0;
    core::RWSpinLock* shard_lock = nullptr;
    std::function<void()> done; 
private:
    core::RWSpinLock* _storage_lock = nullptr;
    std::atomic<size_t>* _counter = nullptr;
};

class VariableAsyncTaskThreadPool {
public:
    std::atomic<size_t> thread_id = {0};
    static VariableAsyncTaskThreadPool& singleton() {
        static VariableAsyncTaskThreadPool pool;
        return pool; 
    }

    void submit(VariableAsyncTask&& task) {
        if (task.done) {
            _channels[task.thread_id % _threads.size()]->send(std::move(task));
        }
    }

private:
    VariableAsyncTaskThreadPool(size_t thread_num = 4): _threads(thread_num), _channels(thread_num) {
        for (size_t i = 0; i < _threads.size(); ++i) {
            _channels[i] = std::make_unique<core::RpcChannel<VariableAsyncTask>>();
            _threads[i] = std::thread(&VariableAsyncTaskThreadPool::running, this, i);
        }
    }

    ~VariableAsyncTaskThreadPool() {
        for (size_t i = 0; i < _threads.size(); ++i) {
            _channels[i]->terminate();
            _threads[i].join();
        }
    }

    void running(size_t i) {
        VariableAsyncTask task;
        while (_channels[i]->recv(task, -1)) {
            if (task.shard_lock == nullptr) {
                task.done();
            } else {
                core::shared_lock_guard<core::RWSpinLock> guard(*task.shard_lock);
                task.done();                
            }
        }
    }

    std::atomic<size_t> _jid = {0};
    std::vector<std::thread> _threads;
    std::vector<std::unique_ptr<core::RpcChannel<VariableAsyncTask>>> _channels;
};


}
}
}

#endif
