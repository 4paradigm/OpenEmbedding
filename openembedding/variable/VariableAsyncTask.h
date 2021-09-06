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
        for (int tests = 0; unlikely(_counter.load(std::memory_order_acquire)); ++tests) {
            if (tests < 128) {
                cpu_relax();
            } else {
                static constexpr std::chrono::microseconds us0{0};
                std::this_thread::sleep_for(us0);
            }
        }
    }

    VariableAsyncTask() {}
    VariableAsyncTask(int thread_id, std::atomic<size_t>& counter,
          core::RWSpinLock& storage_lock, core::RWSpinLock& shard_lock)
        : _thread_id(thread_id), _counter(&counter), 
          _storage_lock(&storage_lock), _shard_lock(&shard_lock) {}
    VariableAsyncTask(const VariableAsyncTask&) = delete;
    VariableAsyncTask(VariableAsyncTask&& other)=default;

    VariableAsyncTask& operator=(VariableAsyncTask other) {
        SCHECK(_done == nullptr);
        new (this) VariableAsyncTask(std::move(other));
        return *this;
    }
    
    ~VariableAsyncTask() {
        if (_done) {
            _done = nullptr;
            _storage_lock->unlock_shared();
            _counter->fetch_sub(1, std::memory_order_relaxed);
        }
    }

    operator bool() {
        return _done.operator bool();
    }

    int thread_id() {
        return _thread_id;
    }

    void done() {
        if (_shard_lock) {
            core::lock_guard<core::RWSpinLock> guard(*_shard_lock);
            _done();
        } else {
            _done();
        }
    }

    void set_done(std::function<void()>&& done) {
        SCHECK(_done == nullptr && _storage_lock && _counter);
        if (done) {
            _counter->fetch_add(1, std::memory_order_relaxed);
            _storage_lock->lock_shared();
            _done = std::move(done);
        }
    }

private:
    size_t _thread_id = 0;
    std::atomic<size_t>* _counter = nullptr;
    core::RWSpinLock* _storage_lock = nullptr;
    core::RWSpinLock* _shard_lock = nullptr;
    std::function<void()> _done; 
    
};

class VariableAsyncTaskThreadPool {
public:
    std::atomic<size_t> thread_id = {0};
    static VariableAsyncTaskThreadPool& singleton() {
        static VariableAsyncTaskThreadPool pool;
        return pool; 
    }

    void submit(VariableAsyncTask&& task) {
        _channels[task.thread_id() % _threads.size()]->send(std::move(task));
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
            VariableAsyncTask done = std::move(task);
            done.done();
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
