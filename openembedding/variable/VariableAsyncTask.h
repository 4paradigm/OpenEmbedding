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
    VariableAsyncTask(int thread_id, std::atomic<size_t>& counter, core::RWSpinLock& shard_lock)
        : _thread_id(thread_id), _counter(&counter), _shard_lock(&shard_lock) {}
    VariableAsyncTask(const VariableAsyncTask&) = delete;
    VariableAsyncTask(VariableAsyncTask&& other) = default;

    VariableAsyncTask& operator=(VariableAsyncTask other) {
        SCHECK(_done == nullptr);
        new (this) VariableAsyncTask(std::move(other));
        return *this;
    }
    
    ~VariableAsyncTask() {}

    operator bool() {
        return _done.operator bool();
    }

    int thread_id() {
        return _thread_id;
    }

    void done() {
        SCHECK(_done);
        if (_shard_lock) {
            core::lock_guard<core::RWSpinLock> guard(*_shard_lock);
            _done();
        } else {
            _done();
        }
        _entity = nullptr;
        _done = nullptr;
        _counter->fetch_sub(1, std::memory_order_relaxed);
    }

    void set_done(std::function<void()>&& done) {
        SCHECK(_done == nullptr && _counter);
        if (done) {
            _counter->fetch_add(1, std::memory_order_relaxed);
            _done = std::move(done);
        }
    }

    void hold_entity(const std::shared_ptr<void>& entity) {
        _entity = entity;
    }

private:
    size_t _thread_id = 0;
    std::atomic<size_t>* _counter = nullptr;
    core::RWSpinLock* _shard_lock = nullptr;
    std::shared_ptr<void> _entity = nullptr;
    std::function<void()> _done; 
};

class VariableAsyncTaskThreadPool {
public:
    static VariableAsyncTaskThreadPool& singleton() {
        static VariableAsyncTaskThreadPool pool;
        return pool; 
    }

    void submit(VariableAsyncTask&& async_task) {
        SCHECK(_initialized);
        core::lock_guard<core::RWSpinLock> guard(_lock);
        ++_num_tasks;
        _tasks.push_back(std::move(async_task));
        if (_tasks.size() >= _batch_num_tasks) {
            for (VariableAsyncTask& task: _tasks) {
                _channels[task.thread_id() % _threads.size()]->send(std::move(task));
            }
            _tasks.clear();
        }
    }

    // very illformed! TODO: remove
    void initialize_batch_task() {
        if (_batch_num_tasks == 0) {
            core::lock_guard<core::RWSpinLock> guard(_lock);
            if (_batch_num_tasks == 0) {
                SLOG(INFO) << "set batch num tasks " << _num_tasks;
                _batch_num_tasks = _num_tasks;
            }
        }
    }

    void initialize(size_t thread_num) {
        SCHECK(!_initialized);
        _initialized = true;
        _num_tasks = 0;
        _batch_num_tasks = 0;
        _threads.resize(thread_num);
        _channels.resize(thread_num);
        for (size_t i = 0; i < _threads.size(); ++i) {
            _channels[i] = std::make_unique<core::RpcChannel<VariableAsyncTask>>();
            _threads[i] = std::thread(&VariableAsyncTaskThreadPool::running, this, i);
        }
    }

    void finalize() {
        SCHECK(_initialized);
        for (size_t i = 0; i < _threads.size(); ++i) {
            _channels[i]->terminate();
            _threads[i].join();
        }
        _initialized = false;
    }

private:
    void running(size_t i) {
        VariableAsyncTask task;
        while (_channels[i]->recv(task, -1)) {
            // must finalize task in loop
            VariableAsyncTask done = std::move(task); 
            done.done();
        }
    }

    bool _initialized = false;
    std::vector<std::thread> _threads;
    std::vector<std::unique_ptr<core::RpcChannel<VariableAsyncTask>>> _channels;

    core::RWSpinLock _lock;
    size_t _num_tasks = 0;
    size_t _batch_num_tasks = 0;
    std::vector<VariableAsyncTask> _tasks;
};


}
}
}

#endif
