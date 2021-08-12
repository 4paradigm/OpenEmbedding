#ifndef PARADIGM4_HYPEREMBEDDING_COMMON_THREAD_POOL_H
#define PARADIGM4_HYPEREMBEDDING_COMMON_THREAD_POOL_H

#include <atomic>
#include <vector>
#include <thread>

#include "../entry/c_api.h"

namespace paradigm4 {
namespace exb {

class exb_lock_guard {
public:
    exb_lock_guard(exb_mutex& mutex): _mutex(&mutex) {
        exb_mutex_lock(_mutex);
    }
    ~exb_lock_guard() {
        exb_mutex_unlock(_mutex);
    }
    exb_lock_guard(exb_lock_guard&&) = default;
    exb_lock_guard& operator=(exb_lock_guard&&) = default;
private:
    exb_mutex* _mutex;
};

class ThreadPool {
public:
    static ThreadPool& singleton() {
        static ThreadPool pool;
        return pool; 
    }

    template<class F>
    void submit(F job) {
        std::function<void()>* p = new std::function<void()>(std::move(job));
        exb_channel_write(_channels[_jid.fetch_add(std::memory_order_acq_rel) % _channels.size()], p);
    }

private:
    ThreadPool(size_t thread_num = std::thread::hardware_concurrency()): _threads(thread_num), _channels(thread_num) {
        for (size_t i = 0; i < _threads.size(); ++i) {
            _channels[i] = exb_channel_create();
            _threads[i] = std::thread(&ThreadPool::running, this, i);
        }
    }

    ~ThreadPool() {
        for (size_t i = 0; i < _threads.size(); ++i) {
            exb_channel_close(_channels[i]);
            _threads[i].join();
            exb_channel_delete(_channels[i]);
        }
    }

    void running(size_t i) {
        void* job;
        while (exb_channel_read(_channels[i], &job)) {
            std::function<void()>* p = static_cast<std::function<void()>*>(job);
            (*p)();
            delete p;
        }
    }

    std::atomic<size_t> _jid = {0};
    std::vector<std::thread> _threads;
    std::vector<exb_channel*> _channels;
};



}
}

#endif
