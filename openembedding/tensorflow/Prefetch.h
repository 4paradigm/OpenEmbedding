#ifndef PARADIGM4_HYPEREMBEDDING_COMMON_PREFETCH_H
#define PARADIGM4_HYPEREMBEDDING_COMMON_PREFETCH_H

#include <unordered_map>

#include "ThreadPool.h"

namespace paradigm4 {
namespace exb {

class BatchIDTable {
public:
    int64_t pull_batch_id(int64_t key) {
        exb_lock_guard guard(_mutex);
        return _table[key];
    }

    void next_batch(int64_t key) {
        exb_lock_guard guard(_mutex);
        ++_table[key];
    }

private:
    exb_mutex _mutex;
    std::unordered_map<int64_t, int64_t> _table;
    int64_t _batch_id;
};

struct PrefetchKey {
    exb_variable* variable = nullptr;
    const uint64_t* indices = nullptr;
    size_t n = 0;
    int64_t batch_id = 0;
    size_t hash()const {
        // boost::hash_combine
        size_t hash = reinterpret_cast<size_t>(variable);
        hash ^= batch_id + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= n + 0x9e3779b9 +(hash << 6) + (hash >> 2);
        // sampling key
        for (size_t i = 0; i < 4 && i < n; ++i) {
            hash ^= indices[hash % n] + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
        return hash;
    }
};

struct PrefetchValue {
    uint64_t check = 0;
    exb_pull_waiter* waiter;
};

class PrefetchTable {
public:
    void push(const PrefetchKey& key, PrefetchValue&& value) {
        exb_lock_guard guard(_mutex);
        _table[key.variable].push_back(std::move(value));
    }

    bool pop(const PrefetchKey& key, PrefetchValue& value) {
        exb_lock_guard guard(_mutex);
        auto it = _table.find(key.variable);
        if (it == _table.end() || it->second.empty()) {
            return false;
        }
        value = std::move(it->second.front());
        it->second.pop_front();
        return true;
    }

    exb_mutex _mutex;
    std::unordered_map<exb_variable*, std::deque<PrefetchValue>> _table;
};

}
}

#endif
