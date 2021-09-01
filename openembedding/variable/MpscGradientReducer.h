#ifndef PARADIGM4_HYPEREMBEDDING_MPSC_GRADIENT_REDUCER_H
#define PARADIGM4_HYPEREMBEDDING_MPSC_GRADIENT_REDUCER_H

#include <pico-ps/common/EasyHashMap.h>
#include "EmbeddingInitializer.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class Key, class T>
class MpscGradientReducer {
public:
    using key_type = Key;
    struct block_type {
        const key_type* keys;
        size_t n;
        const T* gradients;
        const uint64_t* counts;
    };

    MpscGradientReducer(size_t embedding_dim, key_type empty_key)
          : _embedding_dim(embedding_dim), _offsets(empty_key) {}

    // thread safe
    void push_gradients(block_type block) {
        _queue.push(std::move(block));
    }

    block_type reduce_gradients() {
        block_type block;
        while (_queue.pop(block)) {
            const T* grad = block.gradients;
            for (size_t i = 0; i < block.n; ++i) {
                key_type key = block.keys[i];
                if (_offsets.count(key)) {
                    size_t offset = _offsets.at(key);
                    T* sum = _gradients.data() + offset * _embedding_dim;
                    for (size_t j = 0; j < _embedding_dim; ++j) {
                        sum[j] += grad[j];
                    }
                    _counts[offset] += block.counts[i];
                } else {
                    _offsets.force_emplace(key, _offsets.size());
                    _keys.push_back(key);
                    _gradients.insert(_gradients.end(), grad, grad + _embedding_dim);
                    _counts.push_back(block.counts[i]);
                }
                grad += _embedding_dim;
            }
        }
        return {_keys.data(), _keys.size(), _gradients.data(), _counts.data()};
    }

    void clear() {
        _offsets.clear();
        _keys.clear();
        _gradients.clear();
        _counts.clear();
    }

private:
    size_t _embedding_dim = 0;
    core::MpscQueue<block_type> _queue;
    EasyHashMap<key_type, size_t> _offsets;
    core::vector<key_type> _keys;
    core::vector<T> _gradients;
    core::vector<uint64_t> _counts;
};

}
}
}

#endif