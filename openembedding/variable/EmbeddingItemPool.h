#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_ITEM_POOL_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_ITEM_POOL_H

#include <queue>
#include <vector>
#include <pico-core/SpinLock.h>
#include <pico-core/pico_log.h>
#include <pico-core/pico_memory.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

class ItemPoolAllocator {
public:
    explicit ItemPoolAllocator(size_t item_size)
        : _item_size(aligned_size(item_size)),
          _block_size(aligned_size(64 * 1024, _item_size))  {}

    static size_t aligned_size(size_t base_size, size_t align = 0) {
        if (align == 0) {
            if (base_size >= 64) { // avx512 cache line
                align = 64;
            } else if (base_size >= 32) { // avx
                align = 32;
            } else if (base_size >= 16) { // sse
                align = 16;
            } else { 
                align = 8;
            }
        }
        return (base_size + align - 1) / align * align;
    }

    char* allocate() {
        if (_p == 0) {
            _pool.emplace_back(_block_size);
            SCHECK(reinterpret_cast<uintptr_t>(_pool.back().data()) % 8 == 0);
        }
        _p += _item_size;
        char* item = _pool.back().data() + _p;
        if (_p == _block_size) {
            _p = 0;
        }
        return item;
    }

    size_t item_size()const {
        return _item_size;
    }

    size_t block_size()const {
        return _block_size;
    }

private:
    std::deque<core::vector<char>> _pool;
    size_t _item_size = 0;
    size_t _block_size = 0;
    size_t _p = 0;
};

class ItemPoolCenter {
public:
    explicit ItemPoolCenter(size_t item_size)
        : _pool(item_size),
          _block_num_items((_pool.block_size() / _pool.item_size())) {}

    void allocate(std::deque<char*>& items) {
        core::lock_guard<core::RWSpinLock> guard(_lock);
        for (size_t i = 0; i < _block_num_items; ++i) {
            if (_free_items.empty()) {
                items.push_back(_pool.allocate());
            } else {
                items.push_back(_free_items.back());
                _free_items.pop_back();
            }
        }
    }

    void deallocate(std::deque<char*>& items) {
        core::lock_guard<core::RWSpinLock> guard(_lock);
        for (size_t i = 0; i < _block_num_items && !items.empty(); ++i) {
            _free_items.push_back(items.front());
            items.pop_front();
        }
    }

    size_t item_size()const {
        return _pool.item_size();
    }

    size_t block_num_items()const {
        return _block_num_items;
    }

private:
    ItemPoolAllocator _pool;
    size_t _block_num_items = 0;

    core::RWSpinLock _lock;
    std::vector<char*> _free_items;
};

class ItemPoolCenterManager {
public:
    static ItemPoolCenterManager& singleton() {
        static ItemPoolCenterManager manager;
        return manager;
    }

    ItemPoolCenter* center_of(size_t item_size) {
        item_size = ItemPoolAllocator::aligned_size(item_size);
        std::lock_guard<std::mutex> guard(_mutex);
        if (_pools[item_size] == nullptr) {
            _pools[item_size] = std::make_unique<ItemPoolCenter>(item_size);
        }
        return _pools[item_size].get();
    }

private:
    std::mutex _mutex;
    std::unordered_map<size_t, std::unique_ptr<ItemPoolCenter>> _pools; 
};


template<class Head, class T>
class EmbeddingItemPool {
public:
    struct Item: Head {
        T data[1];
    private:
        Item() = delete;
        ~Item() = delete;
        Item(const Item&) = delete;
        Item& operator=(const Item&) = delete;
    };

    static size_t item_size(size_t value_dim) {
        return ItemPoolAllocator::aligned_size(sizeof(Item) - sizeof(T) + value_dim * sizeof(T));
    }

    static void construct(Item* item, size_t value_dim) {
        new (item) Head();
        for (size_t i = 0; i < value_dim; ++i) {
            new (&item->data[i]) T();
        }
    }

    static void destruct(Item* item, size_t value_dim) {
        item->~Head();
        for (size_t i = 0; i < value_dim; ++i) {
            item->data[i].~T();
        }
    }

    EmbeddingItemPool(uint64_t value_dim)
        : _value_dim(value_dim),
          _center(ItemPoolCenterManager::singleton().center_of(item_size(value_dim))) {}

    ~EmbeddingItemPool() {
        while (!_free_items.empty()) {
            _center->deallocate(_free_items);
        }
    }

    Item* new_item() {
        if (_free_items.empty()) {
            _center->allocate(_free_items);
        }
        Item* item = reinterpret_cast<Item*>(_free_items.back());
        _free_items.pop_back();
        construct(item, _value_dim);
        return item;
    }

    void delete_item(Item* item) {
        destruct(item, _value_dim);
        if (_free_items.size() >= 2 * _center->block_num_items()) {
            _center->deallocate(_free_items);
        }
        _free_items.push_back(reinterpret_cast<char*>(item));
    }

    size_t value_dim()const {
        return _value_dim;
    }

private:
    size_t _value_dim;
    ItemPoolCenter* _center;
    std::deque<char*> _free_items;
};

}
}
}

#endif