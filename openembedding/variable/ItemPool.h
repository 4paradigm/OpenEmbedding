#ifndef PARADIGM4_HYPEREMBEDDING_ITEM_POOL_H
#define PARADIGM4_HYPEREMBEDDING_ITEM_POOL_H

#include <pico-core/pico_log.h>
#include <pico-ps/common/EasyHashMap.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

template<class Item>
class ItemAllocator {
public:
    explicit ItemAllocator(size_t item_size)
        : _item_size((item_size + 7 / 8) * 8),
          _block_size(_item_size * (63 * 1024 / _item_size + 1))  {}

    Item* allocate() {
        if (_p == 0) {
            _pool.emplace_back(_block_size);
            SCHECK(reinterpret_cast<uintptr_t>(_pool.back().data()) % 8 == 0);
        }
        _p += _item_size;
        Item* item = reinterpret_cast<Item*>(_pool.back().data() + _p);
        if (_p == _block_size) {
            _p = 0;
        }
        return item;
    }

    size_t item_size()const {
        return _item_size;
    }

    bool empty()const {
        return _pool.empty();
    }
    
    Item* back_item() {
        size_t i = (_p ? _p : _block_size) - _item_size;
        return reinterpret_cast<Item*>(_pool.back().data() + i);
    }

    void pop_back_item() {
        if (_p == 0) {
            _pool.pop_back();
            _p = _block_size;
        }
        _p -= _item_size;
    }
private:
    std::deque<core::vector<char>> _pool;
    size_t _item_size = 0;
    size_t _block_size = 0;
    size_t _p = 0;
};

template<class Head, class T>
class ItemPool {
public:
    struct Item: Head {
        T data[1];
    private:
        Item() = delete;
        ~Item() = delete;
        Item(const Item&) = delete;
        Item& operator=(const Item&) = delete;
    };

    explicit ItemPool(uint64_t value_dim)
        : _value_dim(value_dim), 
          _pool(sizeof(Item) - sizeof(T) + value_dim * sizeof(T)) {}

    ~ItemPool() {
        while (_pool.empty()) {
            destruct(_pool.back_item());
            _pool.pop_back_item();
        }
    }

    Item* new_item() {
        Item* item = _pool.allocate();
        contruct(item);
        return item;
    }

    void construct(Item* item) {
        new (item) Head();
        for (size_t i = 0; i < _value_dim; ++i) {
            new(item.data[i]) T();
        }
    }

    void destruct(Item* item) {
        item->~Head();
        for (size_t i = 0; i < _value_dim; ++i) {
            item.data[i].~T();
        }
    }

    size_t item_size() {
        return _pool.item_size();
    }

private:
    size_t _value_dim = 0;
    ItemAllocator<Item> _pool;
};

template<class T>
class ItemPool<void, T> {
public:
    struct Item {
        T data[1];
    private:
        Item() = delete;
        ~Item() = delete;
        Item(const Item&) = delete;
        Item& operator=(const Item&) = delete;
    };

    ~ItemPool() {
        while (_pool.empty()) {
            destruct(_pool.back_item());
            _pool.pop_back_item();
        }
    }

    explicit ItemPool(size_t value_dim)
        : _value_dim(value_dim),
          _pool(value_dim * sizeof(T)) {}

    Item* new_item() {
        Item* item = _pool.allocate();
        contruct(item);
        return item;
    }

    void construct(Item* item) {
        for (size_t i = 0; i < _value_dim; ++i) {
            new(item.data[i]) T();
        }
    }

    void destruct(Item* item) {
        for (size_t i = 0; i < _value_dim; ++i) {
            item.data[i].~T();
        }
    }

    size_t item_size() {
        return _pool.item_size();
    }
private:
    size_t _value_dim = 0;
    ItemAllocator<Item> _pool;
};

}
}
}

#endif