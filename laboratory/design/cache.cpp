
class PmemEmbeddingVariable: EmbeddingVariable {
public:
    virtual ~PmemEmbeddingVariable() {}
    virtual void load_config(const core::Configure& config) {
        std::string initializer;
        size_t n = num_indices();
        LOAD_CONFIG(config, initializer);
        // 设置initializer会重置所有weights和slots
        // 以第一次的设置的initializer为准，忽略initializer重复设置。
        if (!_initializer) {
            _initializer = Factory<EmbeddingInitializer<T>>::singleton().create(initializer, config[initializer]);
            SCHECK(_initializer) << "create initializer " << initializer 
                  << " with datatype " << DataType::from<T>().to_string() << " failed";
        }

        std::string optimizer;
        LOAD_CONFIG(config, optimizer);
        if (_optimizer && optimizer == _optimizer->category()) {
            // 支持修改learning_rate等
            _optimizer->load_config(config[optimizer]);
        } else if (!optimizer.empty()) {
            SCHECK(_optimizer == nullptr) << "not support change optimizer.";
            _optimizer = Factory<EmbeddingOptimizer<T>>::singleton().create(optimizer, config[optimizer]);
            SCHECK(_optimizer) << "create optimizer " << optimizer 
                  << " with datatype " << DataType::from<T>().to_string() << " failed";
            _state_dim = _optimizer->state_dim(_embedding_dim);
        }
    }

    virtual void vocabulary_resize(uint64_t vocabulary_size) {
        _vocabulary_size = vocabulary_size;
    }

    virtual void get_weights(const uint64_t* indices, size_t n, char* weights, RWSpinLockGuard& guard, version) {
        read_only_get_weights(indices, n, weights, version);
    }

    virtual void read_only_get_weights(const uint64_t* indices, size_t n, char* weights, version) {
        /// TODO:
        /// TODO: cache initialize
        /// TODO: pmem initialize
    }

    virtual void update_weights() {
        _offsets.clear();
        _gradients.clear();
        _counts.clear();
        core::vector<GradientBlock> blocks;
        {
            std::lock_guard<core::RWSpinLock> guard(_block_lock);
            blocks = _blocks;
            _blocks.clear();
        }
        for (GradientBlock block: blocks) {
            const T* grad = block.gradients;
            for (size_t i = 0; i < block.n; ++i) {
                uint64_t index = block.indices[i];
                if (_offsets.count(index)) {
                    size_t offset = _offsets.at(index);
                    T* sum = _gradients.data() + offset * _embedding_dim;
                    for (size_t j = 0; j < _embedding_dim; ++j) {
                        sum[j] += grad[j];
                    }
                    _counts[offset] += block.counts[i];
                } else {
                    _offsets.force_emplace(index, _offsets.size());
                    _gradients.insert(_gradients.end(), grad, grad + _embedding_dim);
                    _counts.push_back(block.counts[i]);
                }
                grad += _embedding_dim;
            }
        }

        /// TODO:
    }
};

struct Item {
    size_t version;
    size_t key;
    float value[embedding_dim + state_dim];
    // slot_dim = k * embedding_dim + b;
};

struct Item {
    size_t version;
    size_t key;
    float value[];
    // slot_dim = k * embedding_dim + b;
};

struct ItemView {
    char* buffer;
    size_t embedding_dim;
    size_t state_dim;

    size_t& version() {
        return *(size_t*)(buffer);
    }
    size_t& key() {
        return *(size_t*)(buffer + 8);
    }
    float* weights() {
        return (float*)(buffer + 16);
    }
    float* state() {
        return (float*)(buffer + 16) + embedding_dim;
    }
    size_t size() {
        /// TODO: 64;
        return sizeof(float) * (embedding_dim + state_dim) + 16;
    }
};

class Array {
    ItemView operator[](size_t i) {
        ItemView a = temp;
        a.buffer = buffer + temp.size() * i;
        return a;
    }
    ItemView temp;
    void* buffer;
};

template<class T>
struct AsyncFlush {
    PmemEmbeddingVariable<T>* var;
    std::vector<ItemView> items; 
    void operator()() {
        core::lock_guard<core::RWSpinLock> guard(var->g);
        // flush cache
        // remove cache
    }
};

class AsyncFlushThreadPool : public NoncopyableObject {
public:
    AsyncFlushThreadPool() : AsyncFlushThreadPool(std::thread::hardware_concurrency()) {
    }
    explicit AsyncFlushThreadPool(size_t thread_num) {
        _threads.resize(thread_num);
        for (size_t thread_id = 0; thread_id < thread_num; ++thread_id) {
            _threads[thread_id] = std::thread(&ThreadGroup::run, this, thread_id);
        }
    }
    ~AsyncFlushThreadPool() {
        if (!_exec_chan.closed()) {
            _exec_chan.close();
        }
        join();
    }
    bool async_exec(int version, AsyncFlush&& task) {
        core::lock_guard<core::RWSpinLock> guard(_lock);
        if (version == 0) {
            _count += 1;
        }
        if (version == 2 && _execute_batch == 1) {
            _execute_batch = _count;
            _exec_chan.set_capacity(_execute_batch);
        }
        _que.push_back(std::move(task));
        if (_que.size() >= _execute_batch) {
            for (AsyncFlush&& block: _que) {
                _exec_chan.write(block);
            }
            _que.clear();
        }
    }
    void join() {
        for (auto& thrd : _threads) {
            if (thrd.joinable()) {
                thrd.join();
            }
        }
    }
    void set_execute_batch(int value) {
        _execute_batch = value;
    }
private:
    void run(int thread_id) {
        AsyncFlush task;
        while (_exec_chan.read(task)) {
            task();
        }
    }

private:
    std::atomic<size_t> _que_size;
    core::RWSpinLock _lock;
    size_t _count = 0;
    size_t _execute_batch = 1;
    std::vector<AsyncFlush> _que; 

    std::vector<std::thread> _threads;
    core::Channel<AsyncFlush> _exec_chan;
};

