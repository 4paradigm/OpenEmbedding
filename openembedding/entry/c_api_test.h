
#include <pico-core/ThreadGroup.h>

#include "c_api.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

std::atomic<size_t> pull_err = {0};
std::atomic<size_t> pull_succ = {0};

inline int random(int n) {
    static thread_local std::random_device rd; 
    static thread_local std::mt19937 gen(rd());
    static thread_local std::uniform_int_distribution<int> dis(0, 1 << 30);
    return dis(gen) % n;
};

struct TestVariableConfig {
    int thread_num = 3;
    int word_num = 1 << 17;
    int block_size = 1 << 11;
    float init_value = 100.0;
    float learning_rate = 1.0;
    bool sparse = false;
};

class TestVariable {
public:
    TestVariable(exb_storage* storage, int node_num, int node_id, TestVariableConfig config, bool sparse)
        : _node_num(node_num), _node_id(node_id), _config(config) {
        _threads = std::make_shared<core::ThreadGroup>(5);
        _variable = exb_create_variable(storage, sparse ? -1 : _config.word_num * _node_num, 1);
        _variable_id = exb_variable_id(_variable);
        _checks.resize(_config.word_num, _config.init_value);
        _states.resize(_config.word_num, 3);
        for (int i = 0; i < _config.word_num; ++i) {
            _indices.push_back(i + _node_id * _config.word_num);
            _gradients.push_back(10000);
        }
        std::random_shuffle(_indices.begin(), _indices.end());
    }

    void run_batch(bool train) {
        int up_size = _config.block_size / 2;
        int batch_size = _config.block_size + up_size * (_config.thread_num - 1); 
        int key_offset = random(_config.word_num - batch_size);
        int val_offset = random(_config.word_num - batch_size);
        if (_batch_id == 0) {
            exb_initializer* initializer = exb_create_initializer("constant");
            exb_set_initializer_property(initializer, "value",
                  std::to_string(_config.init_value).c_str());
            exb_set_initializer(_variable, initializer);

            exb_optimizer* optimizer = exb_create_optimizer("test");
            exb_set_optimizer_property(optimizer, "learning_rate",
                  std::to_string(_config.learning_rate).c_str());
            exb_set_optimizer_property(optimizer, "flip", "10");
            exb_set_optimizer_property(optimizer, "init", "3");
            exb_set_optimizer(_variable, optimizer);
        };
        std::vector<std::thread> ths;
        std::vector<core::AsyncReturn> asyncs;
        for (int i = 0; i < _config.thread_num; ++i) {
            int block_key_offset = key_offset + i * up_size;
            int block_val_offset = val_offset + i * up_size;
            auto run_block_fn = [this, block_key_offset, block_val_offset, train](int){
                run_block(block_key_offset, block_val_offset, train);
            };
            if (random(2)) {
                ths.emplace_back(run_block_fn, 0);
            } else {
                asyncs.push_back(_threads->async_exec(run_block_fn));
            }
        }
        for (std::thread& th: ths) {
            th.join();
        }
        for (core::AsyncReturn& async: asyncs) {
            async.wait();
        }
        for (int i = 0; i < batch_size; ++i) {
            int index = _indices[key_offset + i] - _node_id * _config.word_num;
            if (train) {
                _states[index] = 10.0 - _states[index];
                _checks[index] += _states[index] + _gradients[val_offset + i] * _config.learning_rate;
            } 
        }
        ++_batch_id;
        if (train) {
            ++_version;
        }
    }

    void run_block(int key_offset, int val_offset, bool train) {
        static thread_local std::vector<uint64_t> keys;
        static thread_local std::vector<float> vals;
        keys.clear();
        vals.clear();
        for (int i = 0; i < _config.block_size; ++i) {
            keys.push_back(_indices[i + key_offset]);
            vals.push_back(_gradients[i + val_offset]);
        }
        for (int i = 0; i < _config.block_size * 3; ++i) {
            int j = random(_config.block_size);
            keys.push_back(_indices[j + key_offset]);
            vals.push_back(_gradients[j + val_offset]);
        }
        static thread_local std::vector<float> weights;
        weights.resize(keys.size());
        exb_pull_waiter* waiter = exb_pull_weights(_variable, keys.data(), keys.size(), _version);
        while (!exb_pull_wait(waiter, keys.data(), keys.size(), weights.data())) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            pull_err += 1;
            waiter = exb_pull_weights(_variable, keys.data(), keys.size(), _version);
        }
        pull_succ += 1;
        size_t error = 0;
        for (int i = 0; i < _config.block_size; ++i) {
            int index = keys[i] - _node_id * _config.word_num;
            if (weights[i] != _checks[index]) {
                error++;
                if (error < 100) {
                    SLOG(INFO) << weights[i] << ' ' << _checks[index]  << ' ' << i << ' ' << keys[i];
                }
            }
        }
        SCHECK(!error);
        if (train) {
            exb_wait(exb_push_gradients(_variable, keys.data(), keys.size(), vals.data()));
            exb_wait(exb_push_gradients(_variable, keys.data(), keys.size(), vals.data()));
        }
    }

    int _node_num = 0;
    int _node_id = 0;
    TestVariableConfig _config;
    uint32_t _variable_id = -1;
    exb_variable* _variable = nullptr;
    std::shared_ptr<core::ThreadGroup> _threads;
    std::vector<float> _checks;
    std::vector<float> _states;

    std::vector<uint64_t> _indices;
    std::vector<float> _gradients;
    int _batch_id = 0;
    int _version = 0;
};

void test_loop(exb_context* context, std::vector<exb_storage*>& storages,
      std::vector<TestVariable>& variables, int reps, bool train) {
    core::ThreadGroup threads(5);
    for (int i = 0; i < reps; ++i) {
        std::vector<core::AsyncReturn> asyncs;
        for (auto& var: variables) {
            asyncs.push_back(threads.async_exec([&var, train](int){ var.run_batch(train); }));
        }
        for (core::AsyncReturn& async: asyncs) {
            async.wait();
        }
        asyncs.clear();
        if (train) {
            exb_barrier(context, "update_weights");
            for (exb_storage* storage: storages) {
                asyncs.push_back(threads.async_exec([storage](int){
                    while (random(20)); // random update order
                    exb_wait(exb_update_weights(storage));
                }));
            }
        }
        asyncs.clear();
        if ((i + 1) % 10 == 0) {
            SLOG(INFO) << (i + 1);
        }
    }
}


}
}
}