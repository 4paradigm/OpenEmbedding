
#include <gtest/gtest.h>
#include <pico-core/ThreadGroup.h>
#include <pico-core/MultiProcess.h>
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
        _sparse = sparse;
        _threads = std::make_shared<core::ThreadGroup>(5);
        _variable = exb_create_variable(storage, sparse ? -1 : _config.word_num * _node_num, 1);
        _variable_id = exb_variable_id(_variable);
        _checks.resize(_config.word_num, _config.init_value);
        _states.resize(_config.word_num, 3);
        for (int i = 0; i < _config.word_num; ++i) {
            _indices.push_back(i + _node_id * _config.word_num);
            _gradients.push_back(random(10));
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
        SCHECK(!error) << _variable_id << ' ' << _sparse;
        if (train) {
            exb_wait(exb_push_gradients(_variable, keys.data(), keys.size(), vals.data()));
            exb_wait(exb_push_gradients(_variable, keys.data(), keys.size(), vals.data()));
        }
    }

    bool _sparse = false;
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
        
        if (exb_worker_rank(context) == 0 && exb_should_persist_model(context)) {
            exb_persist_model(context, "ckt_persist", "sign", 2);
            core::FileSystem::rmrf("ckt_persist");
        }
        exb_barrier(context, "should_persist");
    }
}


const char* yaml_config = "";

void c_api_pull_push(int node_num, int word_num, int dim, bool sparse) {
    exb_string master_endpoint;
    exb_master* master = exb_master_start();
    exb_master_endpoint(master, &master_endpoint);
    {
        core::MultiProcess mp(node_num, "");
        exb_connection* connection = exb_connect(yaml_config, master_endpoint.data);
        exb_context* context = exb_context_initialize(connection, node_num);
        exb_storage* storage = exb_create_storage(context);
        exb_variable* variable = exb_create_variable(storage, sparse ? -1 : word_num, dim);
        
        exb_initializer* initializer = exb_create_initializer("constant");
        exb_set_initializer_property(initializer, "value", "100");
        exb_set_initializer(variable, initializer);
        
        std::vector<uint64_t> indices;
        std::vector<float> gradients;
        std::vector<float> answer;
        for (int i = mp.process_index(); i < word_num; i += node_num) {
            indices.push_back(i);
            for (int j = 0; j < dim; ++j) {
                gradients.push_back(i);
                answer.push_back(i + 100 + 10000);
            }

            int k = i * 107 % word_num;
            indices.push_back(k);
            for (int j = 0; j < dim; ++j) {
                gradients.push_back(k);
                answer.push_back(k + 100 + 10000);
            }
        }

        std::vector<float> weights(gradients.size());
        exb_pull_waiter* waiter1 = exb_pull_weights(variable, indices.data(), indices.size(), 1);
        exb_pull_waiter* waiter2 = exb_pull_weights(variable, indices.data(), indices.size(), 2);
        exb_pull_waiter* waiter = exb_pull_weights(variable, indices.data(), indices.size(), 0);
        bool ok = exb_pull_wait(waiter, indices.data(), indices.size(), weights.data());
        SCHECK(ok) << exb_last_error();

        
        exb_wait(exb_push_gradients(variable, indices.data(), indices.size(), gradients.data()));
        exb_wait(exb_push_gradients(variable, indices.data(), indices.size(), gradients.data()));
        exb_wait(exb_push_gradients(variable, indices.data(), indices.size(), gradients.data()));
        
        exb_optimizer* optimizer = exb_create_optimizer("test");
        exb_set_optimizer_property(optimizer, "learning_rate", "1");
        exb_set_optimizer(variable, optimizer);
        exb_barrier(context, "update_weights");
        exb_wait(exb_update_weights(storage));
        exb_pull_wait(waiter1, indices.data(), indices.size(), weights.data());
        EXPECT_EQ(weights, answer);
        SLOG(INFO) << "check1 !!!";

        answer = weights;
        for (float& val: answer) {
            val *= 2;
        }
        exb_wait(exb_push_gradients(variable, indices.data(), indices.size(), weights.data()));
        exb_barrier(context, "update_weights");
        exb_wait(exb_update_weights(storage));
        exb_pull_wait(waiter2, indices.data(), indices.size(), weights.data());
        EXPECT_EQ(weights, answer);

        exb_delete_storage(storage);
        exb_context_finalize(context);
        exb_disconnect(connection);
    }
    exb_master_join(master);
}


void c_api_threads(int node_num, int var_num, int var_type, int reps, bool load = false, int shard_num = -1) {
    std::vector<TestVariableConfig> configs;
    TestVariableConfig config;
    if (var_num == 1) {
        config.block_size = std::max(config.block_size, config.word_num / reps);
    }
    configs.push_back(config);
    config.init_value = 0;
    config.learning_rate = 2;
    configs.push_back(config);
    config.thread_num = 1;
    configs.push_back(config);
    config.init_value = 100;
    config.word_num = 1 << 10;
    config.block_size = 1 << 8;
    configs.push_back(config);
    config.learning_rate = 1;
    config.thread_num = 2;
    configs.push_back(config);
    
    exb_string master_endpoint;
    exb_master* master = exb_master_start("127.0.0.1");
    exb_master_endpoint(master, &master_endpoint);
    {
        core::MultiProcess mp(node_num, "");
        exb_connection* connection = exb_connect(yaml_config, master_endpoint.data);
        exb_context* context = exb_context_initialize(connection, node_num);
        std::vector<exb_storage*> storages;
        std::vector<TestVariable> variables;
        int num = 0;
        exb_storage* storage = exb_create_storage(context);
        storages.push_back(storage);
        for (int i = 0; i < var_num; ++i) {
            variables.emplace_back(storage, node_num, mp.process_index(), configs[i % var_type], i % 2 == 0);
            ++num;
            if (num > (int)storages.size()) {
                storage = exb_create_storage(context, shard_num);
                storages.push_back(storage);
            }
        }

        test_loop(context, storages, variables, reps, true);
        if (load) {
            if (mp.process_index() == 0) {
                exb_dump_model_include_optimizer(context, "ckt0", "ckt0");
                exb_dump_model(context, "ckt1", "ckt1");
                exb_create_model(connection, "ckt0", 1);
                exb_create_model(connection, "ckt1", std::min(3, node_num));
            }
            std::vector<TestVariable> model;
            for (TestVariable& var: variables) {
                model.push_back(var);
            }

            for (const char* sign: {"ckt0", "ckt1"}) {
                for (size_t i = 0; i < model.size(); ++i) {
                    int ms = 2;
                    do {
                        std::this_thread::sleep_for(std::chrono::milliseconds(ms *= 2));
                        model[i]._variable = exb_get_model_variable(
                            connection, sign, exb_variable_id(variables[i]._variable));
                    } while (model[i]._variable == nullptr);
                    model[i]._version = 0;
                }
                test_loop(context, storages, model, reps, false);
                for (const char* file: {"ckt1", "ckt0"}) {
                    test_loop(context, storages, variables, reps, true);
                    SLOG(INFO) << "loading " << sign << " " << file;
                    exb_load_model(context, file);
                    for (size_t i = 0; i < model.size(); ++i) {
                        if (strcmp(file, "ckt0") == 0) {
                            variables[i]._states = model[i]._states;
                        } else for (float& x: variables[i]._states) {
                            x = 3;
                        }
                        variables[i]._checks = model[i]._checks;
                    }
                }
                for (TestVariable& var: model) {
                    exb_release_model_variable(var._variable);
                }
            }
        }
        for (exb_storage* storage: storages) {
            exb_delete_storage(storage);
        }
        exb_context_finalize(context);
        exb_disconnect(connection);
    }
    if (load) {
        core::FileSystem::rmrf("ckt0");
        core::FileSystem::rmrf("ckt1");
    }
    exb_master_join(master);
}


}
}
}