#include <gtest/gtest.h>
#include <pico-core/MultiProcess.h>
#include <pico-core/ThreadGroup.h>

#include "c_api_test.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

void c_api_pull_push(int node_num, int word_num, int dim, bool sparse) {
    exb_string master_endpoint;
    exb_master* master = exb_master_start();
    exb_master_endpoint(master, &master_endpoint);
    {
        core::MultiProcess mp(node_num);
        exb_connection* connection = exb_connect("", master_endpoint.data);
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
        exb_connection* connection = exb_connect("", master_endpoint.data);
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
        core::FileSystem::rmr("ckt0");
        core::FileSystem::rmr("ckt1");
    }
    exb_master_join(master);
}

TEST(c_api, model_mix) {
    c_api_threads(1, 1, 1, 10, true);
    c_api_threads(1, 15, 5, 10, true);
    c_api_threads(2, 10, 5, 10, true);
    c_api_threads(3, 8, 5, 10, true);
    c_api_threads(4, 6, 5, 10, true);
    
    c_api_threads(1, 1, 1, 100, true);
    c_api_threads(2, 10, 5, 100, true);
}

TEST(c_api, model_shard_num) {
    c_api_threads(1, 3, 1, 10, true, 1);
    c_api_threads(1, 3, 5, 10, true, 3);
    c_api_threads(3, 3, 1, 10, true, 7);
    c_api_threads(5, 2, 1, 10, true, 111);
    c_api_threads(8, 2, 2, 10, true, 1024);
}

TEST(c_api, pull_push) {
    for (size_t i = 1; i < 10; ++i) {
        c_api_pull_push(i, 100, 128, false);
        c_api_pull_push(i, 100000, 1, false);
        c_api_pull_push(i, 100000, 8, false);
        c_api_pull_push(i, 100000, 1, true);
        c_api_pull_push(i, 100000, 16, true);
    }
}

TEST(c_api, one) {
    c_api_threads(1, 1, 1, 1000);
    c_api_threads(3, 1, 1, 1000);
    c_api_threads(5, 1, 1, 1000);
    c_api_threads(8, 1, 1, 1000);
}

TEST(c_api, trd) {
    c_api_threads(1, 3, 1, 300);
    c_api_threads(2, 3, 1, 300);
    c_api_threads(3, 3, 1, 300);
    c_api_threads(4, 3, 1, 300);
}

TEST(c_api, mix) {
    for (int node_num = 1; node_num < 9; ++node_num) {
        c_api_threads(node_num, 20, 5, 100);
    }
}

TEST(c_api, rep) {
    for (int i = 0; i < 3; ++i) {
        c_api_threads(2, 7, 2, 300);
        c_api_threads(3, 5, 2, 300);
        c_api_threads(4, 3, 3, 300);
    }
}

}
}
}


int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
