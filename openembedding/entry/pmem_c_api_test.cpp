#include "c_api_test.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

void test_loop_persist(exb_context* context, std::vector<exb_storage*>& storages,
      std::vector<TestVariable>& variables, int reps) {
    core::ThreadGroup threads(5);
    for (int i = 0; i < reps; ++i) {
        std::vector<core::AsyncReturn> asyncs;
        for (auto& var: variables) {
            asyncs.push_back(threads.async_exec([&var](int){ var.run_batch(true); }));
        }
        for (core::AsyncReturn& async: asyncs) {
            async.wait();
        }
        asyncs.clear();
        exb_barrier(context, "update_weights");
        for (exb_storage* storage: storages) {
            asyncs.push_back(threads.async_exec([storage](int){
                while (random(20)); // random update order
                exb_wait(exb_update_weights(storage));
            }));
        }

        asyncs.clear();
        if ((i + 1) % 10 == 0) {
            SLOG(INFO) << (i + 1);
        }
        if (exb_worker_rank(context) == 0 && exb_should_persist_model(context)) {
            exb_persist_model(context, "ckt_persist", "sign", 2);
            core::FileSystem::rmrf("ckt_persist");
        }
        exb_barrier(context, "ckt_persist");
    }
}

void c_api_persist_threads(int var_num, int var_type, int reps, int shard_num = -1) {
    int node_num = 1;
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
        exb_connection* connection = exb_connect(yaml_config, master_endpoint.data);
        exb_context* context = exb_context_initialize(connection, node_num);
        std::vector<exb_storage*> storages;
        std::vector<TestVariable> variables;
        int num = 0;
        exb_storage* storage = exb_create_storage(context);
        storages.push_back(storage);
        for (int i = 0; i < var_num; ++i) {
            variables.emplace_back(storage, node_num, 0, configs[i % var_type], i % 2 == 0);
            ++num;
            if (num > (int)storages.size()) {
                storage = exb_create_storage(context, shard_num);
                storages.push_back(storage);
            }
        }

        // train
        test_loop(context, storages, variables, reps, true);

        // persist
        std::vector<TestVariable> model;
        for (TestVariable& var: variables) {
            model.push_back(var);
        }
        exb_persist_model(context, "ckt0", "sign0", 0);
        
        // train
        test_loop(context, storages, variables, reps, true);
        
        // load persist load
        exb_restore_model(context, "ckt0");
        core::FileSystem::rmrf("ckt0");
        exb_persist_model(context, "ckt0", "sign0", 2);
        exb_restore_model(context, "ckt0");
        for (size_t i = 0; i < model.size(); ++i) {
            variables[i]._states = model[i]._states;
            variables[i]._checks = model[i]._checks;
        }

        // train and train auto persist
        test_loop(context, storages, variables, reps, true);
        test_loop_persist(context, storages, variables, reps);

        // test persist window 2
        model.clear();
        for (TestVariable& var: variables) {
            model.push_back(var);
        }
        core::FileSystem::rmrf("ckt0");
        exb_persist_model(context, "ckt0", "sign0", 2);
        test_loop(context, storages, variables, reps, true);
        exb_persist_model(context, "ckt1", "sign0", 2);
        core::FileSystem::rmrf("ckt1");
        test_loop(context, storages, variables, reps, true);
        exb_persist_model(context, "ckt1", "sign0", 2);
        core::FileSystem::rmrf("ckt1");
        exb_restore_model(context, "ckt0");
        for (size_t i = 0; i < model.size(); ++i) {
            variables[i]._states = model[i]._states;
            variables[i]._checks = model[i]._checks;
        }
        test_loop(context, storages, variables, reps, true);
        for (exb_storage* storage: storages) {
            exb_delete_storage(storage);
        }
        exb_context_finalize(context);
        exb_disconnect(connection);
    }
    core::FileSystem::rmrf("ckt0");
    exb_master_join(master);
}


void c_api_persist_threads_restore(int node_num, int var_num, int var_type, int reps, int shard_num = -1) {
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
    
    exb_string master_endpoint, master_endpoint2;
    exb_master* master = exb_master_start("127.0.0.1");
    exb_master* master2 = exb_master_start("127.0.0.1");
    exb_master_endpoint(master, &master_endpoint);
    exb_master_endpoint(master2, &master_endpoint2);
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

        // train
        test_loop_persist(context, storages, variables, reps);
        // persist
        std::vector<TestVariable> model;
        for (TestVariable& var: variables) {
            model.push_back(var);
        }
        
        if (exb_worker_rank(context) == 0) {
            exb_persist_model(context, "ckt0", "sign0", 2);
        }
        exb_barrier(context, "ckt_persist");
        test_loop(context, storages, variables, reps, true);
        if (exb_worker_rank(context) == 0) {
            exb_persist_model(context, "ckt1", "sign0", 2);
            core::FileSystem::rmrf("ckt1");
        }
        exb_barrier(context, "ckt_persist");
        test_loop(context, storages, variables, reps, true);
        if (exb_worker_rank(context) == 0) {
            exb_persist_model(context, "ckt1", "sign0", 2);
            core::FileSystem::rmrf("ckt1");
        }
        exb_barrier(context, "ckt_persist");


        // exit
        for (exb_storage* storage: storages) {
            exb_delete_storage(storage);
        }
        exb_context_finalize(context);
        exb_disconnect(connection);
        
        // restart and restore
        connection = exb_connect(yaml_config, master_endpoint2.data);
        context = exb_context_initialize(connection, node_num);
        storages.clear();
        variables.clear();
        num = 0;
        storage = exb_create_storage(context);
        storages.push_back(storage);
        for (int i = 0; i < var_num; ++i) {
            variables.emplace_back(storage, node_num, mp.process_index(), configs[i % var_type], i % 2 == 0);
            ++num;
            if (num > (int)storages.size()) {
                storage = exb_create_storage(context, shard_num);
                storages.push_back(storage);
            }
        }

        exb_restore_model(context, "ckt0");
        for (size_t i = 0; i < model.size(); ++i) {
            variables[i]._states = model[i]._states;
            variables[i]._checks = model[i]._checks;
        }
        test_loop(context, storages, variables, reps, true);
        for (exb_storage* storage: storages) {
            exb_delete_storage(storage);
        }
        exb_context_finalize(context);
        exb_disconnect(connection);
    }
    core::FileSystem::rmrf("ckt0");
    exb_master_join(master);
    exb_master_join(master2);

}

TEST(pmem_c_api, pull_push_one_node) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":1 }}";
    c_api_pull_push(1, 1 << 18, 1, true);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, pull_push_tree_node) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":1 }}";
    c_api_pull_push(3, 1 << 20, 8, true);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, one_node) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":1 }}";
    c_api_threads(1, 3, 1, 200);
    c_api_threads(1, 3, 3, 200);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, one_node_model) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":1 }}";
    c_api_threads(1, 3, 1, 10, true);
    c_api_threads(1, 3, 3, 10, true);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, one_node_persist) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":10 }}";
    c_api_persist_threads(3, 1, 10);
    c_api_persist_threads(3, 1, 100);
    c_api_persist_threads(3, 3, 100);
    c_api_persist_threads_restore(1, 5, 1, 100);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, one_node_small_cache) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":0 }}";
    c_api_persist_threads(3, 1, 10);
    c_api_persist_threads(3, 3, 10);
    c_api_persist_threads_restore(1, 5, 1, 100);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, three_node_persist) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":10 }}";
    c_api_persist_threads_restore(3, 5, 1, 10);
    c_api_persist_threads_restore(3, 5, 1, 100);
    c_api_persist_threads_restore(3, 5, 5, 100);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, tree_node) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":10 }}";
    c_api_threads(3, 5, 1, 500);
    c_api_threads(3, 5, 5, 500);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, tree_node_model) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":10 }}";
    c_api_threads(3, 5, 1, 50, true);
    c_api_threads(3, 5, 5, 50, true);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}


}
}
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
