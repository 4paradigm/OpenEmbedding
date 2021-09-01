#include <gtest/gtest.h>
#include <fstream>

#include "Connection.h"
#include "c_api_test.h"

#include <sys/prctl.h>

namespace paradigm4 {
namespace pico {
namespace embedding {

struct HAServer {
    void touch_file(const std::string& path, const std::string& content = "") {
        std::ofstream ofile(path);
        ofile << content;
        ofile.close();
    }

    HAServer(std::string master_endpoint, bool strict) {
        _pid = fork();
        _strict = strict;
        SCHECK(_pid >= 0);
        if (_pid == 0) {
            prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
            google::InstallFailureSignalHandler();
            EnvConfig env;
            core::Configure configure;
            env.load_yaml(configure, master_endpoint);
            RpcConnection conn(env);
            core::LogReporter::set_id("SERVER", conn.rpc()->global_rank());
            std::unique_ptr<ps::Server> server;
            server = conn.create_server();
            server->restore_storages(false);
            server->initialize();
            touch_file(std::to_string(::getpid()));
            server->finalize();
            exit(0);
        }
    }

    bool restored() {
        return core::FileSystem::exists(std::to_string(_pid));
    }

    HAServer(const HAServer&) = delete;
    HAServer& operator=(const HAServer&) = delete;
    HAServer(HAServer&& other) {
        _pid = other._pid;
        other._pid = 0;
    }
    HAServer& operator=(HAServer&& other) {
        _pid = other._pid;
        other._pid = 0;
        return *this;
    }
    ~HAServer() {
        if (_pid > 0) {
            SLOG(INFO) << "kill " << _pid;
            kill(_pid, SIGKILL);
            int status;
            _pid = waitpid(_pid, &status, 0);
            core::FileSystem::rmrf(std::to_string(_pid));
            SLOG(INFO) << "pid " << _pid << " exit with status: " << status;
            if (_strict) {
                // Should not fatal before kill.
                SCHECK(status == 9);
            }
            
        }
    }
     
    pid_t _pid;
    bool _strict;
};

void c_api_ha(bool strict, int var_num, int var_type, int reps, int shard_num=-1) {
    std::vector<TestVariableConfig> configs;
    TestVariableConfig config;
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

// ***
// train
// ***
    exb_string master_endpoint;
    exb_master* master = exb_master_start();
    exb_master_endpoint(master, &master_endpoint);
    exb_connection* connection = exb_connect("", master_endpoint.data);
    exb_context* context = exb_context_initialize(connection, 1);
    int num = 0;
    exb_storage* storage = exb_create_storage(context);
    std::vector<exb_storage*> storages;
    std::vector<TestVariable> variables;
    storages.push_back(storage);
    for (int i = 0; i < var_num; ++i) {
        variables.emplace_back(storage, 1, 0, configs[random(var_type)], i % 2 == 0);
        ++num;
        if (num > (int)storages.size()) {
            storage = exb_create_storage(context);
            storages.push_back(storage);
        }
    }
    test_loop(context, storages, variables, reps, true);
    exb_dump_model(context, "ckt1", "ckt1");
    for (exb_storage* storage: storages) {
        exb_delete_storage(storage);
    }
    storages.clear();
    exb_context_finalize(context);
    exb_disconnect(connection);
    exb_master_join(master);

// ***
// ha predictor
// ***
    const size_t SERVER_NUM = 5;
    const size_t REPLICA_NUM = 3;    
    const size_t KILL_ROUND = 10;
    const size_t KILL_INTERVAL = 5000;

    std::vector<std::unique_ptr<HAServer>> servers;
    master = exb_master_start("127.0.0.1");
    exb_master_endpoint(master, &master_endpoint);
    for (size_t i = 0; i < SERVER_NUM; ++i) {
        servers.emplace_back(std::make_unique<HAServer>(master_endpoint.data, strict));
    }

    EnvConfig env;
    // env.server.server_concurrency = 1;
    env.rpc.tcp.connect_timeout = 5;
    connection = exb_connect(env.to_yaml().dump().c_str(), master_endpoint.data);
    while(exb_running_server_count(connection) < 5) {
        SLOG(INFO) << "running server num " << exb_running_server_count(connection);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    exb_create_model(connection, "ckt1", REPLICA_NUM, shard_num);
    for (auto& variable: variables) {
        variable._version = 0;
        variable._variable = exb_get_model_variable(connection, "ckt1", variable._variable_id, 500);
    }

    pull_err.store(0);
    pull_succ.store(0);
    std::atomic<bool> finished = {false};
    std::thread th = std::thread([&]() {
        for (size_t round = 0; round < KILL_ROUND; ++round) {
            if (strict) {
                for (auto& server: servers) {
                    while (!server->restored()) {
                        SLOG(INFO) << "wait restore...";
                        std::this_thread::sleep_for(std::chrono::milliseconds(100));
                    }
                }
                SLOG(INFO) << "!!!!!!!!end restore!!!!!!!!!!";
            }
            if (random(3)) {
                // Wait server update context, test restore_storage_by_network.
                std::this_thread::sleep_for(std::chrono::milliseconds(random(KILL_INTERVAL)));
            }

            SLOG(INFO) << "!!!!!!!!begin kill!!!!!!!!!!";
            if (round % 3 == 0) {
                for (size_t i = 0; i < REPLICA_NUM - 1; ++i) {
                    servers[random(SERVER_NUM)].reset();
                }
            } else if (round % 3 == 1) {
                for (auto& server: servers) {
                    server.reset();
                }
            } else {
                for (size_t i = 0; i < REPLICA_NUM; ++i) {
                    servers[i].reset();
                }
            }
            SLOG(INFO) << "!!!!!!!!end kill!!!!!!!!!!";
            if (random(4) || round == KILL_ROUND - 1) {
                // wait discover dead node
                // The testing can be continued even not enough dead nodes are restored.
                std::this_thread::sleep_for(std::chrono::milliseconds(random(KILL_INTERVAL)));
            }

            SLOG(INFO) << "!!!!!!!!begin restore!!!!!!!!!!";
            for (auto& server: servers) {
                if (!server) {
                    server = std::make_unique<HAServer>(master_endpoint.data, strict);
                }
            }
        }
        finished = true;
    });

    storages.clear();
    while (!finished) {
        test_loop(context, storages, variables, reps, false);
    }
    th.join();
    // Test pulling after restore.
    size_t cur_succ = pull_succ.load();
    test_loop(context, storages, variables, reps, false);
    EXPECT_NE(cur_succ, pull_succ.load());
    for (auto& variable: variables) {
        exb_release_model_variable(variable._variable);
    }
    servers.clear();
    exb_disconnect(connection);
    exb_master_join(master);
    SLOG(INFO) << "pull err: " << pull_err.load();
    SLOG(INFO) << "pull succ: " << pull_succ.load();
    core::FileSystem::rmr("ckt1");
}

TEST(c_api, ha_1) {
    c_api_ha(true, 1, 1, 100);
    c_api_ha(false, 1, 1, 100);
}

TEST(c_api, ha_10) {
    c_api_ha(true, 10, 5, 100, -1);
    c_api_ha(false, 10, 5, 100, -1);
    c_api_ha(true, 10, 5, 100, 10);
}

}
}
}

int main(int argc, char* argv[]) {
    prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY, 0, 0, 0);
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
