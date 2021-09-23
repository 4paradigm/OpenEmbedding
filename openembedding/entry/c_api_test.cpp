#include <gtest/gtest.h>
#include <pico-core/MultiProcess.h>
#include <pico-core/ThreadGroup.h>

#include "c_api_test.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

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
        c_api_threads(node_num, 20, 5, 100, false, node_num * node_num);
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
