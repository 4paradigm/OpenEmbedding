#include "c_api_test.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

TEST(pmem_c_api, one_node) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":1 }}";
    c_api_threads(1, 3, 1, 200);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, one_node_model) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":1 }}";
    c_api_threads(1, 3, 1, 10, true);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, tree_node) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":10 }}";
    c_api_threads(3, 5, 1, 500);
    yaml_config = "";
    core::FileSystem::rmrf("/mnt/pmem0/tmp/exb_pmem_test");
}

TEST(pmem_c_api, tree_node_model) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/tmp/exb_pmem_test\", \"cache_size\":1 }}";
    c_api_threads(3, 5, 2, 10, true);
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
