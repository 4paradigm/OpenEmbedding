#include "c_api_test.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

TEST(pmem_c_api, pmem) {
    yaml_config = "{\"server\":{\"pmem_pool_root_path\":\"/mnt/pmem0/test\", \"cache_size\":1 }}";
    c_api_threads(1, 3, 1, 200);
    // c_api_threads(1, 3, 1, 10, true);
    
    // c_api_threads(3, 5, 2, 200);
    // c_api_threads(3, 5, 2, 10, true);
    yaml_config = "";
}

}
}
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
