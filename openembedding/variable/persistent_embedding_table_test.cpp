#include <gtest/gtest.h>
#include "PersistentEmbeddingTable.h"
#include <limits>

namespace paradigm4 {
namespace pico {
namespace embedding {

TEST(PersistentEmbeddingTable, MultipleGetAndSet) {
    PersistentManager::singleton().set_cache_size(10);
    PersistentEmbeddingTable<uint64_t,double> pt(64, -1);
    core::Configure config;
    PersistentManager::singleton().set_pmem_pool_root_path("/mnt/pmem0/test");
    std::string pmem_pool_path = PersistentManager::singleton().new_pmem_pool_path();
    SAVE_CONFIG(config, pmem_pool_path);
    pt.load_config(config);

    const double* value;
    double* tmp;
    for(size_t j=0; j<10000; ++j){
        EXPECT_EQ(j, pt.batch_id());
        EXPECT_EQ(nullptr, pt.get_value(j));
        tmp = pt.set_value(j);
        for(size_t i=0; i<64; ++i){
            *tmp = double(i);
            ++tmp;
        }
        value = pt.get_value(j);
        tmp = (double *)value;
        for(size_t i=0; i<64; ++i){
            EXPECT_EQ(double(i), *tmp);
            ++tmp;
        }
        pt.next_batch();
    }
    EXPECT_EQ(10000, pt.batch_id());
    for(size_t j=0; j<10000; ++j){
        value = pt.get_value(j);
        tmp = (double *)value;
        for(size_t i=0; i<64; ++i){
            EXPECT_EQ(double(i), *tmp);
            ++tmp;
        }
        //pt.next_batch();
    }
}

/*TEST(PersistentMemoryPool, GetAndPutFromPMem) {
    PersistentMemoryPool _pmem_pool(64*sizeof(double), "/mnt/pmem0/test", 300);
    auto fs = _pmem_pool.debug_get_free_space();
    for(size_t k=0; k<10; ++k){
        for(size_t j=0; j<100; ++j){
            PersistentItem* pi = _pmem_pool.acquire(j);
            pi->version = j;
            _pmem_pool.flush(pi);
            _pmem_pool.release(pi);
        }
        EXPECT_EQ(k+1, fs.front().id);
        EXPECT_EQ(100, fs.front().free_items.size())

        _pmem_pool.pmem_push_checkpoint(k);
        EXPECT_EQ(k, _pmem_pool.get_checkpoint_batch_id());
        if(k>0){
            _pmem_pool.pmem_pop_checkpoint();
        }
    }
}*/


}
}
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
