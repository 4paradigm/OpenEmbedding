#include <gtest/gtest.h>
#include "PmemEmbeddingTable.h"
#include <limits>

namespace paradigm4 {
namespace pico {
namespace embedding {

std::string pmem_pool_root_path = "/mnt/pmem0/tmp/exb_pmem_test";
TEST(PmemEmbeddingTable, MultipleGetAndSet) {
    PersistManager::singleton().initialize(pmem_pool_root_path);
    PmemEmbeddingArrayTable<uint64_t,double> pt(64, -1);
    PersistManager::singleton().dynamic_cache.set_cache_size(pt.cache_item_memory_cost());
    
    size_t total_items = 5;
    for (size_t j = 0; j < total_items; ++j){
        ASSERT_EQ(j, pt.work_id());
        ASSERT_EQ(nullptr, pt.get_value(j));
        double* value = pt.set_value(j);
        for(size_t i = 0; i < 64; ++i){
            value[i] = i + j;
        }
        const double* get = pt.get_value(j);
        for(size_t i = 0; i < 64; ++i){
            ASSERT_EQ(double(i + j), get[i]);
        }
        pt.next_work();
    }
    ASSERT_EQ(total_items, pt.work_id());
    
    for (size_t k = 0; k < total_items; ++k){
        const double* tmp = pt.get_value(k);
        for(size_t i = 0; i < 64; ++i) {
            ASSERT_EQ(double(i + k), tmp[i]);
        }
    }

    pt.start_commit_checkpoint();
    ASSERT_EQ(pt.checkpoints().size(), 0);
    pt.flush_committing_checkpoint();
    ASSERT_EQ(pt.checkpoints().size(), 1);

    for (size_t j = 0; j < total_items; ++j){
        const double* get = pt.get_value(j);
        for(size_t i = 0; i < 64; ++i){
            ASSERT_EQ(double(i + j), get[i]);
        }
        double* value = pt.set_value(j);
        for(size_t i = 0; i < 64; ++i){
            value[i] = i + j;
        }
        pt.next_work();
    }
    core::FileSystem::rmrf(pmem_pool_root_path);
}

TEST(PmemEmbeddingTable, SingleCheckpoint) {
    PersistManager::singleton().initialize(pmem_pool_root_path);
    PmemEmbeddingHashTable<uint64_t,double> pt(64, -1);
    PersistManager::singleton().dynamic_cache.set_cache_size(pt.cache_item_memory_cost() * 5);
    
// initial status    
    double* tmp;
    EXPECT_EQ(0, pt.work_id());
    EXPECT_EQ(0, pt.checkpoints().size());
    EXPECT_EQ(0, pt.num_pmem_items());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(0, pt.get_all_freespace_slots());
//////
// exp1: set 0,1,2,3,4 at each batch
    for(size_t j=0; j<5; ++j){
        EXPECT_EQ(j, pt.work_id());
        EXPECT_EQ(nullptr, pt.get_value(j));
        tmp = pt.set_value(j);
        for(size_t i=0; i<64; ++i){
            *tmp = double(i+j);
            ++tmp;
        }
        tmp = (double *)pt.get_value(j);
        for(size_t i=0; i<64; ++i){
            EXPECT_EQ(double(i+j), *tmp);
            ++tmp;
        }
        pt.next_work();
    }
    EXPECT_EQ(5, pt.work_id());
    pt.start_commit_checkpoint();  //_committing=5
    //status 1 expect: 
    // deque checkpoints: null
    // _committing = 5
    // _batch_id = 5
    // _free_list: null

    // Content: (dram,batch_id,key,value):
    // dram,0,0,0-63;
    // dram,1,1,1-64;
    // dram,2,2,2-65;
    // dram,3,3,3-66;
    // dram,4,4,4-67;
    EXPECT_EQ(0, pt.checkpoints().size());
    EXPECT_EQ(0, pt.num_pmem_items());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(0, pt.get_all_freespace_slots());
//////
// exp2: reset key 0,1,2,3,4's value at batch 5
    //test 1, set 0 at batch 5
    tmp = pt.set_value(0);
    for(size_t i=0; i<64; ++i){
        *tmp = (*tmp) + 10;
        ++tmp;
    }
    EXPECT_EQ(5, pt.work_id());
    EXPECT_EQ(0, pt.checkpoints().size());
    EXPECT_EQ(1, pt.num_pmem_items());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(1, pt.get_all_freespace_slots());
    //test 2, set 
    for(int k=1; k<5; ++k){
        tmp = pt.set_value(k);
        for(size_t i=0; i<64; ++i){
            *tmp = (*tmp) + 10;
            ++tmp;
        }
    }
    //status 2 expect: 
    // deque checkpoints: null
    // _committing = 5
    // _batch_id = 5
    // _free_list: null

    // Content: (dram,batch_id,key,value):
    // dram,5,0,10-73;
    // dram,5,1,11-74;
    // dram,5,2,12-75;
    // dram,5,3,13-76;
    // dram,5,4,14-77;
//////
// exp3: reset 0,1,2,3,4 at batch 6
    pt.next_work();
    EXPECT_EQ(6, pt.work_id());
    EXPECT_EQ(1, pt.checkpoints().size());
    EXPECT_EQ(5, pt.num_pmem_items());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(5, pt.get_all_freespace_slots());
    //test 1, set 0 at batch 6
    tmp = pt.set_value(0);
    for(size_t i=0; i<64; ++i){
        *tmp = (*tmp) + 10;
        ++tmp;
    }
    pt.next_work();
    EXPECT_EQ(7, pt.work_id());
    EXPECT_EQ(1, pt.checkpoints().size());
    EXPECT_EQ(5, pt.num_pmem_items());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(5, pt.get_all_freespace_slots());
    
    pt.start_commit_checkpoint(); //_committing=7
    //test 1, set 0 at batch 6
    for(size_t k=0; k<100; ++k){
        tmp = pt.set_value(0);
        for(size_t i=0; i<64; ++i){
            *tmp = (*tmp) + 10;
            ++tmp;
        }
        //pt.next_work();
    }
    pt.next_work();
    EXPECT_EQ(8, pt.work_id());
    EXPECT_EQ(1, pt.checkpoints().size());
    EXPECT_EQ(6, pt.num_pmem_items());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(6, pt.get_all_freespace_slots()); 
    //_free_space  batch_id=0 key=0,1,2,3,4; batch_id=1 key=0

    //test
    for(int k=5; k>=0; --k){
        tmp = pt.set_value(k);
        for(size_t i=0; i<64; ++i){
            *tmp = (*tmp) + 10;
            ++tmp;
        }
        pt.next_work();
    }
    EXPECT_EQ(14, pt.work_id());
    EXPECT_EQ(2, pt.checkpoints().size());
    EXPECT_EQ(12, pt.num_pmem_items());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(11, pt.get_all_freespace_slots());
    //_free_space  batch_id=0 key=0,1,2,3,4; batch_id=1 key=0; batch_id=2 key=0,1,2,3,4
    if(pt.checkpoints().size()>=2){
        pt.pop_checkpoint();
    }
    pt.next_work();
    EXPECT_EQ(15, pt.work_id());
    EXPECT_EQ(1, pt.checkpoints().size());
    EXPECT_EQ(12, pt.num_pmem_items());
    EXPECT_EQ(5, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(11, pt.get_all_freespace_slots());
///TODO:继续其他各种case

    core::FileSystem::rmrf(pmem_pool_root_path);
}



}
}
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
