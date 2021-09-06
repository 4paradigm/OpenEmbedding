#include <gtest/gtest.h>
#include "PersistentEmbeddingTable.h"
#include <limits>

namespace paradigm4 {
namespace pico {
namespace embedding {

TEST(PersistentEmbeddingTable, MultipleGetAndSet) {
    PersistentManager::singleton().set_cache_size(10);
    PersistentManager::singleton().set_pmem_pool_root_path("/mnt/pmem0/test");
    PersistentEmbeddingTable<uint64_t,double> pt(64, -1);
    
    size_t total_items = 5;
    for(size_t j = 0; j < total_items; ++j){
        ASSERT_EQ(j, pt.batch_id());
        ASSERT_EQ(nullptr, pt.get_value(j));
        double* value = pt.set_value(j);
        for(size_t i = 0; i < 64; ++i){
            value[i] = i + j;
        }
        const double* get = pt.get_value(j);
        for(size_t i = 0; i < 64; ++i){
            ASSERT_EQ(double(i + j), get[i]);
        }
        pt.next_batch();
    }
    ASSERT_EQ(total_items, pt.batch_id());
    
    for(size_t k = 0; k < total_items; ++k){
        const double* tmp = pt.get_value(k);
        for(size_t i = 0; i < 64; ++i) {
            ASSERT_EQ(double(i + k), tmp[i]);
        }
    }
}

TEST(PersistentEmbeddingTable, SingleCheckpoint) {  
    PersistentManager::singleton().set_cache_size(5);
    PersistentManager::singleton().set_pmem_pool_root_path("/mnt/pmem0/test");
    PersistentEmbeddingTable<uint64_t,double> pt(64, -1);
    
// initial status    
    double* tmp;
    EXPECT_EQ(0, pt.batch_id());
    EXPECT_EQ(0, pt.checkpoints().size());
    EXPECT_EQ(0, pt.get_pmem_vector_size());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(0, pt.get_all_freespace_slots());
//////
// exp1: set 0,1,2,3,4 at each batch
    for(size_t j=0; j<5; ++j){
        EXPECT_EQ(j, pt.batch_id());
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
        pt.next_batch();
    }
    EXPECT_EQ(5, pt.batch_id());
    pt.start_commit_checkpoint();
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
    EXPECT_EQ(0, pt.get_pmem_vector_size());
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
    EXPECT_EQ(5, pt.batch_id());
    EXPECT_EQ(0, pt.checkpoints().size());
    EXPECT_EQ(1, pt.get_pmem_vector_size());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(0, pt.get_all_freespace_slots());
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
    EXPECT_EQ(5, pt.batch_id());
    EXPECT_EQ(0, pt.checkpoints().size());
    EXPECT_EQ(5, pt.get_pmem_vector_size());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(0, pt.get_all_freespace_slots());
//////
// exp3: reset 0,1,2,3,4 at batch 6
    pt.next_batch();
    //test 1, set 0 at batch 6
    tmp = pt.set_value(0);
    for(size_t i=0; i<64; ++i){
        *tmp = (*tmp) + 10;
        ++tmp;
    }
    //EXPECT nothing change at pmem, only key 0 is still in dram 
    //?? 如果一个field的小心小于cache size，是不是永远无法完成一个transaction？ 
    // since我们结束一个transaction的方式是当有item因为cache满了被提出cache时候
    EXPECT_EQ(5, pt.batch_id());
    EXPECT_EQ(0, pt.checkpoints().size());
    EXPECT_EQ(5, pt.get_pmem_vector_size());
    EXPECT_EQ(0, pt.get_avaiable_freespace_slots());
    EXPECT_EQ(0, pt.get_all_freespace_slots());

///TODO:继续其他各种case
}



}
}
}

int main(int argc, char* argv[]) {
    testing::InitGoogleTest(&argc, argv);
    int ret = RUN_ALL_TESTS();
    return ret;
}
