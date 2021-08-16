#ifndef PARADIGM4_HYPEREMBEDDING_EMBEDDING_COMMUNICATION_H
#define PARADIGM4_HYPEREMBEDDING_EMBEDDING_COMMUNICATION_H

#include <pico-core/RpcService.h>
#include "ObjectPool.h"

namespace paradigm4 {
namespace pico {
namespace embedding {

// Here comm_rank is in [0, comm_size], and the corresponding rpc global_rank can be any value.
class Communication {
    enum reqs {
        BOARD_CAST = 0,
        LOAD_MODEL_SIGN = 1,
        BARRIER = 2,
    };
public:
    Communication();
    Communication(core::RpcService* rpc, int32_t comm_size, std::string rpc_name = "sync_runner_rpc_api");

    ~Communication();

    int32_t comm_rank() {
        return _comm_rank;
    }

    int32_t comm_size() {
        return _comm_size;
    }

    comm_rank_t barrier(std::string name);

    template<class Fn>
    auto sync_bcast(const std::string& name, Fn fn) {
        comm_rank_t from = barrier(name);
        decltype(fn()) result;
        if (_comm_rank == from) {
            result = fn();
        }
        boardcast(name, result, from);
        return result;
    }

    template<class T>
    void boardcast(std::string name, T& value, comm_rank_t from) {
        core::BinaryArchive ar;
        ar << value;
        inner_boardcast(name, ar, from);
        ar >> value;
    }

    bool load_model_sign(const std::string& model_sign);

private:
    void serving();

    void inner_boardcast(std::string name, core::BinaryArchive& ar, comm_rank_t from);

    core::RpcService* _rpc = nullptr;
    int32_t _comm_size = 0;
    std::string _rpc_name;
        
    int32_t _comm_rank = 0;
   
    std::string _model_sign;
    std::thread _serving_th;
    std::unique_ptr<core::RpcServer> _rpc_server;
    std::unique_ptr<core::RpcClient> _rpc_client;
    ObjectPool<std::shared_ptr<core::Dealer>> _dealer;
    std::unordered_map<std::string, std::vector<core::RpcRequest>> _reqs;
    std::unordered_map<std::string, std::vector<core::RpcRequest>> _barriers;
};

}
}
}

#endif